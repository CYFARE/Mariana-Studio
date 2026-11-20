"""Download Hugging Face datasets as ready-to-train JSON Lines.

The dataset viewer is used instead of the Parquet files: 100 rows at a time,
written atomically to ``datasets/<org>--<name>/data.jsonl`` with a small
``dataset.json`` record.  The existing chat-row normaliser understands the
common instruction, conversation, question/answer and text layouts, so the
downloaded folder can be handed to ``FinetuneService.plan`` unchanged.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..errors import BadRequestError, ConflictError
from ..models.downloads import validate_repo
from . import discovery

DATASETS_FOLDER = "datasets"
DEFAULT_ROWS = 5000
MAX_ROWS = 100_000
MAX_DATASET_BYTES = 2 << 30  # one split can never fill the workspace by accident


def dataset_folder(root: Path, repo: str) -> Path:
    """Where one Hub dataset is stored inside the workspace."""
    return Path(root).resolve() / DATASETS_FOLDER / repo.replace("/", "--")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


class DatasetManager:
    """Start and track dataset downloads as ordinary Studio jobs."""

    def __init__(self, root: Path, registry):
        self.root = Path(root).resolve()
        self.registry = registry

    # -- listing ---------------------------------------------------------
    def installed(self) -> list[dict]:
        """Every downloaded dataset folder with a readable ``data.jsonl``."""
        base = self.root / DATASETS_FOLDER
        if not base.is_dir():
            return []
        try:
            folders = sorted(base.iterdir())
        except OSError:
            return []
        entries = []
        for folder in folders:
            data = folder / "data.jsonl"
            if not folder.is_dir() or not data.is_file():
                continue
            metadata = _read_json(folder / "dataset.json")
            repo = str(metadata.get("repo") or folder.name.replace("--", "/"))
            try:
                stat = data.stat()
                created = float(metadata.get("created") or stat.st_mtime)
                rows = int(metadata.get("rows") or 0)
            except (OSError, TypeError, ValueError):
                continue
            entries.append({
                "repo": repo, "folder": str(folder.resolve()),
                "config": str(metadata.get("config") or ""), "split": str(metadata.get("split") or ""),
                "rows": rows, "bytes": stat.st_size,
                "created": created,
                "url": discovery.dataset_url(repo),
            })
        entries.sort(key=lambda item: item["created"], reverse=True)
        return entries

    # -- downloads -------------------------------------------------------
    def start_huggingface(self, repo: str, config: str = "", split: str = "",
                          max_rows: int = DEFAULT_ROWS) -> dict:
        """Download one split through the dataset viewer as a background job."""
        repo = validate_repo(repo)
        config = discovery.validate_split(config, "Configuration")
        split = discovery.validate_split(split, "Split")
        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError):
            raise BadRequestError("max_rows must be a number") from None
        if not discovery.ROWS_PAGE <= max_rows <= MAX_ROWS:
            raise BadRequestError(
                f"max_rows must be between {discovery.ROWS_PAGE} and {MAX_ROWS}"
            )
        for active in self.registry.active("dataset"):
            if active["meta"].get("repo") == repo:
                raise ConflictError("This dataset is already downloading")
        folder = dataset_folder(self.root, repo)
        job = self.registry.create(
            "dataset", f"Download data · {repo}",
            meta={
                "repo": repo, "source": "huggingface", "view": "finetune",
                "target": str(folder.relative_to(self.root)), "config": config, "split": split,
            },
        )
        job.update(unit="rows", stage="Queued")
        self.registry.start(job, lambda item: self._run(item, repo, config, split, max_rows, folder))
        return job.snapshot()

    def _run(self, job, repo: str, config: str, split: str, max_rows: int, folder: Path) -> dict:
        if not config or not split:
            info = discovery.dataset_info(repo)
            choice = discovery.select_split(info["splits"], config, split)
            config, split = choice["config"], choice["split"]
            job.update(config=config, split=split)
            job.log(f"Using {config}/{split}")
        job.update(stage=f"Reading {config}/{split}")
        folder.mkdir(parents=True, exist_ok=True)
        part = folder / "data.jsonl.part"
        final = folder / "data.jsonl"
        rows_written = offset = 0
        written_bytes = 0
        total = 0
        try:
            with part.open("w", encoding="utf-8", newline="\n") as handle:
                while rows_written < max_rows:
                    job.check_cancelled()
                    page = discovery.dataset_rows(
                        repo, config, split, offset,
                        min(discovery.ROWS_PAGE, max_rows - rows_written),
                    )
                    rows = page.get("rows")
                    if not isinstance(rows, list) or not rows:
                        break
                    if not total:
                        total = int(page.get("num_rows_total") or page.get("num_rows") or 0)
                        job.update(total=min(total, max_rows) if total else max_rows)
                    for entry in rows:
                        row = entry.get("row") if isinstance(entry, dict) else None
                        if not isinstance(row, dict):
                            continue
                        line = json.dumps(row, ensure_ascii=False) + "\n"
                        written_bytes += len(line.encode("utf-8"))
                        if written_bytes > MAX_DATASET_BYTES:
                            raise BadRequestError(
                                "This download grew beyond 2 GB; choose fewer rows or a smaller split."
                            )
                        handle.write(line)
                        rows_written += 1
                        if rows_written >= max_rows:
                            break
                    offset += len(rows)
                    job.update(current=rows_written, stage=f"Saved {rows_written:,} rows")
                    if total and offset >= total:
                        break
            if not rows_written:
                raise BadRequestError(
                    "This split returned no rows. Choose another configuration or split, "
                    "or download the dataset files and point the data folder at them."
                )
            os.replace(part, final)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        stat = final.stat()
        metadata = {
            "repo": repo, "config": config, "split": split, "rows": rows_written,
            "bytes": stat.st_size, "created": time.time(), "source": "huggingface",
            "url": discovery.dataset_url(repo),
        }
        temporary = folder / "dataset.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, folder / "dataset.json")
        job.log(f"Saved {rows_written:,} rows to {folder}")
        return {
            "repo": repo, "folder": str(folder.resolve()), "config": config,
            "split": split, "rows": rows_written, "bytes": stat.st_size,
        }
