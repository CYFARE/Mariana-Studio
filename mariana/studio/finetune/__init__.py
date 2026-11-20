"""Fine-tuning engines: Mariana checkpoints natively, any Hugging Face model via ``trainer``.

``FinetuneService`` is the single entry point: it resolves a model id through
the gateway, picks the engine, returns plans in one shape for both, and fronts
the Discovery centre that downloads datasets and third-party bases.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import UnavailableError
from ..models.catalog import TRANSFORMERS_HINT, ModelError, has_module
from ..validation import number
from . import datasets, discovery, hf, native

__all__ = ["DatasetManager", "FinetuneService", "hf", "native"]

DatasetManager = datasets.DatasetManager


class FinetuneService:
    """Inspect, plan, start and discover fine-tunes for any model in the catalog."""

    def __init__(self, root: Path, gateway, training, registry=None):
        self.root = Path(root).resolve()
        self.gateway = gateway
        self.training = training
        self.datasets = DatasetManager(self.root, registry) if registry is not None else None

    def _entry(self, model_id: str | None) -> dict:
        if not model_id:
            raise ModelError("Select a base model to fine-tune")
        entry = self.gateway.get(str(model_id))
        if not entry["capabilities"].get("finetune"):
            raise ModelError(entry.get("finetune_reason") or "This model cannot be fine-tuned")
        return entry

    def bases(self, force: bool = False) -> dict:
        """Every model, split into tunable bases and those that are not (with the reason)."""
        tunable, other = [], []
        for entry in self.gateway.catalog(force):
            if entry["kind"] == "mariana-ncm-tokenizer":
                continue
            item = {
                "id": entry["id"], "name": entry["name"], "kind": entry["kind"],
                "kind_label": entry["kind_label"], "source_label": entry["source_label"],
                "bytes": entry["bytes"], "architecture": entry["detail"].get("architecture"),
                "parameters": entry["detail"].get("parameters"),
                "engine": "mariana" if entry["kind"].startswith("mariana") else "hf",
                "reason": entry.get("finetune_reason", ""),
                "repo": entry["detail"].get("repo") or entry["detail"].get("base_model") or "",
            }
            (tunable if entry["capabilities"].get("finetune") else other).append(item)
        return {
            "tunable": tunable, "other": other,
            "transformers_installed": has_module("transformers"),
            "transformers_hint": TRANSFORMERS_HINT,
        }

    def inspect(self, model_id: str | None) -> dict:
        entry = self._entry(model_id)
        if entry["kind"].startswith("mariana"):
            return native.inspect(self.root, entry["absolute_path"])
        return hf.inspect(entry)

    def plan(self, config: dict) -> dict:
        entry = self._entry(config.get("base"))
        if entry["kind"].startswith("mariana"):
            return native.plan(self.root, {**config, "base": entry["absolute_path"]})
        return hf.plan(self.root, entry, config)

    def start(self, config: dict) -> dict:
        """Validate, then launch through the shared training manager."""
        planned = self.plan(config)
        if planned["engine"] == "hf":
            if not planned["ready"]:
                raise ModelError(planned["base"]["requirement"])
            return self.training.start_external(planned["config"])
        job_config = dict(planned["config"])
        job_config["smart"] = False
        return self.training.start(job_config)

    # -- discovery centre -------------------------------------------------
    def discover(self, platform: str = "datasets", query: str = "",
                 limit: int = 24) -> dict:
        """Search public hubs and mark results already present in the workspace."""
        results, warnings = discovery.search(platform, query, limit)
        installed = self.datasets.installed() if self.datasets is not None else []
        local = {item["repo"]: item for item in installed}
        catalog = {entry["id"]: entry for entry in self.gateway.catalog()}
        self._annotate(results, local, catalog)
        featured = discovery.featured()
        self._annotate(featured["datasets"] + featured["models"], local, catalog)
        return {
            "results": results, "warnings": warnings, "installed": installed,
            "featured": featured,
            "platforms": [{"id": key, "label": label} for key, label in discovery.PLATFORMS.items()],
        }

    def _annotate(self, items: list[dict], installed: dict[str, dict],
                  catalog: dict[str, dict]) -> None:
        """Add workspace state to remote results; it is what makes them actionable."""
        for item in items:
            if item["kind"] == "dataset":
                if item["platform"] != "huggingface":
                    continue  # only Hub datasets have a saved-folder mapping
                saved = installed.get(item["repo"])
                if saved:
                    item["local_folder"] = saved["folder"]
                    item["rows"] = saved["rows"]
                continue
            if item["kind"] != "model" or item["platform"] != "huggingface":
                continue
            entry = catalog.get("models/" + item["repo"].replace("/", "--"))
            if entry is None:
                continue
            item["local_id"] = entry["id"]
            item["local"] = True
            item["finetune"] = bool(entry["capabilities"].get("finetune"))
            item["reason"] = entry.get("finetune_reason", "")

    def dataset_info(self, repo: str) -> dict:
        """Splits and card metadata for a dataset, plus a saved copy if one exists."""
        info = discovery.dataset_info(repo)
        if self.datasets is not None:
            info["local"] = next(
                (item for item in self.datasets.installed() if item["repo"] == info["repo"]),
                None,
            )
        return info

    def start_dataset(self, config: dict) -> dict:
        """Download one dataset split as JSON Lines and make it the data folder."""
        if self.datasets is None:
            raise UnavailableError("Dataset downloads need the Studio job runner; restart Studio.")
        return self.datasets.start_huggingface(
            str(config.get("repo") or ""),
            str(config.get("config") or ""),
            str(config.get("split") or ""),
            number(config, "max_rows", datasets.DEFAULT_ROWS, discovery.ROWS_PAGE, datasets.MAX_ROWS),
        )
