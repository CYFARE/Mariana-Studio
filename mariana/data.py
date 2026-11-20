"""Byte-level data pipeline (platform-independent, no tokenizer deps).

Training normally takes an explicit corpus path. Without one, a small built-in
demo text keeps smoke tests offline and self-contained.
"""

from __future__ import annotations

import os

import numpy as np
import torch

# A short, self-contained text for smoke tests. Real training should pass --data.
DEMO_TEXT = (
    "Mariana Trench, the deepest known point of Earth's oceans, "
    "lies in the western Pacific Ocean near Guam. Its floor reaches "
    "nearly eleven thousand meters below the surface. The pressure "
    "there is more than a thousand times that at sea level, yet life "
    "persists. Microbes, amphipods, and even strange translucent "
    "snailfish have been filmed in the hadal zone. Expeditions use "
    "specialized submersibles with titanium hulls and thick acrylic "
    "viewports to withstand the crushing weight. Light never reaches "
    "the bottom, so creatures rely on chemosynthesis near seeps and "
    "falls of organic matter from above. The water is near freezing, "
    "and the sediment is soft with fine ooze accumulated over millions "
    "of years. Studying these extremes teaches us about the limits of "
    "life and informs the search for organisms on icy moons. "
    "Engineers draw inspiration from deep-sea animals to design better "
    "materials and robots. Every dive reveals new species and reminds "
    "us how much of our own planet remains unexplored. "
    "At the Challenger Deep the pressure is roughly 1,086 bar, or about "
    "15,750 pounds per square inch. Sunlight fades within the first few "
    "hundred meters, so the deep food web depends on material sinking from "
    "the surface. Wood falls and whale carcasses create temporary oases that "
    "support specialized communities for decades. Remotely operated vehicles "
    "let scientists photograph and sample these habitats without exposing "
    "people to the risk. Studying the trench also helps calibrate instruments "
    "for ocean worlds beyond Earth."
)


def fetch_corpus(path: str | None = None) -> bytes:
    """Load a local corpus, or return the built-in demo corpus.

    Training prefers an explicit ``--data`` path. The built-in text exists so a
    fresh checkout can run a short smoke test without a network request; it is
    not a language-modeling corpus.
    """
    if path:
        with open(path, "rb") as f:
            return f.read()
    return DEMO_TEXT.encode("utf-8")


def _tensor_rows(ids: torch.Tensor, width: int, stride: int | None = None):
    if ids.numel() < width:
        return ids[:0].view(0, width)
    stride = width if stride is None else stride
    if stride == width:
        return ids[:ids.numel() // width * width].view(-1, width)
    count = 1 + (ids.numel() - width) // stride
    return ids[:width + (count - 1) * stride].unfold(0, width, stride)


def _repeated_rows(data: bytes, width: int, stride: int | None = None):
    while len(data) < 500_000:
        data = data + data
    ids = torch.tensor(bytearray(data), dtype=torch.uint8)
    return _tensor_rows(ids, width, stride)


def corpus_rows(data: bytes | str | os.PathLike, width: int):
    if not data:
        raise ValueError("training corpus is empty")
    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        size = os.path.getsize(path)
        if not size:
            raise ValueError("training corpus is empty")
        if size >= 500_000:
            mapped = np.memmap(path, dtype=np.uint8, mode="c")
            return _tensor_rows(torch.from_numpy(mapped), width)
        with open(path, "rb") as handle:
            data = handle.read()
    return _repeated_rows(data, width)


def corpus_rows_split(data: bytes | str | os.PathLike, width: int,
                      val_frac: float = 0.05, stride: int | None = None,
                      split_width: int | None = None):
    """Create disjoint source partitions before any small-corpus repetition."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    stride = width if stride is None else stride
    split_width = width if split_width is None else split_width
    if stride <= 0:
        raise ValueError("stride must be positive")
    if split_width < width:
        raise ValueError("split_width must be at least the row width")
    if not data:
        raise ValueError("training corpus is empty")

    path = None
    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        size = os.path.getsize(path)
        if not size:
            raise ValueError("training corpus is empty")
        if size < 500_000:
            with open(path, "rb") as handle:
                data = handle.read()
            path = None
    size = len(data) if path is None else os.path.getsize(path)
    if size < 2 * split_width:
        raise ValueError(
            f"need at least {2 * split_width} source bytes for a disjoint "
            f"train/validation split; got {size}"
        )
    val_size = min(max(split_width, int(size * val_frac)), size - split_width)

    if path is not None:
        mapped = np.memmap(path, dtype=np.uint8, mode="c")
        cutoff = size - val_size
        train = _tensor_rows(torch.from_numpy(mapped[:cutoff]), width, stride)
        val = _tensor_rows(torch.from_numpy(mapped[cutoff:]), width, stride)
        return train, val

    train = _repeated_rows(data[:-val_size], width, stride)
    val = _repeated_rows(data[-val_size:], width, stride)
    return train, val


def build_dataset(data: bytes | str | os.PathLike, ctx: int, val_frac: float = 0.05):
    """Split source bytes, then chunk into rows without repetition leakage."""
    return corpus_rows_split(data, ctx + 1, val_frac)
