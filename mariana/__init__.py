"""Mariana: a tiny hybrid Diffusion-AR language model (DRV-LM).

The public names are imported lazily so that light-weight entry points (the
Studio CLI, packaging tools, help output) do not import PyTorch until they need
it.  ``ensure_core()`` is called by the CLI entry points before the model
modules load.
"""

from __future__ import annotations

__version__ = "2.3.2"
__all__ = [
    "Config",
    "MarianaLM",
    "format_code",
    "generate_latent",
    "summarize",
    "__version__",
]


def __getattr__(name: str):
    if name == "Config":
        from .config import Config

        return Config
    if name == "MarianaLM":
        from .model import MarianaLM

        return MarianaLM
    if name in {"format_code", "generate_latent", "summarize"}:
        from . import latent

        return getattr(latent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
