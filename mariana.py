#!/usr/bin/env python3
"""Mariana: a tiny hybrid Diffusion-AR language model (DRV-LM).

This is a thin backward-compatible wrapper around the ``mariana`` package.

Examples:
    python mariana.py                      # train + demo (same as: -m mariana train)
    python mariana.py --steps 3000
    python mariana.py train --ncm --steps 3000  # two-stage Next-Concept Model
    python mariana.py --gui                     # local training + chat studio
    python mariana.py gen --prompt "KING: "
    python mariana.py think --prompt "KING: "   # print the private latent code
    python mariana.py bench
    python mariana.py export --format all  # GGUF + ONNX + safetensors -> exports/
    python mariana.py setup                # pyenv + venv + core requirements (any OS)
    python mariana.py update               # update from the latest GitHub release

Device selection is automatic: CUDA -> Intel XPU -> Apple MPS -> CPU.
"""

import os
import sys

# Mitigate CUDA fragmentation on small GPUs by using expandable segments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", str(min(24, os.cpu_count() or 4)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

from mariana.bootstrap import ensure_core  # noqa: E402 - environment is configured first

# Setup and updating must work even when the runtime is missing, so they run
# before torch (and before ensure_core tries to install anything).
if __name__ == "__main__" and sys.argv[1:2] == ["update"]:
    from mariana.update import main as update_main

    raise SystemExit(update_main(sys.argv[2:]))

if __name__ == "__main__" and sys.argv[1:2] == ["setup"]:
    from mariana.setup import main as setup_main

    raise SystemExit(setup_main(sys.argv[2:]))

if __name__ == "__main__":
    from mariana.setup import reexec_in_venv

    reexec_in_venv()

ensure_core()

try:
    import torch
except ImportError as error:
    print(f"PyTorch could not be imported: {error}")
    print("Install the core runtime with: pip install -r requirements.txt")
    raise SystemExit(1) from None

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(1)
try:
    torch.set_flush_denormal(True)
except Exception:
    pass

from mariana.cli import main

if __name__ == "__main__":
    main()
