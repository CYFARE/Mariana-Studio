# Setup

`python mariana.py setup` prepares a working Mariana environment on Linux,
macOS and Windows in one command. It chooses a CPython interpreter, creates a
project virtual environment, installs the fixed requirements files inside it,
and verifies that PyTorch and NumPy import.

The command runs before PyTorch is imported, so it works on a fresh checkout
where nothing is installed yet. It never installs into the system interpreter,
so PEP 668 ("externally managed environment") distributions are left untouched.

## Quick start

```bash
git clone https://github.com/CYFARE/Mariana-Studio.git
cd Mariana-Studio
python mariana.py setup
python mariana.py studio
```

The `mariana.py` wrapper re-execs into `.venv` automatically, so no activation
step is required. Activate the environment only for direct tools such as `pip`,
`pytest` or `pyflakes`:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (cmd)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Without activation, call the environment directly:

```bash
.venv/bin/python mariana.py studio         # Linux / macOS
.venv\Scripts\python.exe mariana.py studio # Windows
```

Set `MARIANA_NO_VENV=1` to stop the wrapper from selecting `.venv` (for example
when you deliberately want another interpreter).

Re-running `setup` is safe: it reuses `.venv`, upgrades pip and applies the
selected requirements. Changing `--python` to a different `major.minor` line
requires `--recreate`, because an existing virtual environment keeps the
interpreter it was built with.

## How the environment is selected

`python mariana.py ...` and `python -m mariana ...` look for `.venv` (or a legacy
`venv`) next to the checkout. When the current interpreter is the system Python,
the wrapper re-executes itself with the venv interpreter and marks the child
with `MARIANA_VENV_ACTIVE=1`, so this happens once and never loops. An already
active virtual environment is respected.

This matters beyond imports: Studio installs optional requirements with the
interpreter that runs the server. Starting Studio through the wrapper therefore
means pip installs into `.venv`, while PEP 668 distributions such as Arch,
Debian and Fedora never see a system-wide install.

Set `MARIANA_NO_VENV=1` to keep the current interpreter. Set `MARIANA_VENV` to
the path of a custom environment created with `--venv`. The `setup`, `update`
and `--help` paths always work without the venv.

## Flags

| Flag | Meaning |
| --- | --- |
| `--python VERSION` | CPython line or exact patch, e.g. `3.14` or `3.14.2`. Default: the line of the interpreter running setup. |
| `--venv PATH` | Virtual environment directory. Default: `.venv` in the checkout. |
| `--device auto` | `requirements.txt`, the GPU-first wheel with automatic CPU fallback. Default. |
| `--device cpu` | `requirements-cpu.txt`, the small CPU-only wheel. |
| `--device cuda` | `requirements-cuda.txt`, the CUDA wheel (identical to the default on purpose). |
| `--device xpu` | `requirements-xpu.txt`, the Intel Arc / XPU wheel. |
| `--extras LIST` | Comma-separated optional groups: `data`, `export`, `finetune`, or `all`. |
| `--recreate` | Delete and rebuild an existing virtual environment. Refuses to delete a directory that is not a venv. |
| `--install-pyenv` | Clone pyenv into `PYENV_ROOT` (or `~/.pyenv`) when it is missing. |
| `--no-pyenv` | Ignore pyenv and use an installed interpreter, for CI or a distro-managed Python. |

Examples:

```bash
# CPU-only server
python mariana.py setup --device cpu

# Intel Arc graphics
python mariana.py setup --device xpu

# Reproducible Python line plus every optional toolchain
python mariana.py setup --python 3.14 --extras all

# Existing virtual environment that should be rebuilt
python mariana.py setup --recreate
```

## Interpreter policy

1. If `--python` is omitted, setup requests the same `major.minor` as the
   interpreter running the command, when that line is at least Python 3.10.
   Otherwise it requests 3.12.
2. When pyenv is found (through `PATH`, `PYENV_ROOT`, or `~/.pyenv`), setup uses
   it. An installed matching version is reused; otherwise the newest patch is
   installed with `pyenv install`. Installing CPython compiles it, so the first
   run can take several minutes.
3. Without pyenv, setup accepts an installed interpreter on `PATH` (for example
   `python3.13`). If none matches, it stops with the exact pyenv command to run.
4. `--install-pyenv` clones pyenv when it is absent. `--no-pyenv` skips the
   version manager entirely.

The project supports Python 3.10 and newer. Lines newer than the newest tested
one (3.15) are allowed, with a warning, as long as pip can find wheels.

When `--install-pyenv` clones pyenv, the checkout lives at `PYENV_ROOT` (or
`~/.pyenv`). Setup uses it directly, so no shell configuration is required for
Mariana. To use the same pyenv from other terminals, add it to your shell:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

On Windows, add `%USERPROFILE%\.pyenv\pyenv-win\bin` and
`%USERPROFILE%\.pyenv\pyenv-win\shims` to `PATH` instead.

## Requirements

| Group | File | Installs |
| --- | --- | --- |
| core (default) | `requirements.txt` | PyTorch and NumPy, GPU-first with CPU fallback |
| CPU | `requirements-cpu.txt` | CPU-only PyTorch wheels |
| CUDA | `requirements-cuda.txt` | CUDA PyTorch wheels |
| XPU | `requirements-xpu.txt` | Intel XPU PyTorch wheels |
| `data` | `requirements-data.txt` | Hugging Face datasets, Hub client, Parquet |
| `export` | `requirements-export.txt` | GGUF, ONNX, ONNX Runtime, safetensors |
| `finetune` | `requirements-finetune.txt` | Transformers and safetensors |

Only one device profile is installed at a time. Optional groups are installed
after the core runtime; if one fails, setup reports the failure, keeps the
working core environment and exits successfully so training can still start.

## What setup verifies

Inside the new environment, setup imports `torch` and `numpy`, prints their
versions and reports whether CUDA, XPU or MPS is available. A failed import is
an error that names the interpreter and virtual environment involved; a missing
device is not, because Mariana falls back to CPU.

## Troubleshooting

**`Could not create a virtual environment` on Debian/Ubuntu.** The `venv`
module is a separate package:

```bash
sudo apt install python3-venv
```

**`pyenv install` fails while compiling.** pyenv needs a C toolchain and the
development headers. On Debian/Ubuntu:

```bash
sudo apt install build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev libffi-dev liblzma-dev
```

macOS users need the Xcode command-line tools (`xcode-select --install`).
Windows users of pyenv-win get a prebuilt installer instead of a compile.

**`pyenv has no CPython 3.x to install`.** Update the pyenv checkout
(`git pull`) or choose a line it knows: `python mariana.py setup --python 3.13`.

**No matching wheels.** A brand-new CPython line may not have PyTorch wheels
for your platform yet. Choose the previous line:

```bash
python mariana.py setup --python 3.13 --recreate
```

**Corporate proxy.** Set the usual pip variables before running setup, for
example `HTTPS_PROXY` and `PIP_CERT`; setup forwards the environment to pip.

**Studio says `pip refused to install ... externally managed`.** Studio was
started with the system Python, so pip correctly refused a system-wide install.
Restart it through the wrapper (which selects `.venv` automatically):

```bash
python mariana.py studio
```

Then retry the dependency panel. `.venv/bin/python mariana.py studio`
(`.venv\Scripts\python.exe mariana.py studio` on Windows) does the same thing
without relying on the wrapper.

**Remove the environment.** Delete `.venv` (and `.gitignore` already excludes
it). The system interpreter and pyenv installations are unaffected.

## Related

- [Readme quick start](../README.md#quick-start)
- [Update the checkout](update.md)
- [Engineering standard](../AGENTS.md)
