# Contributing to Mariana

Thanks for taking the time to improve Mariana. The project is a hybrid
diffusion-AR language model, a local Studio, and an agent runtime; small,
well-tested changes are easier to review than broad rewrites.

## Before you start

1. Read [`AGENTS.md`](AGENTS.md). It is the engineering standard for the
   repository and covers the package map, naming, layering, errors, security,
   tests, and the definition of done.
2. Read [`docs/README.md`](docs/README.md) for the runbooks and whitepapers.
3. Check the existing tests and the source path a change belongs to. A bug fix
   should land with the regression test that would have caught it.

Mariana is licensed under the GNU Affero General Public License v3.0. By
contributing, you agree that your contribution is licensed under the same terms.

## Development setup

```bash
git clone https://github.com/CYFARE/Mariana.git
cd Mariana

python mariana.py setup --device cpu   # pyenv + .venv + core requirements
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install pytest pyflakes
```

The `mariana.py` wrapper selects `.venv` automatically for Studio and training;
activate it here because the verification commands below call
`python -m pytest` directly.

`--device cuda` (NVIDIA), `--device xpu` (Intel Arc) and the default
`--device auto` are also available; `--extras all` installs the optional export,
fine-tune and dataset toolchains. Setup never touches the system interpreter —
see [`docs/setup.md`](docs/setup.md) for every flag and for troubleshooting.

Manual equivalent:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt   # CI and CPU-only development
# or
pip install -r requirements.txt       # GPU-first default
pip install pytest pyflakes
```

Optional feature groups are separate so a core checkout stays small:

```bash
pip install -r requirements-export.txt     # GGUF / ONNX / safetensors
pip install -r requirements-finetune.txt   # third-party LoRA/full fine-tuning
pip install -r requirements-data.txt       # Hugging Face datasets and Parquet
```

## Verify your change

```bash
python -m pytest -q                 # complete CPU suite
python -m pyflakes mariana tests    # unused imports and undefined names
python mariana.py studio --no-browser
```

Then exercise the view or command you changed by hand. The Studio UI is a local
page with no build step, so a browser reload is enough for a frontend change.

Useful targeted commands:

```bash
python -m pytest tests/test_train.py -q
python -m pytest tests/test_ncm.py -q
python -m pytest tests/test_agents.py -q
python -m pytest tests/test_studio_assets.py -q
```

## Where things live

| Path | Owns |
| --- | --- |
| `mariana/*.py` | Core DRV-LM: config, model, training, generation, latent/SAINT, CLI. |
| `mariana/ncm/` | Next-Concept Modeling. |
| `mariana/export/` | GGUF, ONNX, and safetensors conversion. |
| `mariana/agents/` | AML/AMP, flow runtime, swarm coordinator, abilities, store, scheduler. |
| `mariana/studio/` | Studio backend services and API. |
| `mariana/studio/web/` | Studio frontend (vanilla JS/CSS, no build step). |
| `scripts/` | Standalone utilities. |
| `tests/` | pytest suite, one file per source path. |
| `docs/` | Runbooks and whitepapers. |

Runtime data (`trained-model/`, `models/`, `datasets/`, `exports/`, `backups/`,
`agent-workspace/`) is git-ignored. Do not add generated artifacts to a pull
request.

## Rules that are easy to miss

- **Naming.** `snake_case` modules and functions, `PascalCase` classes,
  `UPPER_SNAKE_CASE` constants, kebab-case docs and CLI flags, `snake_case` JSON
  keys. See the tables in `AGENTS.md`.
- **Tests.** One test file per source path. Use `tmp_path`, tiny CPU configs, and
  a local mock server on `127.0.0.1:0`. Tests must not touch the network or
  require a GPU. Poll with `wait_for`, never a bare `sleep`.
- **Studio errors.** Raise `StudioError` subclasses. Every message says what is
  wrong and what to do next. The wire shape is
  `{"error": "<message>", "code": "<snake_case>"}`.
- **Long work is a job.** Return an HTTP response quickly; run GPU work in a
  subprocess and I/O in a daemon thread through `jobs.JobRegistry`.
- **Filesystem writes.** Stay inside the runtime directories. Resolve a
  user-supplied path, check `relative_to()`, and write atomically.
- **Optional dependencies.** Import them inside the function that needs them.
  An unavailable backend returns `"available": false` with the exact install
  hint instead of failing startup.
- **Frontend security.** The page ships with a strict Content-Security-Policy.
  No inline `<script>`, `style=""`, or `on*=""` attributes. Build DOM with
  `Studio.h()` and `textContent`; only `Studio.markdown.render()` may produce
  HTML, and it escapes first.
- **Public specifications.** AML and AMP are additive public specs. Bump the
  minor version and update `mariana/agents/specs/*.md` in the same change.
- **Technique claims.** If you change a training technique, update its
  whitepaper in `docs/whitepapers/` and keep measured claims conditional on the
  hardware and protocol that produced them.

## Pull-request checklist

- [ ] The change has a focused scope and the root cause is addressed.
- [ ] A regression test is included for a bug fix.
- [ ] `python -m pytest -q` passes.
- [ ] `python -m pyflakes mariana tests` is clean.
- [ ] `python mariana.py studio --no-browser` starts and the changed view was
      exercised by hand.
- [ ] README, docs, or whitepapers are updated if behavior changed.
- [ ] No generated files, checkpoints, secrets, or runtime data are committed.
- [ ] AML/AMP changes bump the minor version and update the canonical specs.

## Security

Do not open a public issue for a vulnerability. See [`SECURITY.md`](SECURITY.md)
for the reporting process and the threat model.
