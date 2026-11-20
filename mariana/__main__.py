import sys

from .bootstrap import ensure_core

# Setup and updating must work even when the runtime is missing, so they run
# before torch (and before ensure_core tries to install anything).
if __name__ == "__main__" and sys.argv[1:2] == ["update"]:
    from .update import main as update_main

    raise SystemExit(update_main(sys.argv[2:]))

if __name__ == "__main__" and sys.argv[1:2] == ["setup"]:
    from .setup import main as setup_main

    raise SystemExit(setup_main(sys.argv[2:]))

if __name__ == "__main__":
    from .setup import reexec_in_venv

    reexec_in_venv(module=True)

ensure_core()

from .cli import main  # noqa: E402 - bootstrapping must happen before torch is imported

if __name__ == "__main__":
    main()
