import sys

from .bootstrap import ensure_core

# Updating must work even when the runtime is broken, so it runs before torch.
if __name__ == "__main__" and sys.argv[1:2] == ["update"]:
    from .update import main as update_main

    raise SystemExit(update_main(sys.argv[2:]))

ensure_core()

from .cli import main  # noqa: E402 - bootstrapping must happen before torch is imported

if __name__ == "__main__":
    main()
