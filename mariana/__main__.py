from .bootstrap import ensure_core

ensure_core()

from .cli import main  # noqa: E402 - bootstrapping must happen before torch is imported

if __name__ == "__main__":
    main()
