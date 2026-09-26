"""Single command entry point for the canonical OFES workflow."""
from __future__ import annotations

import sys


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "run"
    if command != "run":
        raise SystemExit(f"Unknown command {command!r}; the supported command is 'run'.")
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        del sys.argv[1]
    from Detection_for_OFES.workflows.default_pipeline import main as run_default
    run_default()


if __name__ == "__main__":
    main()
