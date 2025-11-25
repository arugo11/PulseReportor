from __future__ import annotations

import sys

from .cli import app


def main() -> None:
    if len(sys.argv) <= 1 or sys.argv[1] != "run_full_session":
        sys.argv.insert(1, "run_full_session")
    app()


if __name__ == "__main__":
    main()
