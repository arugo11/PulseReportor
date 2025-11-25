from __future__ import annotations

import sys

from .cli import app


def main() -> None:
    if len(sys.argv) <= 1 or sys.argv[1] != "collect":
        sys.argv.insert(1, "collect")
    app()


if __name__ == "__main__":
    main()
