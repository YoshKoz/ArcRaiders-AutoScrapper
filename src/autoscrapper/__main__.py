from __future__ import annotations

import sys

from .cli import config as config_cli
from .cli import progress as progress_cli
from .cli import rules as rules_cli
from .cli import scan as scan_cli
from .cli.home import show_home_menu


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return show_home_menu()

    cmd, *rest = args
    cmd = cmd.lower().strip()

    if cmd == "scan":
        return scan_cli.main(rest)
    if cmd == "rules":
        return rules_cli.main(rest)
    if cmd == "progress":
        return progress_cli.main(rest)
    if cmd in {"config", "scan-config", "scan_configuration", "settings"}:
        return config_cli.main(rest)

    print(f"Unknown command: {cmd}\n")
    return show_home_menu()


if __name__ == "__main__":
    raise SystemExit(main())
