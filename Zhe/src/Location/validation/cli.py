from __future__ import annotations

from ..forecast.cli import build_parser
from ..forecast.cli import COMMANDS


def main() -> None:
    args = build_parser().parse_args()
    output = COMMANDS[args.command].run(args)
    print(f"[validation] {args.command} output: {output}")


if __name__ == "__main__":
    main()
