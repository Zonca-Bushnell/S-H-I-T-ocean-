from __future__ import annotations

import argparse
from pathlib import Path

from . import isopycnal_a
from .models import assert_model_definitions


class _CheckModelDefinitions:
    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--forecast-root", default=str(Path(__file__).resolve().parent))

    @staticmethod
    def run(args) -> Path:
        root = Path(args.forecast_root).resolve()
        failures = assert_model_definitions(root)
        if failures:
            raise RuntimeError("Model definition audit failed:\n" + "\n".join(f"- {item}" for item in failures))
        marker = root / "MODEL_DEFINITION_AUDIT_OK.txt"
        marker.write_text(
            "OK: formal forecast models are baseline_li_depth_layer, model_A_isopycnal, model_B_isopycnal_streamfunction, and model_C_PE_isopycnal_PV_closure.\n",
            encoding="utf-8",
        )
        return marker


COMMANDS = {"isopycnal-a": isopycnal_a, "model-b": isopycnal_a, "model-c": isopycnal_a, "check-model-definitions": _CheckModelDefinitions}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Non-circular forecast experiments for Kuroshio 3D eddies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, module in COMMANDS.items():
        sub = subparsers.add_parser(name)
        module.add_arguments(sub)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = COMMANDS[args.command].run(args)
    print(f"[forecast] {args.command} output: {output}")


if __name__ == "__main__":
    main()
