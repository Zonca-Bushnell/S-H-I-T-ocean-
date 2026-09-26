"""Run section-bipolar diagnostics and materialize its non-destructive catalog."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", type=Path, required=True)
    parser.add_argument("--vertical-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--vertical-definition", required=True)
    parser.add_argument("--supported-min-fraction", required=True)
    parser.add_argument("--core-min-fraction", required=True)
    parser.add_argument("--strict-core-min-fraction", required=True)
    args = parser.parse_args()
    subprocess.run([
        sys.executable, "-m", "Detection_for_OFES.stages.section_bipolar",
        "--velocity-file", str(args.velocity_file), "--vertical-root", str(args.vertical_root),
        "--output-root", str(args.diagnostic_root), "--day", args.day,
    ], check=True)
    diagnostic = args.diagnostic_root / args.vertical_root.name
    subprocess.run([
        sys.executable, "-m", "Detection_for_OFES.stages.section_catalog",
        "--vertical-root", str(args.vertical_root), "--bipolarity-root", str(diagnostic),
        "--output-root", str(args.catalog_root), "--day", args.day,
        "--vertical-definition", args.vertical_definition,
        "--supported-min-fraction", args.supported_min_fraction,
        "--core-min-fraction", args.core_min_fraction,
        "--strict-core-min-fraction", args.strict_core_min_fraction,
    ], check=True)


if __name__ == "__main__":
    main()
