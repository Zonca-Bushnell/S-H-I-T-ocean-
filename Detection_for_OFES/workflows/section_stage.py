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
    args = parser.parse_args()
    subprocess.run([
        sys.executable, "-m", "Detection_for_OFES.tools.diagnose_deep_hua_section_bipolarity",
        "--velocity-file", str(args.velocity_file), "--vertical-root", str(args.vertical_root),
        "--output-root", str(args.diagnostic_root), "--day", args.day,
    ], check=True)
    diagnostic = args.diagnostic_root / args.vertical_root.name
    subprocess.run([
        sys.executable, "-m", "Detection_for_OFES.tools.build_section_bipolar_vertical_catalog",
        "--vertical-root", str(args.vertical_root), "--bipolarity-root", str(diagnostic),
        "--output-root", str(args.catalog_root), "--day", args.day,
        "--vertical-definition", args.vertical_definition,
    ], check=True)


if __name__ == "__main__":
    main()
