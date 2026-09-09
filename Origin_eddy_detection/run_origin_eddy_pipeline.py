from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    runpy.run_module("src.eddy_pipeline.cli", run_name="__main__")


if __name__ == "__main__":
    main()
