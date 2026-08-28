from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("plot_original_eddy_discontinuity_9panel.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
