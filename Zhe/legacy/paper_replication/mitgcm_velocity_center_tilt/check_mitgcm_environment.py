from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the MITgcm build environment for the velocity-center tilt experiment.")
    parser.add_argument("--mitgcm-root", default="/root/Verify/vendor/MITgcm")
    parser.add_argument("--output-root", default="/root/autodl-fs/kuroshiou_mitgcm_velocity_center_tilt_validation")
    return parser.parse_args()


def _version(command: str) -> str | None:
    exe = shutil.which(command)
    if not exe:
        return None
    try:
        completed = subprocess.run([exe, "--version"], check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return f"{exe}: version check failed: {exc}"
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    return f"{exe}: {first_line}"


def main() -> int:
    args = _parse_args()
    mitgcm_root = Path(args.mitgcm_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tools = {
        "make": _version("make"),
        "gfortran": _version("gfortran"),
        "x86_64-conda-linux-gnu-gfortran": _version("x86_64-conda-linux-gnu-gfortran"),
        "mpif90": _version("mpif90"),
    }
    genmake = mitgcm_root / "tools" / "genmake2"
    manifest = {
        "mitgcm_root": str(mitgcm_root),
        "mitgcm_root_exists": mitgcm_root.exists(),
        "genmake2_exists": genmake.exists(),
        "tools": tools,
        "ready_to_compile": bool((tools["gfortran"] or tools["x86_64-conda-linux-gnu-gfortran"] or tools["mpif90"]) and genmake.exists()),
        "suggested_conda_install": "mamba install -n eddy_verify -y -c conda-forge gfortran_linux-64 netcdf-fortran",
    }
    (output_root / "mitgcm_environment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["ready_to_compile"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
