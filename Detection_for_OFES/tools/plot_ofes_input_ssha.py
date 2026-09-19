"""Plot one OFES SSHA input field without temporal or spatial filtering."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


DEFAULT_INPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter")
DEFAULT_OUTPUT_ROOT = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\filter_diagnostics_19910101")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default="1991-01-01")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    day = date.fromisoformat(args.day)
    source = args.input_root / f"global_phy_{day:%Y%m%d}.nc"
    if not source.exists():
        raise FileNotFoundError(source)
    args.output_root.mkdir(parents=True, exist_ok=True)

    with Dataset(source) as ds:
        lon = np.asarray(ds.variables["longitude"][:], dtype="f8")
        lat = np.asarray(ds.variables["latitude"][:], dtype="f8")
        ssh = np.ma.asarray(ds.variables["zos_glor"][0], dtype="f8").filled(np.nan)
        units = str(getattr(ds.variables["zos_glor"], "units", "unknown"))
        science_tag = str(getattr(ds, "science_tag", "not specified"))
        source_note = str(getattr(ds, "source", "not specified"))
    ssh[~np.isfinite(ssh)] = np.nan
    vmax = max(float(np.nanpercentile(np.abs(ssh), 99.5)), 0.1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    mesh = ax.pcolormesh(lon, lat, ssh, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    mesh.set_rasterized(True)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        f"OFES input SSHA | {day.isoformat()}\n"
        "No running mean; no spatial filter; no scale separation\n"
        "Existing input definition: daily field minus Jan01-Jan19 mean"
    )
    fig.colorbar(mesh, ax=ax, label=f"SSHA ({units})")
    png = args.output_root / f"ofes_input_unfiltered_ssha_global_{day:%Y%m%d}.png"
    fig.savefig(png, dpi=180)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)

    manifest = {
        "status": "complete",
        "day": day.isoformat(),
        "source_file": str(source),
        "source": source_note,
        "science_tag": science_tag,
        "processing_applied_by_this_tool": "none",
        "input_definition": "daily zos_glor already provided by the input adapter; source metadata identifies it as raw_minus_jan_mean_diagnostic",
        "units": units,
        "colour_limit_abs": vmax,
        "figure": str(png),
    }
    manifest_path = args.output_root / f"ofes_input_unfiltered_ssha_manifest_{day:%Y%m%d}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(png), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
