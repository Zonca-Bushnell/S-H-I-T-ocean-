"""Plot unfiltered OFES SLA defined from native eta and the annual eta MSS."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Detection_for_OFES.ofes_io import ctl_path, parse_ctl, read_variable_latlon_daily_only


OFES_ROOT = Path(r"F:\OFES\external_OFES2")
MSS_PATH = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012"
    r"\climatology\ofes2_eta_annual_mss_1993_2012.npz"
)
OUTPUT_ROOT = Path(
    r"E:\DATA\01_Eddy_correspond\02_OFES"
    r"\origin_unified_eta_mss_three_kernel_surface_jan01_jan19\comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofes-root", type=Path, default=OFES_ROOT)
    parser.add_argument("--mss-path", type=Path, default=MSS_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--day", default="1991-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = date.fromisoformat(args.day)
    meta = parse_ctl(ctl_path(args.ofes_root, "eta"))
    lon = np.asarray(meta.x.values, dtype="f8")
    lat = np.asarray(meta.y.values, dtype="f8")
    eta = np.asarray(read_variable_latlon_daily_only(args.ofes_root, "eta", target), dtype="f8")
    with np.load(args.mss_path, allow_pickle=False) as data:
        mss = np.asarray(data["eta_annual_mss_cm"], dtype="f8")
    if eta.shape != mss.shape:
        raise RuntimeError(f"eta/MSS shape mismatch: {eta.shape} != {mss.shape}")
    sla = eta - mss
    sla[~np.isfinite(eta) | ~np.isfinite(mss)] = np.nan
    limit = float(np.nanpercentile(np.abs(sla), 99.0))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#24282e")
    figure, axis = plt.subplots(figsize=(20, 9.5), constrained_layout=True)
    image = axis.imshow(
        sla, origin="lower", aspect="auto", interpolation="nearest",
        extent=(lon[0], lon[-1], lat[0], lat[-1]), cmap=cmap, vmin=-limit, vmax=limit,
    )
    axis.set_title(
        f"OFES native SLA | {target.isoformat()}\n"
        r"$SLA=\eta-MSS_{\eta,1993-2012}$; no pair correction, temporal mean, spatial filter, or regridding",
        fontsize=15,
    )
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude")
    axis.set_xticks(np.arange(0, 361, 60))
    axis.set_yticks(np.arange(-60, 61, 30))
    axis.grid(True, color="black", alpha=0.18, linewidth=0.45)
    figure.colorbar(image, ax=axis, label="SLA (cm)", shrink=0.88)
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = args.output_root / f"ofes_native_eta_mss_sla_{target:%Y%m%d}"
    figure.savefig(stem.with_suffix(".png"), dpi=180)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    manifest = {
        "status": "complete",
        "day": target.isoformat(),
        "definition": "SLA=eta(day)-day-weighted annual mean eta over 1993-2012",
        "atmospheric_pressure_policy": "no algebraic pair correction",
        "processing": "native grid; no temporal or spatial filter",
        "finite_fraction": float(np.isfinite(sla).mean()),
        "std_cm": float(np.nanstd(sla)),
        "q95_abs_cm": float(np.nanpercentile(np.abs(sla), 95.0)),
        "color_limit_cm": limit,
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
    }
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
