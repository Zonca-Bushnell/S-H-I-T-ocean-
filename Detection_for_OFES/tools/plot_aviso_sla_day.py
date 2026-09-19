"""Fetch and plot one official AVISO/Copernicus Marine SLA day.

The legacy global daily DT gridded directory is not exposed by every AVISO
account after the AVISO/CMEMS migration.  This tool therefore uses the
official AVISO Sea Views PNG as a transparent fallback.  It never labels a
PNG quicklook as a locally decoded NetCDF field.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path


QUICKLOOK_TEMPLATE = (
    "https://bulletin.aviso.altimetry.fr/images/produits/aviso/"
    "aviso_{day}/gim/glo/"
    "duacs_global_nrt_msla_merged_h_{day}_glo_msla_n0_t0.png"
)


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "OFES-AVISO-preview/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive one official AVISO global SLA map.")
    parser.add_argument("--day", default="1993-01-01", help="YYYY-MM-DD")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\DATA\01_Eddy_correspond\03_AVISO_altimetry_preview"),
    )
    parser.add_argument(
        "--netcdf-path",
        type=Path,
        default=None,
        help="Existing Copernicus Marine NetCDF to plot; otherwise fetch the official quicklook PNG.",
    )
    return parser.parse_args()


def plot_netcdf(nc_path: Path, png_path: Path, pdf_path: Path, day_text: str) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(nc_path) as ds:
        def pick(names: tuple[str, ...]) -> str:
            for name in names:
                if name in ds.variables:
                    return name
            raise KeyError(f"None of {names!r} found in {nc_path}")

        lon_name = pick(("longitude", "lon"))
        lat_name = pick(("latitude", "lat"))
        sla_name = pick(("sla", "SLA", "sea_level_anomaly"))
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        var = ds.variables[sla_name]
        sla = np.ma.asarray(var[:], dtype=float).filled(np.nan)
        while sla.ndim > 2:
            sla = sla[0]
        fill = getattr(var, "_FillValue", None)
        if fill is not None:
            sla[np.isclose(sla, float(fill), equal_nan=False)] = np.nan
        units = str(getattr(var, "units", "unknown"))
        # Copernicus SLA is normally metres; plot in centimetres for readability.
        plot_units = units
        if units.lower() in {"m", "meter", "meters", "metre", "metres"}:
            sla = sla * 100.0
            plot_units = "cm (converted from m)"
        finite = np.isfinite(sla)
        if not finite.any():
            raise ValueError("SLA grid contains no finite values")
        vmax = float(np.nanpercentile(np.abs(sla), 99.0))
        vmax = max(vmax, 1.0)
        vmax = min(vmax, 30.0)
        metadata = {
            "local_netcdf": str(nc_path),
            "dataset_variable": sla_name,
            "longitude_variable": lon_name,
            "latitude_variable": lat_name,
            "source_units": units,
            "plot_units": plot_units,
            "grid_shape": list(sla.shape),
            "longitude_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
            "latitude_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
            "finite_fraction": float(finite.mean()),
            "data_min_plot_units": float(np.nanmin(sla)),
            "data_max_plot_units": float(np.nanmax(sla)),
            "colorbar_limits_plot_units": [-vmax, vmax],
        }

    fig, ax = plt.subplots(figsize=(16, 8.5), constrained_layout=True)
    ax.set_facecolor("#22262d")
    mesh = ax.pcolormesh(lon, lat, sla, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"AVISO/Copernicus Marine DT all-satellite SLA | {day_text[:4]}-{day_text[4:6]}-{day_text[6:]}")
    ax.grid(True, color="k", alpha=0.25, linewidth=0.5)
    cb = fig.colorbar(mesh, ax=ax, pad=0.015)
    cb.set_label(plot_units)
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return metadata


def main() -> None:
    args = parse_args()
    day = datetime.strptime(args.day, "%Y-%m-%d").date()
    day_text = day.strftime("%Y%m%d")
    out_dir = args.output_root / day_text
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"aviso_dt_allsat_sla_global_{day_text}.png"
    pdf_path = out_dir / f"aviso_dt_allsat_sla_global_{day_text}.pdf"
    manifest_path = out_dir / f"aviso_dt_allsat_sla_manifest_{day_text}.json"
    url = QUICKLOOK_TEMPLATE.format(day=day_text)

    nc_path = args.netcdf_path or (out_dir / f"aviso_cmems_allsat_sla_{day_text}.nc")
    if nc_path.exists():
        png_path = out_dir / f"aviso_cmems_allsat_sla_global_{day_text}.png"
        pdf_path = out_dir / f"aviso_cmems_allsat_sla_global_{day_text}.pdf"
    manifest: dict[str, object] = {
        "status": "running",
        "product": "AVISO/DUACS global Sea Level Anomaly",
        "series": "Copernicus Marine DT all-satellite gridded SLA",
        "requested_day": args.day,
        "field": "SLA-H",
        "source_url": (
            "https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047"
            if nc_path.exists()
            else url
        ),
        "source_format": "NetCDF-4" if nc_path.exists() else "official PNG quicklook",
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D" if nc_path.exists() else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        if nc_path.exists():
            manifest.update(plot_netcdf(nc_path, png_path, pdf_path, day_text))
            manifest.update({"status": "complete", "local_png": str(png_path), "local_pdf": str(pdf_path)})
        else:
            fetch(url, png_path)
            from PIL import Image

            with Image.open(png_path) as image:
                image.convert("RGB").save(pdf_path, "PDF", resolution=150.0)
                manifest.update({"status": "complete", "image_size_px": list(image.size), "local_png": str(png_path), "local_pdf": str(pdf_path)})
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[aviso-sla] wrote {png_path}")
    print(f"[aviso-sla] wrote {pdf_path}")


if __name__ == "__main__":
    main()
