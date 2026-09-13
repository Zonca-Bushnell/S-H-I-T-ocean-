from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, Rectangle


def local_xy_km(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    mx = 111.2 * math.cos(math.radians(lat0))
    return ((lon - lon0 + 180.0) % 360.0 - 180.0) * mx, (lat - lat0) * 111.2


def main() -> None:
    root = Path(r"E:\DATA\01_Eddy_correspond\02_OFES\origin_streamline_cpu_jan01_jan19_life1")
    det = root / "hua_b3_start2_detection"
    out = root / "figures" / "latest_velocity_streamline_surface_and_family_with_edges"
    out.mkdir(parents=True, exist_ok=True)

    centers = pd.read_parquet(det / "centers_hua_style.parquet")
    structures = pd.read_parquet(det / "structures_hua_style.parquet")
    centers = centers[centers["boundary_mode"].astype(str).eq("velocity_streamline_contour")].copy()
    centers["date"] = pd.to_datetime(centers["date"]).dt.strftime("%Y-%m-%d")
    structures["date"] = pd.to_datetime(structures["date"]).dt.strftime("%Y-%m-%d")

    surface = centers[(centers["depth_index"].astype(int) == 0) & (centers["hua_pass"].astype(bool))].copy()
    surf_struct = structures[structures["depth_index"].astype(int).eq(0)][["date", "hua_object_id", "radius_km"]].copy()
    surface = surface.merge(surf_struct, on=["date", "hua_object_id"], how="left")
    surface = surface.dropna(subset=["center_lon_refined", "center_lat_refined", "radius_km"])

    colors = {"cyclonic": "#2563eb", "anticyclonic": "#dc2626"}
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 13.2), constrained_layout=True)
    for ax, day in zip(axes, ["1991-01-01", "1991-01-10", "1991-01-19"]):
        day_df = surface[surface["date"].eq(day)].copy()
        kuro = day_df[(day_df.center_lon_refined.between(120, 145)) & (day_df.center_lat_refined.between(20, 35))]
        other = day_df.drop(kuro.index)
        if len(other) > 650:
            other = other.sample(650, random_state=20260913)
        show = pd.concat([other, kuro], ignore_index=True)
        for pol, grp in show.groupby("polarity"):
            color = colors.get(str(pol), "#555555")
            ax.scatter(grp["center_lon_refined"], grp["center_lat_refined"], s=5, c=color, alpha=0.35, linewidths=0, label=f"{pol} centers")
            for _, row in grp.iterrows():
                lon = float(row["center_lon_refined"])
                lat = float(row["center_lat_refined"])
                radius_km = float(row["radius_km"])
                if not np.isfinite(radius_km) or radius_km <= 0:
                    continue
                deg_lat = radius_km / 111.2
                deg_lon = deg_lat / max(math.cos(math.radians(lat)), 0.18)
                ax.add_patch(Ellipse((lon, lat), 2.0 * deg_lon, 2.0 * deg_lat, fill=False, ec=color, lw=0.38, alpha=0.42))
        ax.add_patch(Rectangle((120, 20), 25, 15, fill=False, ec="black", lw=1.2, ls="--", label="Kuroshio 120-145E, 20-35N"))
        ax.set_xlim(0, 360)
        ax.set_ylim(-80, 80)
        ax.set_title(f"{day} velocity-streamline surface Hua-pass edges | shown={len(show):,}/{len(day_df):,}; Kuroshio={len(kuro):,}")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower left", ncol=3, frameon=True, fontsize=8)
    fig.suptitle(
        "OFES latest velocity_streamline_contour surface overview with eddy edges\n"
        "Edge = accepted boundary radius proxy from velocity-streamline detection metadata",
        fontsize=14,
    )
    overview_png = out / "ofes_velocity_streamline_surface_overview_with_edges.png"
    fig.savefig(overview_png, dpi=180)
    fig.savefig(overview_png.with_suffix(".pdf"))
    plt.close(fig)

    surface_struct = structures[structures.depth_index.astype(int).eq(0)].copy()
    surface_struct = surface_struct[(surface_struct.center_lon_refined.between(120, 145)) & (surface_struct.center_lat_refined.between(20, 35))]
    summary = structures.groupby("hua_object_id").agg(
        date=("date", "first"),
        polarity=("polarity", "first"),
        pass_layers=("depth_index", "nunique"),
        radius_km=("radius_km", "first"),
        max_depth=("depth_m", "max"),
    ).reset_index()
    summary = summary.merge(surface_struct[["hua_object_id"]], on="hua_object_id", how="inner")
    selected = []
    for pol in ["cyclonic", "anticyclonic"]:
        cand = summary[summary.polarity.eq(pol)].sort_values(["pass_layers", "max_depth", "radius_km"], ascending=False)
        if not cand.empty:
            selected.append(cand.iloc[0])
    if not selected:
        raise RuntimeError("No Kuroshio velocity-streamline objects found for family panels")

    panel_paths = []
    for row in selected:
        oid = str(row.hua_object_id)
        part = structures[structures.hua_object_id.eq(oid)].sort_values("depth_index").copy()
        center_part = centers[centers.hua_object_id.eq(oid)].sort_values("depth_index").copy()
        surface_row = part.iloc[0]
        lon0 = float(surface_row.center_lon_refined)
        lat0 = float(surface_row.center_lat_refined)
        radius0 = float(surface_row.radius_km)
        dx, dy, edge_r, depth = [], [], [], []
        for _, rr in part.iterrows():
            x, y = local_xy_km(float(rr.center_lon_refined), float(rr.center_lat_refined), lon0, lat0)
            dx.append(x)
            dy.append(y)
            edge_r.append(float(rr.radius_km))
            depth.append(float(rr.depth_m))
        dx = np.asarray(dx)
        dy = np.asarray(dy)
        edge_r = np.asarray(edge_r)
        depth = np.asarray(depth)
        center_surface = center_part[center_part.depth_index.astype(int).eq(0)].head(1)
        streamline_points = int(float(center_surface.streamline_points.iloc[0])) if not center_surface.empty else -1

        fig = plt.figure(figsize=(17, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 4)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        ax3 = fig.add_subplot(gs[0:2, 2:4])
        ax4 = fig.add_subplot(gs[2, 2:4])

        ax1.plot(dx, depth, "-o", ms=3, lw=1.5, color="#1f4b99")
        ax1.fill_betweenx(depth, dx - edge_r, dx + edge_r, color="#1f4b99", alpha=0.10, label="accepted edge span")
        ax1.axvline(0, color="0.3", lw=0.8)
        ax1.invert_yaxis()
        ax1.grid(alpha=0.25)
        ax1.set_xlabel("delta x from surface center (km)")
        ax1.set_ylabel("depth (m)")
        ax1.set_title("1 delta x with edge span")
        ax1.legend(fontsize=8)

        ax2.plot(dy, depth, "-o", ms=3, lw=1.5, color="#1f4b99")
        ax2.fill_betweenx(depth, dy - edge_r, dy + edge_r, color="#1f4b99", alpha=0.10, label="accepted edge span")
        ax2.axvline(0, color="0.3", lw=0.8)
        ax2.invert_yaxis()
        ax2.grid(alpha=0.25)
        ax2.set_xlabel("delta y from surface center (km)")
        ax2.set_ylabel("depth (m)")
        ax2.set_title("2 delta y with edge span")
        ax2.legend(fontsize=8)

        sc = ax3.scatter(dx, dy, c=depth, cmap="viridis", s=26, zorder=3, label="layer centers")
        for xi, yi, ri in zip(dx, dy, edge_r):
            ax3.add_patch(Circle((xi, yi), ri, fill=False, ec="0.35", lw=0.55, alpha=0.38))
        ax3.add_patch(Circle((0, 0), radius0, fill=False, ec="black", lw=1.9, label="surface accepted edge"))
        ax3.scatter([0], [0], marker="*", s=120, c="red", edgecolors="black", linewidths=0.4, zorder=4, label="surface center")
        ax3.set_aspect("equal", adjustable="box")
        lim = max(4.0 * max(float(np.nanmax(edge_r)), 1.0), float(np.nanmax(np.hypot(dx, dy))) + float(np.nanmax(edge_r)))
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.grid(alpha=0.25)
        ax3.set_xlabel("east from surface center (km)")
        ax3.set_ylabel("north from surface center (km)")
        ax3.set_title(f"3 layer-center family and accepted edges\nsurface streamline closed; metadata points={streamline_points}")
        ax3.legend(loc="upper right", fontsize=8)
        cb = fig.colorbar(sc, ax=ax3, fraction=0.035, pad=0.02)
        cb.set_label("depth (m)")

        ax4.plot(part.center_lon_refined, part.center_lat_refined, "-o", ms=3, color="#1f4b99")
        for _, rr in part.iterrows():
            deg = float(rr.radius_km) / 111.2
            ax4.add_patch(Circle((float(rr.center_lon_refined), float(rr.center_lat_refined)), deg, fill=False, ec="0.35", lw=0.45, alpha=0.38))
        ax4.add_patch(Circle((lon0, lat0), radius0 / 111.2, fill=False, ec="black", lw=1.4, label="surface edge"))
        ax4.scatter([lon0], [lat0], marker="*", s=95, c="red", edgecolors="black", linewidths=0.4)
        ax4.set_aspect("equal", adjustable="datalim")
        ax4.grid(alpha=0.25)
        ax4.set_xlabel("longitude")
        ax4.set_ylabel("latitude")
        ax4.set_title("4 lon-lat vertical family with edges")
        ax4.legend(fontsize=8)

        fig.suptitle(
            f"OFES velocity_streamline family-panel with eddy edges | {oid} | {row.date} | "
            f"{row.polarity} | layers={int(row.pass_layers)} | radius={radius0:.1f} km",
            fontsize=14,
        )
        panel_png = out / f"ofes_velocity_streamline_family_panel_with_edges_{oid}.png"
        fig.savefig(panel_png, dpi=180)
        fig.savefig(panel_png.with_suffix(".pdf"))
        plt.close(fig)
        panel_paths.append(str(panel_png))

    manifest = {
        "boundary_mode": "velocity_streamline_contour",
        "overview_png": str(overview_png),
        "family_panels": panel_paths,
        "edge_note": "Edges use accepted boundary radius proxy from velocity_streamline_contour metadata; streamline vertex coordinates were not persisted in centers_hua_style.parquet.",
    }
    (out / "surface_family_with_edges_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
