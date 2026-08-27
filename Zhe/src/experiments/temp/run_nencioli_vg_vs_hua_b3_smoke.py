from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class SmokeWindow:
    start: date
    end: date
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class HuaParams:
    ssh_window_cells: int = 7
    surface_search_cells: int = 3
    start_radius_cells: int = 2
    max_radius_cells: int = 8
    speed_ratio_max: float = 3.0
    angle_jump_max_deg: float = 150.0
    tangent_tolerance_deg: float = 24.0
    symmetry_tolerance_deg: float = 120.0
    min_tangent_fraction: float = 0.55
    min_reversal_fraction: float = 0.55
    min_finite_fraction: float = 0.75
    direction_exception_extra: int = 2


def parse_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def date_range(start: date, end: date) -> list[date]:
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def time_lookup(ds: Dataset) -> dict[date, int]:
    tvar = ds.variables["time"]
    times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
    return {date(int(t.year), int(t.month), int(t.day)): i for i, t in enumerate(times)}


def grid_spacing_km(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    mid_lat = float(np.nanmedian(lat))
    dx = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lon))))) * EARTH_RADIUS_M * math.cos(math.radians(mid_lat)) / 1000.0
    dy = np.deg2rad(float(np.nanmedian(np.abs(np.diff(lat))))) * EARTH_RADIUS_M / 1000.0
    return abs(dx), abs(dy)


def read_surface_day(filter_root: Path, day: date, window: SmokeWindow) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = filter_root / f"global_phy_{day.year}_bandpass_30_180d.nc"
    if not path.exists():
        path = filter_root / f"global_phy_{day.year}.nc"
    with Dataset(path) as ds:
        lookup = time_lookup(ds)
        tidx = lookup[day]
        lon = np.asarray(ds.variables["longitude"][:], dtype=float)
        lat = np.asarray(ds.variables["latitude"][:], dtype=float)
        lon_idx = np.where((lon >= window.lon_min) & (lon <= window.lon_max))[0]
        lat_idx = np.where((lat >= window.lat_min) & (lat <= window.lat_max))[0]
        u = np.asarray(ds.variables["uo_glor"][tidx, 0, lat_idx, lon_idx], dtype=float)
        v = np.asarray(ds.variables["vo_glor"][tidx, 0, lat_idx, lon_idx], dtype=float)
        zos = np.asarray(ds.variables["zos_glor"][tidx, lat_idx, lon_idx], dtype=float)
    lon2, lat2 = np.meshgrid(lon[lon_idx], lat[lat_idx])
    for arr in (u, v, zos):
        arr[np.abs(arr) > 1e20] = np.nan
    return lon2, lat2, u, v, zos


def nencioli_vg_centers(lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, *, a: int = 4, b: int = 3) -> list[dict[str, object]]:
    """Python port of Nencioli `uv_search.m` center constraints.

    The original 2-D VG detector uses velocity-only criteria: v zero
    crossing, u reversal, local speed minimum, and boundary vector rotation.
    """
    speed = np.hypot(u, v)
    rows: list[dict[str, object]] = []
    ny, nx = u.shape
    borders = max(a, b) + 1
    seen: set[tuple[int, int, int]] = set()
    for j in range(borders - 1, ny - borders + 1):
        wrk = v[j, :]
        signs = np.sign(wrk)
        zero_crossings = np.where(np.diff(signs) != 0)[0]
        for i0 in zero_crossings:
            if i0 < borders - 1 or i0 > nx - borders - 1:
                continue
            polarity_code = 0
            if wrk[i0] >= 0:
                if wrk[i0 - a] > wrk[i0] and wrk[i0 + 1 + a] < wrk[i0 + 1]:
                    polarity_code = -1
            elif wrk[i0] < 0:
                if wrk[i0 - a] < wrk[i0] and wrk[i0 + 1 + a] > wrk[i0 + 1]:
                    polarity_code = 1
            if polarity_code == 0:
                continue
            if polarity_code == -1:
                ok_u = (
                    u[j - a, i0] <= 0 and u[j - a, i0] <= u[j - 1, i0] and u[j + a, i0] >= 0 and u[j + a, i0] >= u[j + 1, i0]
                ) or (
                    u[j - a, i0 + 1] <= 0
                    and u[j - a, i0 + 1] <= u[j - 1, i0 + 1]
                    and u[j + a, i0 + 1] >= 0
                    and u[j + a, i0 + 1] >= u[j + 1, i0 + 1]
                )
            else:
                ok_u = (
                    u[j - a, i0] >= 0 and u[j - a, i0] >= u[j - 1, i0] and u[j + a, i0] <= 0 and u[j + a, i0] <= u[j + 1, i0]
                ) or (
                    u[j - a, i0 + 1] >= 0
                    and u[j - a, i0 + 1] >= u[j - 1, i0 + 1]
                    and u[j + a, i0 + 1] <= 0
                    and u[j + a, i0 + 1] <= u[j + 1, i0 + 1]
                )
            if not ok_u:
                continue
            srch = speed[j - b : j + b + 1, i0 - b : i0 + b + 2]
            if not np.isfinite(srch).any():
                continue
            local = int(np.nanargmin(srch))
            yy, xx = np.unravel_index(local, srch.shape)
            cj = j - b + yy
            ci = i0 - b + xx
            y0, y1 = max(cj - b, 0), min(cj + b + 1, ny)
            x0, x1 = max(ci - b, 0), min(ci + b + 1, nx)
            srch2 = speed[y0:y1, x0:x1]
            if not np.isfinite(srch2).any() or not np.nanmin(srch2) == np.nanmin(srch):
                continue
            if not nencioli_rotation_pass(u, v, ci, cj, a=a):
                continue
            lat_value = float(lat[cj, ci])
            code = -polarity_code if lat_value < 0 else polarity_code
            polarity = "cyclonic" if code == 1 else "anticyclonic"
            key = (cj, ci, code)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"method": "nencioli_vg", "i": ci, "j": cj, "lon": float(lon[cj, ci]), "lat": lat_value, "polarity": polarity})
    return rows


def nencioli_rotation_pass(u: np.ndarray, v: np.ndarray, center_i: int, center_j: int, *, a: int) -> bool:
    d = a - 1
    patch_u = u[center_j - d : center_j + d + 1, center_i - d : center_i + d + 1]
    patch_v = v[center_j - d : center_j + d + 1, center_i - d : center_i + d + 1]
    if patch_u.shape != (2 * d + 1, 2 * d + 1) or not np.isfinite(patch_u).all() or not np.isfinite(patch_v).all():
        return False
    ub = np.concatenate([patch_u[0, :], patch_u[1:, -1], patch_u[-1, -2::-1], patch_u[-2::-1, 0]])
    vb = np.concatenate([patch_v[0, :], patch_v[1:, -1], patch_v[-1, -2::-1], patch_v[-2::-1, 0]])
    quadrants = np.zeros_like(ub, dtype=int)
    quadrants[(ub >= 0) & (vb >= 0)] = 1
    quadrants[(ub < 0) & (vb >= 0)] = 2
    quadrants[(ub < 0) & (vb < 0)] = 3
    quadrants[(ub >= 0) & (vb < 0)] = 4
    spin = np.where(quadrants == 4)[0]
    if len(spin) == 0 or len(spin) == len(quadrants):
        return False
    q = quadrants.astype(int).copy()
    if spin[0] == 0:
        not_four = np.where(q != 4)[0]
        if len(not_four) == 0:
            return False
        spin_end = not_four[0] - 1
    else:
        spin_end = spin[-1]
    q[spin_end + 1 :] += 4
    dq = np.diff(q)
    return not np.any(dq > 1) and not np.any(dq < 0)


def local_extrema(zos: np.ndarray, window: int, *, max_candidates: int) -> list[dict[str, object]]:
    half = max(1, window // 2)
    rows: list[dict[str, object]] = []
    ny, nx = zos.shape
    for j in range(half, ny - half):
        for i in range(half, nx - half):
            value = float(zos[j, i])
            if not np.isfinite(value):
                continue
            patch = zos[j - half : j + half + 1, i - half : i + half + 1]
            if not np.isfinite(patch).any():
                continue
            if value == float(np.nanmax(patch)):
                rows.append({"ssh_extremum_type": "ssh_max", "seed_i": i, "seed_j": j, "ssh_value_m": value})
            elif value == float(np.nanmin(patch)):
                rows.append({"ssh_extremum_type": "ssh_min", "seed_i": i, "seed_j": j, "ssh_value_m": value})
    rows.sort(key=lambda item: abs(float(item["ssh_value_m"])), reverse=True)
    return rows[:max_candidates] if max_candidates > 0 else rows


def seeded_speed_min(speed: np.ndarray, seed_i: int, seed_j: int, radius_cells: int) -> tuple[int, int, float]:
    yy, xx = np.ogrid[: speed.shape[0], : speed.shape[1]]
    mask = (xx - seed_i) ** 2 + (yy - seed_j) ** 2 <= radius_cells**2
    mask &= np.isfinite(speed)
    if not np.any(mask):
        return seed_i, seed_j, float("nan")
    flat = np.where(mask.ravel())[0]
    pick = int(flat[np.nanargmin(speed.ravel()[flat])])
    cj, ci = np.unravel_index(pick, speed.shape)
    return int(ci), int(cj), float(speed[cj, ci])


def circle_offsets(radius_cells: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    n = max(16, int(round(8 * radius_cells)))
    for theta in np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, n, endpoint=False):
        point = (int(round(radius_cells * math.cos(theta))), int(round(radius_cells * math.sin(theta))))
        if not points or points[-1] != point:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) - np.asarray(b) + math.pi) % (2.0 * math.pi) - math.pi


def hua_circle_check(u: np.ndarray, v: np.ndarray, center_i: int, center_j: int, radius_cells: int, params: HuaParams) -> dict[str, object]:
    offsets = circle_offsets(radius_cells)
    ii = np.asarray([center_i + dx for dx, _ in offsets], dtype=int)
    jj = np.asarray([center_j + dy for _, dy in offsets], dtype=int)
    inside = (ii >= 0) & (ii < u.shape[1]) & (jj >= 0) & (jj < u.shape[0])
    uu = np.full(len(offsets), np.nan)
    vv = np.full(len(offsets), np.nan)
    uu[inside] = u[jj[inside], ii[inside]]
    vv[inside] = v[jj[inside], ii[inside]]
    sp = np.hypot(uu, vv)
    finite = np.isfinite(sp) & (sp > 1e-10)
    if finite.mean() < params.min_finite_fraction:
        return {"circle_passed": False, "dominant_failure": "invalid_velocity", "radius_cells": radius_cells}
    angles = np.arctan2(vv, uu)
    positive = negative = 0
    for n in range(len(offsets)):
        m = (n + 1) % len(offsets)
        if not (finite[n] and finite[m]):
            return {"circle_passed": False, "dominant_failure": "invalid_velocity", "radius_cells": radius_cells}
        ratio = float(sp[m] / sp[n])
        if ratio > params.speed_ratio_max or ratio < 1.0 / params.speed_ratio_max:
            return {"circle_passed": False, "dominant_failure": "velocity_ratio", "radius_cells": radius_cells}
        dtheta = float(angle_diff(angles[n], angles[m]))
        if abs(math.degrees(dtheta)) > params.angle_jump_max_deg:
            return {"circle_passed": False, "dominant_failure": "angle_jump", "radius_cells": radius_cells}
        positive += int(dtheta > 0)
        negative += int(dtheta < 0)
    max_exceptions = int(math.floor(radius_cells / 5.0) + 1 + params.direction_exception_extra)
    if min(positive, negative) > max_exceptions:
        return {"circle_passed": False, "dominant_failure": "too_many_direction_exceptions", "radius_cells": radius_cells}
    dx = np.asarray([p[0] for p in offsets], dtype=float)
    dy = np.asarray([p[1] for p in offsets], dtype=float)
    th = np.arctan2(dy, dx)
    tx = -np.sin(th)
    ty = np.cos(th)
    tangent_cos = np.abs((uu * tx + vv * ty) / np.maximum(sp, 1e-12))
    tangent_fraction = float(np.sum(finite & (tangent_cos >= math.cos(math.radians(params.tangent_tolerance_deg)))) / np.sum(finite))
    if tangent_fraction < params.min_tangent_fraction:
        return {"circle_passed": False, "dominant_failure": "tangent_alignment", "radius_cells": radius_cells, "tangent_pass_fraction": tangent_fraction}
    half = len(offsets) // 2
    reversal_ok = 0
    reversal_total = 0
    symmetry_ok = 0
    symmetry_total = 0
    for n in range(half):
        m = (n + half) % len(offsets)
        if not (finite[n] and finite[m]):
            continue
        reversal_total += 1
        reversal_ok += int(uu[n] * uu[m] + vv[n] * vv[m] < 0)
        symmetry_total += 1
        symmetry_ok += int(abs(abs(float(angle_diff(angles[n], angles[m]))) - math.pi) <= math.radians(params.symmetry_tolerance_deg))
    reversal_fraction = float(reversal_ok / reversal_total) if reversal_total else 0.0
    symmetry_fraction = float(symmetry_ok / symmetry_total) if symmetry_total else 0.0
    if reversal_fraction < params.min_reversal_fraction:
        return {"circle_passed": False, "dominant_failure": "opposite_reversal", "radius_cells": radius_cells, "opposite_reversal_fraction": reversal_fraction}
    tangential = uu * tx + vv * ty
    circulation_sign = float(np.sign(np.nanmedian(tangential[finite])))
    return {
        "circle_passed": True,
        "dominant_failure": "none",
        "radius_cells": radius_cells,
        "tangent_pass_fraction": tangent_fraction,
        "symmetry_pass_fraction": symmetry_fraction,
        "opposite_reversal_fraction": reversal_fraction,
        "circulation_sign": circulation_sign,
    }


def hua_surface_centers(lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, zos: np.ndarray, params: HuaParams, *, max_candidates: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    speed = np.hypot(u, v)
    for seed in local_extrema(zos, params.ssh_window_cells, max_candidates=max_candidates):
        ci, cj, center_speed = seeded_speed_min(speed, int(seed["seed_i"]), int(seed["seed_j"]), params.surface_search_cells)
        best: dict[str, object] | None = None
        first_fail: dict[str, object] | None = None
        for radius in range(params.start_radius_cells, params.max_radius_cells + 1):
            check = hua_circle_check(u, v, ci, cj, radius, params)
            if bool(check["circle_passed"]):
                best = check
            else:
                first_fail = check
                break
        check = best or first_fail or {"circle_passed": False, "dominant_failure": "no_circle", "radius_cells": np.nan}
        circulation = float(check.get("circulation_sign", np.nan))
        if np.isfinite(circulation) and circulation != 0:
            f_sign = 1.0 if float(lat[cj, ci]) >= 0 else -1.0
            polarity = "cyclonic" if circulation == f_sign else "anticyclonic"
        else:
            polarity = "cyclonic" if seed["ssh_extremum_type"] == "ssh_max" else "anticyclonic"
        if best is None:
            continue
        rows.append(
            {
                "method": "hua_b3_surface",
                "i": ci,
                "j": cj,
                "lon": float(lon[cj, ci]),
                "lat": float(lat[cj, ci]),
                "polarity": polarity,
                "seed_lon": float(lon[int(seed["seed_j"]), int(seed["seed_i"])]),
                "seed_lat": float(lat[int(seed["seed_j"]), int(seed["seed_i"])]),
                "center_speed_ms": center_speed,
                "accepted_radius_cells": float(check["radius_cells"]),
                "dominant_failure": str(check.get("dominant_failure", "none")),
            }
        )
    return rows


def choose_window_from_summary(summary_csv: Path, *, days: int) -> SmokeWindow:
    if not summary_csv.exists():
        return SmokeWindow(parse_date("2020-01-01"), parse_date("2020-01-07"), 126.0, 130.0, 25.0, 29.0)
    counts: dict[date, int] = {}
    lon_values: dict[date, list[float]] = {}
    lat_values: dict[date, list[float]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                day = parse_date(row["date"])
                layers = int(float(row.get("pass_layers", "0")))
                lon = float(row["surface_center_lon"])
                lat = float(row["surface_center_lat"])
            except Exception:
                continue
            if layers <= 0 or not (np.isfinite(lon) and np.isfinite(lat)):
                continue
            counts[day] = counts.get(day, 0) + 1
            lon_values.setdefault(day, []).append(lon)
            lat_values.setdefault(day, []).append(lat)
    if not counts:
        return SmokeWindow(parse_date("2020-01-01"), parse_date("2020-01-07"), 126.0, 130.0, 25.0, 29.0)
    best_start = max(counts, key=lambda d: sum(counts.get(d + timedelta(days=k), 0) for k in range(days)))
    chosen_days = [best_start + timedelta(days=k) for k in range(days)]
    lons = [x for d in chosen_days for x in lon_values.get(d, [])]
    lats = [y for d in chosen_days for y in lat_values.get(d, [])]
    if not lons or not lats:
        return SmokeWindow(best_start, best_start + timedelta(days=days - 1), 126.0, 130.0, 25.0, 29.0)
    lon0 = float(np.nanmedian(lons))
    lat0 = float(np.nanmedian(lats))
    return SmokeWindow(best_start, best_start + timedelta(days=days - 1), lon0 - 2.0, lon0 + 2.0, lat0 - 2.0, lat0 + 2.0)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a)) / 1000.0


def match_centers(nencioli: list[dict[str, object]], hua: list[dict[str, object]], *, max_km: float) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    matches: list[dict[str, object]] = []
    used_hua: set[int] = set()
    for n_idx, n in enumerate(nencioli):
        best_idx = None
        best_dist = float("inf")
        for h_idx, h in enumerate(hua):
            if h_idx in used_hua or n["polarity"] != h["polarity"]:
                continue
            dist = haversine_km(float(n["lon"]), float(n["lat"]), float(h["lon"]), float(h["lat"]))
            if dist < best_dist:
                best_dist = dist
                best_idx = h_idx
        if best_idx is not None and best_dist <= max_km:
            used_hua.add(best_idx)
            h = hua[best_idx]
            matches.append(
                {
                    "nencioli_index": n_idx,
                    "hua_index": best_idx,
                    "polarity": n["polarity"],
                    "distance_km": best_dist,
                    "nencioli_lon": n["lon"],
                    "nencioli_lat": n["lat"],
                    "hua_lon": h["lon"],
                    "hua_lat": h["lat"],
                }
            )
    matched_n = {int(m["nencioli_index"]) for m in matches}
    n_only = [row for i, row in enumerate(nencioli) if i not in matched_n]
    h_only = [row for i, row in enumerate(hua) if i not in used_hua]
    return matches, n_only, h_only


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_day_overlay(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    nencioli: list[dict[str, object]],
    hua: list[dict[str, object]],
    *,
    day: date,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    speed = np.hypot(u, v)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.pcolormesh(lon, lat, speed, shading="auto", cmap="magma")
    step = max(1, min(u.shape) // 20)
    ax.quiver(lon[::step, ::step], lat[::step, ::step], u[::step, ::step], v[::step, ::step], color="white", alpha=0.55, scale=8)
    for rows, marker, color, label in (
        (nencioli, "o", "#22c55e", "Nencioli VG"),
        (hua, "x", "#38bdf8", "Hua b3 surface"),
    ):
        if rows:
            ax.scatter([float(r["lon"]) for r in rows], [float(r["lat"]) for r in rows], marker=marker, s=58, color=color, label=label, linewidths=1.8)
    ax.set_title(f"Surface velocity centers {day:%Y-%m-%d}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="|u',v'| (m/s)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_counts(path: Path, daily_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    days = [str(r["date"]) for r in daily_rows]
    x = np.arange(len(days))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(x - 0.18, [int(r["nencioli_count"]) for r in daily_rows], width=0.36, label="Nencioli VG")
    ax.bar(x + 0.18, [int(r["hua_count"]) for r in daily_rows], width=0.36, label="Hua b3 surface")
    ax.set_xticks(x, days, rotation=45, ha="right")
    ax.set_ylabel("surface centers")
    ax.set_title("Daily surface center counts")
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_offsets(path: Path, matches: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    dist = [float(m["distance_km"]) for m in matches]
    ax.hist(dist, bins=20, color="#6366f1", alpha=0.85)
    ax.set_xlabel("matched center distance (km)")
    ax.set_ylabel("count")
    ax.set_title("Nencioli VG vs Hua b3 matched-center offsets")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original Nencioli VG surface centers with Hua b3 surface centers on a small Kuroshiou crop.")
    parser.add_argument("--filter-root", default="/root/autodl-fs/kuroshiou/Filter")
    parser.add_argument("--hua-summary-csv", default="/root/autodl-fs/kuroshiou/result/hua_b3_start2_detection/frame_object_summary.csv")
    parser.add_argument("--output-dir", default="/root/autodl-fs/kuroshiou/nencioli_vg_vs_hua_b3_smoke")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--lon-min", type=float, default=np.nan)
    parser.add_argument("--lon-max", type=float, default=np.nan)
    parser.add_argument("--lat-min", type=float, default=np.nan)
    parser.add_argument("--lat-max", type=float, default=np.nan)
    parser.add_argument("--max-hua-candidates-per-day", type=int, default=80)
    parser.add_argument("--match-km", type=float, default=25.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.start and args.end and np.isfinite(args.lon_min) and np.isfinite(args.lon_max) and np.isfinite(args.lat_min) and np.isfinite(args.lat_max):
        window = SmokeWindow(parse_date(args.start), parse_date(args.end), args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    else:
        window = choose_window_from_summary(Path(args.hua_summary_csv), days=args.days)

    params = HuaParams()
    all_nencioli: list[dict[str, object]] = []
    all_hua: list[dict[str, object]] = []
    all_matches: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    for day in date_range(window.start, window.end):
        lon, lat, u, v, zos = read_surface_day(Path(args.filter_root), day, window)
        nencioli = nencioli_vg_centers(lon, lat, u, v)
        hua = hua_surface_centers(lon, lat, u, v, zos, params, max_candidates=args.max_hua_candidates_per_day)
        matches, n_only, h_only = match_centers(nencioli, hua, max_km=args.match_km)
        for rows in (nencioli, hua, matches, n_only, h_only):
            for row in rows:
                row["date"] = f"{day:%Y-%m-%d}"
        all_nencioli.extend(nencioli)
        all_hua.extend(hua)
        all_matches.extend(matches)
        daily.append(
            {
                "date": f"{day:%Y-%m-%d}",
                "nencioli_count": len(nencioli),
                "hua_count": len(hua),
                "matched_count": len(matches),
                "nencioli_only": len(n_only),
                "hua_only": len(h_only),
            }
        )
        plot_day_overlay(output_dir / "figures" / f"day_{day:%Y%m%d}_velocity_centers_overlay.png", lon, lat, u, v, nencioli, hua, day=day)

    write_csv(output_dir / "nencioli_surface_centers.csv", all_nencioli)
    write_csv(output_dir / "hua_b3_surface_centers.csv", all_hua)
    write_csv(output_dir / "center_matches.csv", all_matches)
    write_csv(output_dir / "daily_counts.csv", daily)
    plot_counts(output_dir / "figures" / "daily_counts_by_polarity.png", daily)
    plot_offsets(output_dir / "figures" / "center_offset_histogram.png", all_matches)

    summary = {
        "window": window.__dict__,
        "nencioli_total": len(all_nencioli),
        "hua_total": len(all_hua),
        "matched_total": len(all_matches),
        "match_fraction_of_nencioli": len(all_matches) / len(all_nencioli) if all_nencioli else 0.0,
        "match_fraction_of_hua": len(all_matches) / len(all_hua) if all_hua else 0.0,
        "median_match_distance_km": float(np.nanmedian([float(m["distance_km"]) for m in all_matches])) if all_matches else np.nan,
        "parameters": {"nencioli": {"a": 4, "b": 3}, "hua": params.__dict__, "match_km": args.match_km},
    }
    (output_dir / "center_match_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Nencioli VG 与 Hua b3 表层速度中心小窗口对比",
        "",
        f"- 窗口：`{window.start}` 到 `{window.end}`，lon `{window.lon_min:.2f}..{window.lon_max:.2f}`，lat `{window.lat_min:.2f}..{window.lat_max:.2f}`。",
        f"- Nencioli VG 中心数：`{len(all_nencioli)}`；Hua b3 surface 中心数：`{len(all_hua)}`；同极性 `{args.match_km:g} km` 内匹配：`{len(all_matches)}`。",
        f"- Nencioli 匹配率：`{summary['match_fraction_of_nencioli']:.3f}`；Hua 匹配率：`{summary['match_fraction_of_hua']:.3f}`；匹配距离中位数：`{summary['median_match_distance_km']:.2f} km`。",
        "",
        "## 方法差异",
        "",
        "- Nencioli 原始 VG 是 velocity-only：先找 `v` 的零穿越与两侧增强，再检查 `u` 的两侧反转与增强，然后在 `b=3` 邻域找局地速度极小，最后要求中心周围 `a-1` 方框边界速度向量完成单调旋转。",
        "- 我们 Hua b3 surface 是 SSH/zos seed + velocity hybrid：先用 `zos_glor` 局地极值给候选，再在候选附近找 `|u',v'|` 低值，并用圆周路径检查速度连续性、切向性、对称性和两侧速度反转。",
        "- 因此两者不完全等价：Nencioli 更纯粹依赖速度零穿越几何；Hua b3 会受 SSH seed 和圆周判据影响，更接近我们后续 3D 深层扩展的入口。",
    ]
    (output_dir / "method_difference_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
