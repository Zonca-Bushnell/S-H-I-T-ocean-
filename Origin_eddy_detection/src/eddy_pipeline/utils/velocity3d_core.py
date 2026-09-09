from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import contourpy
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter, minimum_filter


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class LayerDetection:
    detection_id: int
    date: str
    depth_m: float
    depth_index: int
    polarity: str
    longitude: float
    latitude: float
    core_speed: float
    vorticity: float
    contour_lon: np.ndarray
    contour_lat: np.ndarray
    area_m2: float
    radius_m: float
    method: str
    reversal_passed: bool


@dataclass
class ContourContext:
    scalar: np.ndarray
    finite: np.ndarray
    generator: object
    speed_interp: RegularGridInterpolator | None = None


def haversine_km(lon0, lat0, lon1, lat1):
    lon0 = np.radians(lon0)
    lat0 = np.radians(lat0)
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    dlon = lon1 - lon0
    dlat = lat1 - lat0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a)) / 1000.0


def local_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    x = EARTH_RADIUS_M * np.cos(np.radians(lat0)) * np.radians(np.asarray(lon, dtype="f8") - lon0)
    y = EARTH_RADIUS_M * np.radians(np.asarray(lat, dtype="f8") - lat0)
    return x, y


def local_xy_to_lonlat(x_m: float, y_m: float, lon0: float, lat0: float) -> tuple[float, float]:
    lat = lat0 + np.degrees(y_m / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x_m / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return float(lon), float(lat)


def grid_spacing_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, float]:
    dlon = float(np.nanmedian(np.diff(lon)))
    dlat = float(np.nanmedian(np.diff(lat)))
    dx_by_lat = EARTH_RADIUS_M * np.cos(np.radians(lat)) * np.radians(dlon)
    dy = EARTH_RADIUS_M * np.radians(dlat)
    return dx_by_lat.astype("f8"), float(abs(dy))


def polygon_area_m2(lon: np.ndarray, lat: np.ndarray) -> float:
    if lon.size < 3:
        return 0.0
    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))
    x = EARTH_RADIUS_M * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def relative_vorticity(lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dx_by_lat, dy = grid_spacing_m(lon, lat)
    dvdx = np.gradient(v, axis=1) / dx_by_lat[:, None]
    dudy = np.gradient(u, axis=0) / dy
    return dvdx - dudy


def pseudo_streamfunction(lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dx_by_lat, dy = grid_spacing_m(lon, lat)
    u0 = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v0 = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    psi_x = np.cumsum(v0 * dx_by_lat[:, None], axis=1)
    psi_y = np.cumsum(-u0 * dy, axis=0)
    psi_x -= np.nanmean(psi_x)
    psi_y -= np.nanmean(psi_y)
    return 0.5 * (psi_x + psi_y)


def make_contour_context(scalar: np.ndarray, finite: np.ndarray | None = None, speed: np.ndarray | None = None) -> ContourContext:
    scalar_arr = np.asarray(scalar, dtype="f8")
    finite_arr = np.isfinite(scalar_arr) if finite is None else np.asarray(finite, dtype=bool)
    z = scalar_arr.copy()
    z[~finite_arr] = np.nan
    cg = contourpy.contour_generator(
        x=np.arange(scalar_arr.shape[1], dtype="f8"),
        y=np.arange(scalar_arr.shape[0], dtype="f8"),
        z=z,
        name="serial",
    )
    speed_interp = None
    if speed is not None:
        speed_values = np.asarray(speed, dtype="f8")
        if speed_values.shape == scalar_arr.shape and np.isfinite(speed_values).any():
            speed_interp = RegularGridInterpolator(
                (np.arange(scalar_arr.shape[0], dtype="f8"), np.arange(scalar_arr.shape[1], dtype="f8")),
                speed_values,
                bounds_error=False,
                fill_value=np.nan,
            )
    return ContourContext(scalar=scalar_arr, finite=finite_arr, generator=cg, speed_interp=speed_interp)


def _window_pixels(lon: np.ndarray, lat: np.ndarray, km: float) -> tuple[int, int]:
    dx_by_lat, dy = grid_spacing_m(lon, lat)
    dx = float(np.nanmedian(np.abs(dx_by_lat)))
    rx = max(2, int(round(km * 1000.0 / max(dx, 1.0))))
    ry = max(2, int(round(km * 1000.0 / max(dy, 1.0))))
    return rx, ry


def _sector_mean(values: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    block = values[max(y0, 0) : min(y1, values.shape[0]), max(x0, 0) : min(x1, values.shape[1])]
    if block.size == 0 or not np.isfinite(block).any():
        return np.nan
    return float(np.nanmean(block))


def has_core_velocity_reversal(
    u: np.ndarray,
    v: np.ndarray,
    j: int,
    i: int,
    rx: int,
    ry: int,
    min_speed: float,
) -> bool:
    west_v = _sector_mean(v, j - ry, j + ry + 1, i - rx, i)
    east_v = _sector_mean(v, j - ry, j + ry + 1, i + 1, i + rx + 1)
    south_u = _sector_mean(u, j - ry, j, i - rx, i + rx + 1)
    north_u = _sector_mean(u, j + 1, j + ry + 1, i - rx, i + rx + 1)
    vals = np.array([west_v, east_v, south_u, north_u], dtype="f8")
    if not np.all(np.isfinite(vals)):
        return False
    return (
        west_v * east_v < 0
        and south_u * north_u < 0
        and min(abs(west_v), abs(east_v), abs(south_u), abs(north_u)) >= min_speed
    )


def _closed_contour(
    lon: np.ndarray,
    lat: np.ndarray,
    scalar: np.ndarray,
    finite: np.ndarray,
    j: int,
    i: int,
    min_pixels: int,
    max_pixels: int,
    contour_levels: int,
    local_radius_px: int = 20,
    speed: np.ndarray | None = None,
    selection_mode: str = "area",
    contour_context: ContourContext | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if contour_context is not None:
        scalar = contour_context.scalar
        finite = contour_context.finite
    core_value = float(scalar[j, i])
    y0 = max(0, j - local_radius_px)
    y1 = min(scalar.shape[0], j + local_radius_px + 1)
    x0 = max(0, i - local_radius_px)
    x1 = min(scalar.shape[1], i + local_radius_px + 1)
    local = scalar[y0:y1, x0:x1]
    local_finite = np.isfinite(local)
    if local_finite.sum() < min_pixels:
        return None
    edge = np.concatenate(
        [local[0, local_finite[0]], local[-1, local_finite[-1]], local[local_finite[:, 0], 0], local[local_finite[:, -1], -1]]
    )
    if edge.size == 0:
        return None
    edge_value = float(np.nanmedian(edge))
    if not np.isfinite(core_value) or not np.isfinite(edge_value) or core_value == edge_value:
        return None

    if contour_context is None:
        cg = make_contour_context(scalar, finite).generator
        speed_interp = None
    else:
        cg = contour_context.generator
        speed_interp = contour_context.speed_interp
    levels = np.linspace(core_value, edge_value, contour_levels + 2)[1:-1]
    best = None
    best_score = -np.inf
    if speed_interp is None and speed is not None:
        speed_values = np.asarray(speed, dtype="f8")
        if speed_values.shape == scalar.shape and np.isfinite(speed_values).any():
            speed_interp = RegularGridInterpolator(
                (np.arange(scalar.shape[0], dtype="f8"), np.arange(scalar.shape[1], dtype="f8")),
                speed_values,
                bounds_error=False,
                fill_value=np.nan,
            )
    for level in levels:
        try:
            contours = cg.lines(float(level))
        except ValueError:
            continue
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            if np.hypot(*(contour[0] - contour[-1])) > 1.5:
                continue
            path = MplPath(contour)
            if not path.contains_point((float(i), float(j))):
                continue
            x = contour[:, 0]
            y = contour[:, 1]
            if (
                x.min() < i - local_radius_px
                or x.max() > i + local_radius_px
                or y.min() < j - local_radius_px
                or y.max() > j + local_radius_px
            ):
                continue
            area_pixels = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5
            if area_pixels < min_pixels or area_pixels > max_pixels:
                continue
            score = area_pixels
            if selection_mode == "max_speed" and speed_interp is not None:
                sampled = speed_interp(np.column_stack([y, x]))
                if not np.isfinite(sampled).any():
                    continue
                score = float(np.nanmean(sampled))
            if score > best_score:
                best = contour
                best_score = score
    if best is None:
        return None
    xs = np.clip(best[:, 0].astype("f8"), 0, lon.size - 1)
    ys = np.clip(best[:, 1].astype("f8"), 0, lat.size - 1)
    contour_lon = np.interp(xs, np.arange(lon.size), lon)
    contour_lat = np.interp(ys, np.arange(lat.size), lat)
    return contour_lon.astype("f4"), contour_lat.astype("f4")


def closed_contour_around_core(
    lon: np.ndarray,
    lat: np.ndarray,
    scalar: np.ndarray,
    core_lon: float,
    core_lat: float,
    *,
    min_pixels: int,
    max_pixels: int,
    contour_levels: int,
    local_radius_km: float = 120.0,
    speed: np.ndarray | None = None,
    selection_mode: str = "max_speed",
    contour_context: ContourContext | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    i = int(np.nanargmin(np.abs(np.asarray(lon) - core_lon)))
    j = int(np.nanargmin(np.abs(np.asarray(lat) - core_lat)))
    finite = contour_context.finite if contour_context is not None else np.isfinite(scalar)
    rx, ry = _window_pixels(lon, lat, local_radius_km)
    return _closed_contour(
        lon,
        lat,
        scalar,
        finite,
        j,
        i,
        min_pixels,
        max_pixels,
        contour_levels,
        local_radius_px=max(rx, ry),
        speed=speed,
        selection_mode=selection_mode,
        contour_context=contour_context,
    )


def equivalent_circle_lonlat(core_lon: float, core_lat: float, radius_m: float, n: int = 160) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n)
    x = float(radius_m) * np.cos(theta)
    y = float(radius_m) * np.sin(theta)
    lon = core_lon + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(core_lat))))
    lat = core_lat + np.degrees(y / EARTH_RADIUS_M)
    return lon.astype("f4"), lat.astype("f4")


def _recover_eta(xi: float, a0: float, a1: float, a2: float, a3: float, b0: float, b1: float, b2: float, b3: float, tol: float) -> tuple[bool, float]:
    den_u = a2 + a3 * xi
    den_v = b2 + b3 * xi
    num_u = -(a0 + a1 * xi)
    num_v = -(b0 + b1 * xi)
    if abs(den_u) >= abs(den_v) and abs(den_u) > tol:
        eta = num_u / den_u
        return bool(np.isfinite(eta)), float(eta)
    if abs(den_v) > tol:
        eta = num_v / den_v
        return bool(np.isfinite(eta)), float(eta)
    return False, np.nan


def _solve_bilinear_zero(u00, u10, u01, u11, v00, v10, v01, v11, root_tol: float, inside_tol: float) -> np.ndarray:
    a0 = u00
    a1 = u10 - u00
    a2 = u01 - u00
    a3 = u11 - u10 - u01 + u00
    b0 = v00
    b1 = v10 - v00
    b2 = v01 - v00
    b3 = v11 - v10 - v01 + v00
    mag = max(abs(v) for v in (u00, u10, u01, u11, v00, v10, v01, v11))
    res_tol = max(root_tol, 1e-7 * max(1.0, mag))
    c2 = b1 * a3 - b3 * a1
    c1 = b0 * a3 + b1 * a2 - b2 * a1 - b3 * a0
    c0 = b0 * a2 - b2 * a0
    xi_candidates = []
    if abs(c2) <= root_tol:
        if abs(c1) > root_tol:
            xi_candidates = [float(-c0 / c1)]
    else:
        roots = np.roots([c2, c1, c0])
        xi_candidates = [float(np.real(r)) for r in roots if abs(np.imag(r)) <= 1e-10]

    out = []
    for xi in xi_candidates:
        if xi < -inside_tol or xi > 1 + inside_tol:
            continue
        ok, eta = _recover_eta(xi, a0, a1, a2, a3, b0, b1, b2, b3, root_tol)
        if not ok or eta < -inside_tol or eta > 1 + inside_tol:
            continue
        uu = a0 + a1 * xi + a2 * eta + a3 * xi * eta
        vv = b0 + b1 * xi + b2 * eta + b3 * xi * eta
        if abs(uu) <= res_tol and abs(vv) <= res_tol:
            out.append([min(max(xi, 0.0), 1.0), min(max(eta, 0.0), 1.0)])

    for xi0, eta0 in ([0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]):
        xi, eta = xi0, eta0
        for _ in range(12):
            uu = a0 + a1 * xi + a2 * eta + a3 * xi * eta
            vv = b0 + b1 * xi + b2 * eta + b3 * xi * eta
            jac = np.array([[a1 + a3 * eta, a2 + a3 * xi], [b1 + b3 * eta, b2 + b3 * xi]], dtype="f8")
            if not np.isfinite(jac).all() or abs(np.linalg.det(jac)) < 1e-14:
                break
            delta = np.linalg.solve(jac, np.array([uu, vv], dtype="f8"))
            if not np.isfinite(delta).all():
                break
            xi -= float(delta[0])
            eta -= float(delta[1])
            if np.linalg.norm(delta) < 1e-10:
                break
        if xi < -inside_tol or xi > 1 + inside_tol or eta < -inside_tol or eta > 1 + inside_tol:
            continue
        uu = a0 + a1 * xi + a2 * eta + a3 * xi * eta
        vv = b0 + b1 * xi + b2 * eta + b3 * xi * eta
        if abs(uu) <= 3 * res_tol and abs(vv) <= 3 * res_tol:
            out.append([min(max(xi, 0.0), 1.0), min(max(eta, 0.0), 1.0)])

    if not out:
        return np.empty((0, 2), dtype="f8")
    arr = np.unique(np.round(np.asarray(out, dtype="f8"), 8), axis=0)
    return arr


def select_layer_center_speed_leading(
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    ref_lon: float,
    ref_lat: float,
    search_radius_km: float,
    *,
    zero_point_method: str = "hybrid",
    multi_root_policy: str = "depth_continuity",
    root_tol: float = 1e-8,
    inside_tol: float = 1e-6,
) -> dict:
    x, y = local_xy_m(lon, lat, ref_lon, ref_lat)
    xx, yy = np.meshgrid(x, y)
    domain = (
        np.isfinite(u)
        & np.isfinite(v)
        & (np.hypot(xx, yy) <= float(search_radius_km) * 1000.0)
    )
    info = {
        "longitude": np.nan,
        "latitude": np.nan,
        "center_method": "none",
        "speed_at_core": np.nan,
        "n_exact_roots": 0,
        "search_radius_km": float(search_radius_km),
    }
    if not domain.any():
        return info

    zero_point_method = str(zero_point_method).lower().strip()
    if zero_point_method not in {"hybrid", "exact", "fallback"}:
        zero_point_method = "hybrid"
    roots = []
    if zero_point_method in {"hybrid", "exact"}:
        jj, ii = np.where(domain)
        j0 = max(0, int(jj.min()) - 1)
        j1 = min(u.shape[0] - 1, int(jj.max()) + 1)
        i0 = max(0, int(ii.min()) - 1)
        i1 = min(u.shape[1] - 1, int(ii.max()) + 1)
        for j in range(j0, j1):
            for i in range(i0, i1):
                if not domain[j : j + 2, i : i + 2].any():
                    continue
                vals = [u[j, i], u[j, i + 1], u[j + 1, i], u[j + 1, i + 1], v[j, i], v[j, i + 1], v[j + 1, i], v[j + 1, i + 1]]
                if not np.isfinite(vals).all():
                    continue
                if min(vals[:4]) > 0 or max(vals[:4]) < 0 or min(vals[4:]) > 0 or max(vals[4:]) < 0:
                    continue
                xi_eta = _solve_bilinear_zero(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], max(root_tol, 1e-12), max(inside_tol, 1e-9))
                for xi, eta in xi_eta:
                    xr = x[i] + xi * (x[i + 1] - x[i])
                    yr = y[j] + eta * (y[j + 1] - y[j])
                    if np.hypot(xr, yr) <= float(search_radius_km) * 1000.0 * (1 + 1e-6):
                        roots.append([xr, yr])
        if roots:
            roots_arr = np.unique(np.round(np.asarray(roots, dtype="f8"), 6), axis=0)
            dist = np.hypot(roots_arr[:, 0], roots_arr[:, 1])
            k = int(np.argmin(dist))
            lon_c, lat_c = local_xy_to_lonlat(float(roots_arr[k, 0]), float(roots_arr[k, 1]), ref_lon, ref_lat)
            ii = int(np.nanargmin(np.abs(lon - lon_c)))
            jj = int(np.nanargmin(np.abs(lat - lat_c)))
            info.update(
                longitude=lon_c,
                latitude=lat_c,
                center_method="exact",
                speed_at_core=float(np.hypot(u[jj, ii], v[jj, ii])),
                n_exact_roots=int(roots_arr.shape[0]),
            )
            return info
        if zero_point_method == "exact":
            info["n_exact_roots"] = 0
            return info

    speed = np.full(u.shape, np.inf, dtype="f8")
    ok = domain & np.isfinite(u) & np.isfinite(v)
    speed[ok] = np.hypot(u[ok], v[ok])
    if np.isfinite(speed).any():
        j, i = np.unravel_index(int(np.nanargmin(speed)), speed.shape)
        info.update(
            longitude=float(lon[i]),
            latitude=float(lat[j]),
            center_method="fallback",
            speed_at_core=float(speed[j, i]),
            n_exact_roots=int(len(roots)),
        )
    return info


def detect_velocity_layer(
    lon: np.ndarray,
    lat: np.ndarray,
    depth_m: float,
    depth_index: int,
    u: np.ndarray,
    v: np.ndarray,
    date_value: str | date,
    *,
    core_window_km: float = 80.0,
    min_core_reversal_speed: float = 0.02,
    max_core_speed_percentile: float = 15.0,
    min_closed_contour_pixels: int = 12,
    max_closed_contour_pixels: int = 20000,
    min_core_distance_km: float = 40.0,
    max_candidates_per_layer: int = 80,
    contour_levels: int = 16,
    **_: object,
) -> list[LayerDetection]:
    date_label = date_value.isoformat() if isinstance(date_value, date) else str(date_value)
    finite = np.isfinite(u) & np.isfinite(v)
    if finite.sum() < min_closed_contour_pixels:
        return []
    speed = np.empty(u.shape, dtype="f8")
    np.hypot(u, v, out=speed)
    speed[~finite] = np.nan
    threshold = float(np.nanpercentile(speed, max_core_speed_percentile))
    rx, ry = _window_pixels(lon, lat, core_window_km)
    min_speed = minimum_filter(np.nan_to_num(speed, nan=np.inf), size=(2 * ry + 1, 2 * rx + 1), mode="nearest")
    candidates = np.argwhere((speed <= threshold) & (speed == min_speed) & finite)
    if candidates.size == 0:
        return []
    zeta = relative_vorticity(lon, lat, np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0))
    psi = pseudo_streamfunction(lon, lat, u, v)
    contour_context = make_contour_context(psi, finite)
    order = np.argsort(speed[candidates[:, 0], candidates[:, 1]])
    detections: list[LayerDetection] = []
    used_centers: list[tuple[float, float]] = []
    for cand_index in order[: max_candidates_per_layer * 4]:
        j, i = [int(v_) for v_ in candidates[cand_index]]
        if j < ry or i < rx or j >= u.shape[0] - ry or i >= u.shape[1] - rx:
            continue
        lon_i = float(lon[i])
        lat_j = float(lat[j])
        if any(haversine_km(lon_i, lat_j, old_lon, old_lat) < min_core_distance_km for old_lon, old_lat in used_centers):
            continue
        if not has_core_velocity_reversal(u, v, j, i, rx, ry, min_core_reversal_speed):
            continue
        z = float(zeta[j, i])
        if not np.isfinite(z) or z == 0:
            continue
        polarity = "cyclonic" if z > 0 else "anticyclonic"
        contour = _closed_contour(
            lon,
            lat,
            psi,
            finite,
            j,
            i,
            min_closed_contour_pixels,
            max_closed_contour_pixels,
            contour_levels,
            contour_context=contour_context,
        )
        if contour is None:
            continue
        contour_lon, contour_lat = contour
        area = polygon_area_m2(contour_lon, contour_lat)
        radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
        detections.append(
            LayerDetection(
                detection_id=len(detections),
                date=date_label,
                depth_m=float(depth_m),
                depth_index=int(depth_index),
                polarity=polarity,
                longitude=lon_i,
                latitude=lat_j,
                core_speed=float(speed[j, i]),
                vorticity=z,
                contour_lon=contour_lon,
                contour_lat=contour_lat,
                area_m2=area,
                radius_m=radius,
                method="velocity_core_reversal",
                reversal_passed=True,
            )
        )
        used_centers.append((lon_i, lat_j))
        if len(detections) >= max_candidates_per_layer:
            break
    return detections


def detect_surface_sla_fallback(
    lon: np.ndarray,
    lat: np.ndarray,
    adt: np.ndarray,
    depth_m: float,
    depth_index: int,
    date_value: str | date,
    *,
    surface_pixel_limit=(5, 2000),
    contour_levels: int = 16,
    min_core_distance_km: float = 40.0,
    max_candidates_per_layer: int = 80,
    **_: object,
) -> list[LayerDetection]:
    date_label = date_value.isoformat() if isinstance(date_value, date) else str(date_value)
    finite = np.isfinite(adt)
    if finite.sum() < surface_pixel_limit[0]:
        return []
    min_pixels, max_pixels = [int(v) for v in surface_pixel_limit]
    high = float(np.nanpercentile(adt, 85.0))
    low = float(np.nanpercentile(adt, 15.0))
    max_field = maximum_filter(np.nan_to_num(adt, nan=-np.inf), size=11, mode="nearest")
    min_field = minimum_filter(np.nan_to_num(adt, nan=np.inf), size=11, mode="nearest")
    candidates = [("anticyclonic", *idx) for idx in np.argwhere((adt == max_field) & (adt >= high) & finite)]
    candidates += [("cyclonic", *idx) for idx in np.argwhere((adt == min_field) & (adt <= low) & finite)]
    median_height = float(np.nanmedian(adt))
    contour_context = make_contour_context(adt, finite)
    candidates.sort(key=lambda item: abs(float(adt[int(item[1]), int(item[2])]) - median_height), reverse=True)
    detections: list[LayerDetection] = []
    used_centers: list[tuple[float, float]] = []
    for polarity, j_raw, i_raw in candidates[: max_candidates_per_layer * 4]:
        j = int(j_raw)
        i = int(i_raw)
        lon_i = float(lon[i])
        lat_j = float(lat[j])
        if any(haversine_km(lon_i, lat_j, old_lon, old_lat) < min_core_distance_km for old_lon, old_lat in used_centers):
            continue
        contour = _closed_contour(lon, lat, adt, finite, j, i, min_pixels, max_pixels, contour_levels, contour_context=contour_context)
        if contour is None:
            continue
        contour_lon, contour_lat = contour
        area = polygon_area_m2(contour_lon, contour_lat)
        radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
        detections.append(
            LayerDetection(
                detection_id=len(detections),
                date=date_label,
                depth_m=float(depth_m),
                depth_index=int(depth_index),
                polarity=polarity,
                longitude=lon_i,
                latitude=lat_j,
                core_speed=np.nan,
                vorticity=np.nan,
                contour_lon=contour_lon,
                contour_lat=contour_lat,
                area_m2=area,
                radius_m=radius,
                method="sla_surface",
                reversal_passed=False,
            )
        )
        used_centers.append((lon_i, lat_j))
        if len(detections) >= max_candidates_per_layer:
            break
    return detections
