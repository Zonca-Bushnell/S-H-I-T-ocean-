"""Native OFES field helpers extracted from the retired rebuild-W workflow.

This module contains no reconstructed-W formula or executable rebuild pipeline.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

EARTH_RADIUS_M = 6_371_000.0
@dataclass(frozen=True)
class SelectedObject:
    hua_object_id: str
    date: str
    polarity: str
    pass_layers: int
    max_jump_km: float
    max_jump_over_r: float
    radius_km: float
    center_lon: float
    center_lat: float

def classify_native_w_multipole(grid: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    if 'ofes_w_native_meso_m_s' in grid:
        native_source = 'ofes_w_native_meso_m_s'
    elif 'ofes_w_native_raw_m_s' in grid:
        native_source = 'ofes_w_native_raw_m_s'
    else:
        native_source = 'ofes_w_native_m_s'
    native = np.asarray(grid[native_source], dtype='f4')
    depth = np.asarray(grid['depth_m'], dtype='f4')
    x = np.asarray(grid['x_over_r'], dtype='f4')
    y = np.asarray(grid['y_over_r'], dtype='f4')
    dmin = float(getattr(args, 'multipole_depth_min_m', 300.0))
    dmax = float(getattr(args, 'multipole_depth_max_m', 500.0))
    r_inner = float(getattr(args, 'multipole_radius_inner_r', 0.0))
    r_outer = float(getattr(args, 'multipole_radius_outer_r', 1.0))
    n_azimuth = max(12, int(getattr(args, 'multipole_azimuth_count', 60)))
    min_valid_azimuth_fraction = float(getattr(args, 'multipole_min_valid_azimuth_fraction', 0.8))
    min_sector_valid_fraction = float(getattr(args, 'multipole_min_sector_valid_fraction', 0.35))
    boundary_max_nan_fraction = float(getattr(args, 'multipole_boundary_max_nan_fraction', 0.35))
    min_amp = float(getattr(args, 'multipole_min_amp_1e6_m_s', 0.5)) * 1e-06
    min_snr = float(getattr(args, 'multipole_min_snr', 2.0))
    min_harmonic_dominance = float(getattr(args, 'multipole_min_harmonic_dominance', 1.1))
    r_inner = max(0.0, min(r_inner, r_outer))
    depth_mask = (depth >= min(dmin, dmax)) & (depth <= max(dmin, dmax))
    if not np.any(depth_mask):
        depth_mask = np.isfinite(depth)
    with np.errstate(invalid='ignore'):
        field = np.nanmean(native[depth_mask], axis=0)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    rr = np.hypot(xx, yy)
    boundary_inner = max(r_inner, r_outer * 0.85)
    boundary = (rr >= boundary_inner) & (rr <= r_outer)
    boundary_count = int(np.count_nonzero(boundary))
    boundary_nan_fraction = float(np.count_nonzero(boundary & ~np.isfinite(field)) / boundary_count) if boundary_count else 1.0
    values = np.full(n_azimuth, np.nan, dtype='f4')
    sector_std = np.full(n_azimuth, np.nan, dtype='f4')
    sector_valid_fraction = np.zeros(n_azimuth, dtype='f4')
    sector_valid_count = np.zeros(n_azimuth, dtype='i4')
    width = 2.0 * np.pi / float(n_azimuth)
    radial_count = max(24, int(math.ceil((r_outer - r_inner) / max(float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else 0.1, 1e-06))))
    radial_r = np.linspace(r_inner, r_outer, radial_count, dtype='f8')
    radial_length = max(r_outer - r_inner, 1e-12)
    for i in range(n_azimuth):
        center = (i + 0.5) * width
        ray_x = radial_r * math.cos(center)
        ray_y = radial_r * math.sin(center)
        if x.size < 2 or y.size < 2:
            continue
        ray_ix = (ray_x - float(x[0])) / float(x[1] - x[0])
        ray_iy = (ray_y - float(y[0])) / float(y[1] - y[0])
        samples = ndimage.map_coordinates(field, np.vstack([ray_iy, ray_ix]), order=1, mode='constant', cval=np.nan)
        finite_ray = np.isfinite(samples)
        valid_count = int(np.count_nonzero(finite_ray))
        sector_valid_count[i] = valid_count
        sector_valid_fraction[i] = valid_count / float(radial_count)
        if sector_valid_fraction[i] >= min_sector_valid_fraction:
            if valid_count >= 2:
                integral = float(np.trapezoid(samples[finite_ray], radial_r[finite_ray]))
                values[i] = integral / radial_length
            elif valid_count == 1:
                values[i] = float(samples[finite_ray][0])
            sector_std[i] = float(np.nanstd(samples[finite_ray]))
    valid = np.isfinite(values)
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / float(n_azimuth)
    qc_status = 'pass'
    centered = values.copy()
    centered_smooth = values.copy()
    zero_crossings = -1
    dominant_mode = -1
    harmonic_dominance = float('nan')
    amplitude = float('nan')
    snr = float('nan')
    phase_rad = float('nan')
    phase_valid = False
    if valid_count < max(4, int(math.ceil(n_azimuth * min_valid_azimuth_fraction))):
        label = 'qc_sparse_azimuth'
        qc_status = 'fail_sparse_azimuth'
    elif boundary_nan_fraction > boundary_max_nan_fraction:
        label = 'qc_missing_boundary'
        qc_status = 'fail_missing_boundary'
    else:
        centered[valid] = centered[valid] - float(np.nanmean(centered[valid]))
        filled = fill_circular_values(centered)
        centered_smooth = ndimage.gaussian_filter1d(filled.astype('f4'), sigma=1.0, mode='wrap')
        zero_crossings = count_circular_zero_crossings(centered_smooth)
        amplitude = q95_abs(centered_smooth)
        finite_std = sector_std[np.isfinite(sector_std)]
        if finite_std.size:
            noise = float(np.nanmedian(finite_std / np.sqrt(np.maximum(sector_valid_count[np.isfinite(sector_std)], 1))))
        else:
            diff = np.diff(np.r_[centered_smooth, centered_smooth[0]])
            noise = float(1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff))) / math.sqrt(2.0))
        noise = max(noise, 1e-12)
        snr = amplitude / noise
        harmonics = circular_harmonic_amplitudes(centered_smooth, max_mode=4)
        centers = (np.arange(n_azimuth, dtype='f8') + 0.5) * width
        c1 = np.nanmean(centered_smooth.astype('f8') * np.exp(-1j * centers))
        if np.isfinite(c1.real) and np.isfinite(c1.imag) and (np.abs(c1) > 0.0):
            phase_rad = float(-np.angle(c1))
            phase_valid = True
        if harmonics.size:
            dominant_mode = int(np.nanargmax(harmonics) + 1)
            sorted_h = np.sort(harmonics[np.isfinite(harmonics)])
            harmonic_dominance = float(sorted_h[-1] / max(sorted_h[-2], 1e-12)) if sorted_h.size >= 2 else float('inf')
        if amplitude < min_amp:
            label = 'qc_low_amplitude'
            qc_status = 'fail_low_amplitude'
        elif snr < min_snr:
            label = 'qc_low_snr'
            qc_status = 'fail_low_snr'
        elif harmonic_dominance < min_harmonic_dominance:
            label = 'qc_mixed_modes'
            qc_status = 'fail_mixed_modes'
        elif zero_crossings <= 1:
            label = 'monopole'
        elif zero_crossings == 2 and dominant_mode == 1:
            label = 'dipole'
        elif zero_crossings == 4 and dominant_mode == 2:
            label = 'quadrupole'
        else:
            label = 'other'
    return {'multipole_class': label, 'multipole_qc_status': qc_status, 'multipole_zero_crossings': int(zero_crossings), 'multipole_valid_azimuth_count': valid_count, 'multipole_azimuth_count': int(n_azimuth), 'multipole_valid_azimuth_fraction': float(valid_fraction), 'multipole_sector_valid_fraction': sector_valid_fraction, 'multipole_boundary_nan_fraction': float(boundary_nan_fraction), 'multipole_amp_1e6_m_s': float(amplitude * 1000000.0) if np.isfinite(amplitude) else float('nan'), 'multipole_snr': float(snr), 'multipole_dominant_mode': int(dominant_mode), 'multipole_harmonic_dominance': float(harmonic_dominance), 'multipole_azimuth_native_w_1e6_m_s': values * 1000000.0, 'multipole_azimuth_centered_native_w_1e6_m_s': centered_smooth * 1000000.0, 'multipole_depth_range_m': f'{min(dmin, dmax):g}-{max(dmin, dmax):g}', 'multipole_radius_ring_r': f'{r_inner:g}-{r_outer:g}', 'multipole_classifier': 'native_w_60azimuth_true_radial_integral_qc', 'multipole_radial_sample_count': int(radial_count), 'multipole_native_w_source': native_source, 'dipole_phase_angle_rad': float(phase_rad), 'dipole_phase_angle_deg': float(np.degrees(phase_rad)) if phase_valid else float('nan'), 'dipole_phase_valid': bool(phase_valid and label == 'dipole')}

def fill_circular_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype='f8')
    if not np.any(np.isfinite(finite)):
        return np.full(finite.shape, np.nan, dtype='f8')
    filled = finite.copy()
    valid_idx = np.flatnonzero(np.isfinite(filled))
    if valid_idx.size == 1:
        filled[~np.isfinite(filled)] = filled[valid_idx[0]]
        return filled
    idx = np.arange(filled.size)
    extended_idx = np.r_[valid_idx, valid_idx[0] + filled.size]
    extended_values = np.r_[filled[valid_idx], filled[valid_idx[0]]]
    return np.interp(idx, extended_idx, extended_values, period=filled.size)

def circular_harmonic_amplitudes(values: np.ndarray, max_mode: int=4) -> np.ndarray:
    filled = fill_circular_values(values)
    if not np.any(np.isfinite(filled)):
        return np.full(max_mode, np.nan, dtype='f8')
    filled = filled - float(np.nanmean(filled))
    coeff = np.fft.rfft(filled)
    out = np.full(max_mode, np.nan, dtype='f8')
    for mode in range(1, max_mode + 1):
        if mode < coeff.size:
            out[mode - 1] = float(np.abs(coeff[mode]))
    return out

def count_circular_zero_crossings(values: np.ndarray) -> int:
    finite = np.asarray(values, dtype='f8')
    if not np.any(np.isfinite(finite)):
        return -1
    valid_idx = np.flatnonzero(np.isfinite(finite))
    if valid_idx.size < 2:
        return -1
    filled = fill_circular_values(finite)
    eps = max(float(np.nanpercentile(np.abs(filled), 20)) * 0.05, 1e-20)
    signs = np.sign(filled)
    signs[np.abs(filled) <= eps] = 0.0
    for i in range(signs.size):
        if signs[i] == 0:
            prev = signs[(i - 1) % signs.size]
            nxt = signs[(i + 1) % signs.size]
            signs[i] = prev if prev != 0 else nxt
    return int(np.count_nonzero(signs != np.roll(signs, 1)))

def sample_stack(raw: np.memmap, meta, lon_grid: np.ndarray, lat_grid: np.ndarray, nlev: int) -> np.ndarray:
    lon = meta.x.values
    lat = meta.y.values
    lon_step = float(np.nanmedian(np.diff(lon)))
    lat_step = float(np.nanmedian(np.diff(lat)))
    ii = (lon_grid - float(lon[0])) / lon_step % len(lon)
    jj = np.clip((lat_grid - float(lat[0])) / lat_step, 0, len(lat) - 1)
    out = np.empty((nlev, lon_grid.shape[0], lon_grid.shape[1]), dtype='f4')
    coords = np.vstack([ii.ravel(), jj.ravel()])
    i_floor = np.floor(ii).astype(int)
    i_ceil = np.ceil(ii).astype(int)
    j_floor = np.floor(jj).astype(int)
    j_ceil = np.ceil(jj).astype(int)
    crosses_lon_wrap = bool(np.ptp(ii) > len(lon) / 2)
    if not crosses_lon_wrap:
        i0 = max(0, int(np.nanmin(i_floor)) - 2)
        i1 = min(len(lon) - 1, int(np.nanmax(i_ceil)) + 2)
        j0 = max(0, int(np.nanmin(j_floor)) - 2)
        j1 = min(len(lat) - 1, int(np.nanmax(j_ceil)) + 2)
        if i1 > i0 and j1 > j0:
            local_coords = np.vstack([(ii - i0).ravel(), (jj - j0).ravel()])
            for k in range(nlev):
                layer = np.asarray(raw[i0:i1 + 1, j0:j1 + 1, k], dtype='f4')
                layer[np.abs(layer) > 1e+30] = np.nan
                sampled = ndimage.map_coordinates(layer, local_coords, order=1, mode='nearest', cval=np.nan).reshape(lon_grid.shape)
                out[k] = sampled
            return out
    for k in range(nlev):
        layer = np.asarray(raw[:, :, k], dtype='f4')
        layer[np.abs(layer) > 1e+30] = np.nan
        sampled = ndimage.map_coordinates(layer, coords, order=1, mode='wrap', cval=np.nan).reshape(lon_grid.shape)
        out[k] = sampled
    return out

def align_native_w_vertical(native_w_raw: np.ndarray, source_depth_m: np.ndarray, target_depth_m: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, object]]:
    source = np.asarray(source_depth_m, dtype='f8')
    target = np.asarray(target_depth_m, dtype='f8')
    raw = np.asarray(native_w_raw, dtype='f4')
    if source.size == 0 or target.size == 0:
        aligned = np.full((target.size, raw.shape[1], raw.shape[2]), np.nan, dtype='f4')
    else:
        flat_raw = raw.reshape(raw.shape[0], -1).astype('f8', copy=False)
        nsource, ncolumns = flat_raw.shape
        valid = np.isfinite(flat_raw)
        indices = np.arange(nsource, dtype='i4')[:, None]
        previous_valid = np.maximum.accumulate(np.where(valid, indices, -1), axis=0)
        following_valid = np.minimum.accumulate(np.where(valid, indices, nsource)[::-1], axis=0)[::-1]
        first_valid = following_valid[0]
        last_valid = previous_valid[-1]
        columns = np.arange(ncolumns)
        flat_out = np.full((target.size, ncolumns), np.nan, dtype='f4')
        for target_index, target_value in enumerate(target):
            below = int(np.searchsorted(source, target_value, side='right') - 1)
            above = int(np.searchsorted(source, target_value, side='left'))
            lower = previous_valid[below] if below >= 0 else np.full(ncolumns, -1, dtype='i4')
            upper = following_valid[above] if above < nsource else np.full(ncolumns, nsource, dtype='i4')
            lower = np.where(lower < 0, first_valid, lower)
            upper = np.where(upper >= nsource, last_valid, upper)
            usable = (first_valid < nsource) & (lower >= 0) & (upper < nsource)
            lower_safe = np.clip(lower, 0, nsource - 1)
            upper_safe = np.clip(upper, 0, nsource - 1)
            lower_values = flat_raw[lower_safe, columns]
            upper_values = flat_raw[upper_safe, columns]
            span = source[upper_safe] - source[lower_safe]
            fraction = np.divide(float(target_value) - source[lower_safe], span, out=np.zeros(ncolumns, dtype='f8'), where=span != 0.0)
            values = lower_values + fraction * (upper_values - lower_values)
            flat_out[target_index, usable] = values[usable].astype('f4')
        aligned = flat_out.reshape((target.size, raw.shape[1], raw.shape[2]))
    return (aligned, {'native_w_vertical_alignment': 'layer_center', 'native_w_ctl_position': 'T cell bottom', 'native_w_comparison_position': 'interpolated to prho/u/v layer-center depths', 'native_w_source_depth_m': source.astype('f4'), 'native_w_target_depth_m': target.astype('f4'), 'native_w_alignment_extrapolated_top': bool(source.size and target.size and (target[0] < source[0])), 'native_w_alignment_extrapolated_bottom': bool(source.size and target.size and (target[-1] > source[-1]))})

def local_lon_lat_grid(xxr: np.ndarray, yyr: np.ndarray, lon0: float, lat0: float, radius_km: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my = meters_per_degree(lat0)
    lon = (lon0 + xxr * radius_km * 1000.0 / mx) % 360.0
    lat = lat0 + yyr * radius_km * 1000.0 / my
    return (lon, lat)

def normalize_density_units(prho: np.ndarray) -> np.ndarray:
    finite = prho[np.isfinite(prho)]
    if finite.size and float(np.nanmedian(finite)) > 1000.0:
        return prho - 1000.0
    return prho

def cressman_kernel_2d(x: np.ndarray, y: np.ndarray, radius_r: float) -> np.ndarray:
    dx = float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else radius_r
    dy = float(np.nanmedian(np.abs(np.diff(y)))) if y.size > 1 else radius_r
    nx = max(1, int(math.ceil(radius_r / max(dx, 1e-12))))
    ny = max(1, int(math.ceil(radius_r / max(dy, 1e-12))))
    ox = np.arange(-nx, nx + 1, dtype='f8') * dx
    oy = np.arange(-ny, ny + 1, dtype='f8') * dy
    xx, yy = np.meshgrid(ox, oy, indexing='xy')
    d2 = xx * xx + yy * yy
    r2 = radius_r * radius_r
    weights = np.where(d2 <= r2, (r2 - d2) / np.maximum(r2 + d2, 1e-12), 0.0)
    return weights.astype('f4')

def cressman_map_3d(values: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(values.shape, np.nan, dtype='f4')
    support = np.zeros(values.shape, dtype=bool)
    for k in range(values.shape[0]):
        valid = np.isfinite(values[k])
        if not np.any(valid):
            continue
        numerator = ndimage.convolve(np.where(valid, values[k], 0.0).astype('f4'), kernel, mode='constant', cval=0.0)
        denominator = ndimage.convolve(valid.astype('f4'), kernel, mode='constant', cval=0.0)
        layer = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-08)
        out[k] = layer
        support[k] = np.isfinite(layer)
    return (out, support)

def finalize_sum_count(total: np.ndarray, count: np.ndarray, min_count: int) -> np.ndarray:
    threshold = max(1, int(min_count))
    return np.divide(total, count, out=np.full(total.shape, np.nan, dtype='f4'), where=count >= threshold).astype('f4')

def draw_field_pillow(canvas: Image.Image, box: tuple[int, int, int, int], x: np.ndarray, y: np.ndarray, data: np.ndarray, vmin: float, vmax: float, title: str, main_font, small_font) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    available_w = box[2] - box[0] - 95
    available_h = box[3] - box[1] - 95
    side = min(available_w, available_h)
    left = box[0] + 55 + max(0, available_w - side) // 2
    top = box[1] + 38 + max(0, available_h - side) // 2
    plot = (left, top, left + side, top + side)
    img = array_to_rgb(data, vmin, vmax).resize((plot[2] - plot[0], plot[3] - plot[1]), Image.Resampling.BILINEAR)
    canvas.paste(img, plot[:2])
    th = np.linspace(0, 2 * np.pi, 361)
    visible_radius = min(abs(float(x[0])), abs(float(x[-1])), abs(float(y[0])), abs(float(y[-1])))
    for rr in [radius for radius in (1.0, 4.0) if radius <= visible_radius + 1e-06]:
        pts = [map_xy(plot, rr * math.cos(t), rr * math.sin(t), float(x[0]), float(x[-1]), float(y[0]), float(y[-1])) for t in th]
        draw.line(pts, fill=(0, 0, 0), width=2)
    cx, cy = map_xy(plot, 0.0, 0.0, float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 0, 0))
    draw.text((box[0] + 10, box[1] + 10), title, fill=(30, 36, 48), font=main_font)
    draw.text((plot[0], box[3] - 30), 'x/R', fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 10, plot[1]), 'y/R', fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 10, box[1] + 35), f'+/- {vmax:.2g} x10^-6 m/s', fill=(80, 88, 100), font=small_font)
    colorbar_box = (plot[2] + 14, plot[1], min(plot[2] + 34, box[2] - 12), plot[3])
    draw_colorbar_pillow(canvas, colorbar_box, vmin, vmax, small_font)

def preferred_native_w_composite_key(composite: dict[str, object], section: bool) -> tuple[str, str]:
    prefix = 'section_' if section else 'composite_'
    candidates = [('ofes_w_native_meso_aligned_m_s', 'native W meso 50-500 km, mode-1 aligned'), ('ofes_w_native_meso_m_s', 'native W meso 50-500 km, unaligned'), ('ofes_w_native_m_s', 'native W layer-center')]
    for key, label in candidates:
        full_key = f'{prefix}{key}'
        if full_key in composite:
            return (full_key, label)
    return (f'{prefix}ofes_w_native_m_s', 'native W layer-center')

def plot_composite_native_w_cross_section_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite['x_over_r'], dtype='f4')
    depth = np.asarray(composite['depth_m'], dtype='f4')
    native_key, native_label = preferred_native_w_composite_key(composite, section=True)
    native = np.asarray(composite[native_key], dtype='f4')
    vals = native[np.isfinite(native)]
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1000000.0 if vals.size else 1.0, 1e-12)
    canvas = Image.new('RGB', (1050, 900), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    main_font = load_font(20)
    small_font = load_font(15)
    class_text = str(composite.get('multipole_class', 'all'))
    region_text = str(composite.get('region', '')).strip()
    region_suffix = f' | {region_text}' if region_text else ''
    method = str(composite.get('composite_method', 'cressman'))
    method_text = f"Cressman R={composite['cressman_radius_r']}R" if method == 'cressman' else 'direct pointwise mean'
    draw.text((35, 28), f"Native W crossing section | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | {class_text}{region_suffix}", fill=(20, 24, 32), font=title_font)
    draw.text((35, 68), f"{native_label}; objects={composite['object_count']}; {method_text}; min objects={composite['cressman_min_objects']}; +/-{lim:.2g} x10^-6 m/s", fill=(80, 88, 100), font=small_font)
    draw_section_field_pillow(canvas, (70, 125, 980, 830), x, depth, native * 1000000.0, -lim, lim, native_label, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix('.pdf'), 'PDF', resolution=180.0)

def plot_composite_native_w_focus_pillow(composite: dict[str, object], png_path: Path) -> None:
    x = np.asarray(composite['x_over_r'], dtype='f4')
    y = np.asarray(composite['y_over_r'], dtype='f4')
    depth = np.asarray(composite['depth_m'], dtype='f4')
    native_key, native_label = preferred_native_w_composite_key(composite, section=False)
    native = np.asarray(composite[native_key], dtype='f4')
    rows = [('0-100 m surface', depth_average_indices(depth, 0.0, 100.0)), ('300-500 m mean', depth_average_indices(depth, 300.0, 500.0)), ('450 m slice', np.array([int(np.nanargmin(np.abs(depth - 450.0)))]))]
    fields2d: list[tuple[str, np.ndarray]] = []
    for row_name, idx in rows:
        with np.errstate(invalid='ignore'):
            fields2d.append((row_name, np.nanmean(native[idx], axis=0)))
    finite_parts = [arr[np.isfinite(arr)] for _, arr in fields2d if np.isfinite(arr).any()]
    vals = np.concatenate(finite_parts) if finite_parts else np.array([], dtype='f4')
    lim = max(float(np.nanpercentile(np.abs(vals), 95)) * 1000000.0 if vals.size else 1.0, 1e-12)
    canvas = Image.new('RGB', (760, 1770), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26)
    main_font = load_font(16)
    small_font = load_font(12)
    region_text = str(composite.get('region', '')).strip()
    region_suffix = f' | {region_text}' if region_text else ''
    draw.text((35, 25), f"Native W focus | {lat_label(float(composite['target_lat']))} | {composite['polarity']} | {composite.get('multipole_class', 'all')}{region_suffix}", fill=(20, 24, 32), font=title_font)
    draw.text((35, 62), f"{native_label}; objects={composite['object_count']}; +/-{lim:.2g} x10^-6 m/s", fill=(80, 88, 100), font=small_font)
    for row_idx, (row_name, data) in enumerate(fields2d):
        box = (70, 130 + row_idx * 535, 680, 630 + row_idx * 535)
        draw.text((20, 105 + row_idx * 535), row_name, fill=(30, 36, 48), font=main_font)
        draw_field_pillow(canvas, box, x, y, data * 1000000.0, -lim, lim, native_label, main_font, small_font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path)
    canvas.save(png_path.with_suffix('.pdf'), 'PDF', resolution=180.0)

def draw_section_field_pillow(canvas: Image.Image, box: tuple[int, int, int, int], x: np.ndarray, depth: np.ndarray, data: np.ndarray, vmin: float, vmax: float, title: str, main_font, small_font, unit_label: str='1e-6 m/s') -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(180, 186, 196), width=1)
    plot = (box[0] + 70, box[1] + 42, box[2] - 40, box[3] - 58)
    img = array_to_rgb(data, vmin, vmax).resize((plot[2] - plot[0], plot[3] - plot[1]), Image.Resampling.BILINEAR)
    canvas.paste(img, plot[:2])
    draw.rectangle(plot, outline=(100, 110, 125), width=1)
    zero_x = section_x_to_px(plot, 0.0, float(x[0]), float(x[-1]))
    draw.line((zero_x, plot[1], zero_x, plot[3]), fill=(30, 30, 30), width=2)
    for rr in [-4, -2, 0, 2, 4]:
        tx = section_x_to_px(plot, float(rr), float(x[0]), float(x[-1]))
        draw.line((tx, plot[3], tx, plot[3] + 5), fill=(80, 88, 100), width=1)
        draw.text((tx - 10, plot[3] + 9), f'{rr:g}', fill=(80, 88, 100), font=small_font)
    for dz in [0, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
        if depth[0] <= dz <= depth[-1]:
            ty = section_depth_to_py(plot, float(dz), float(depth[0]), float(depth[-1]))
            draw.line((plot[0] - 5, ty, plot[0], ty), fill=(80, 88, 100), width=1)
            draw.text((box[0] + 10, ty - 8), f'{dz}', fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 12, box[1] + 10), title, fill=(30, 36, 48), font=main_font)
    draw.text((plot[0], box[3] - 30), 'x/R', fill=(80, 88, 100), font=small_font)
    draw.text((box[0] + 8, plot[1]), 'depth m', fill=(80, 88, 100), font=small_font)
    colorbar_box = (plot[2] + 10, plot[1], min(plot[2] + 30, box[2] - 8), plot[3])
    draw_colorbar_pillow(canvas, colorbar_box, vmin, vmax, small_font, unit_label)

def section_x_to_px(plot: tuple[int, int, int, int], value: float, xmin: float, xmax: float) -> int:
    return plot[0] + int(round((value - xmin) / max(xmax - xmin, 1e-12) * (plot[2] - plot[0] - 1)))

def section_depth_to_py(plot: tuple[int, int, int, int], value: float, dmin: float, dmax: float) -> int:
    return plot[1] + int(round((value - dmin) / max(dmax - dmin, 1e-12) * (plot[3] - plot[1] - 1)))

def array_to_rgb(values: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    arr = np.asarray(values, dtype='f4')
    nan = ~np.isfinite(arr)
    scaled = np.clip((arr - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    colors = rdbu_r_colors(np.nan_to_num(scaled, nan=0.5))
    colors[nan] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(colors, mode='RGB')

def draw_colorbar_pillow(canvas: Image.Image, box: tuple[int, int, int, int], vmin: float, vmax: float, small_font, unit_label: str='1e-6 m/s') -> None:
    draw = ImageDraw.Draw(canvas)
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    values = np.linspace(vmax, vmin, max(2, box[3] - box[1]), dtype='f4')[:, None]
    bar = array_to_rgb(values, vmin, vmax).resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR)
    canvas.paste(bar, box[:2])
    draw.rectangle(box, outline=(95, 103, 116), width=1)
    tick_x0 = box[2] + 2
    ticks = [(vmax, box[1], f'{vmax:.2g}'), (0.0, (box[1] + box[3]) // 2, '0'), (vmin, box[3], f'{vmin:.2g}')]
    for _, y, label in ticks:
        draw.line((box[2], y, tick_x0 + 4, y), fill=(75, 85, 99), width=1)
        draw.text((tick_x0 + 7, y - 8), label, fill=(75, 85, 99), font=small_font)
    draw.text((box[0] - 2, box[3] + 8), unit_label, fill=(75, 85, 99), font=small_font)

def rdbu_r_colors(scaled: np.ndarray) -> np.ndarray:
    blue = np.array([5, 113, 176], dtype='f4')
    white = np.array([247, 247, 247], dtype='f4')
    red = np.array([202, 0, 32], dtype='f4')
    rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
    low = scaled <= 0.5
    rgb[low] = (blue * (1.0 - scaled[low, None] / 0.5) + white * (scaled[low, None] / 0.5)).astype(np.uint8)
    high = ~low
    rgb[high] = (white * (1.0 - (scaled[high, None] - 0.5) / 0.5) + red * ((scaled[high, None] - 0.5) / 0.5)).astype(np.uint8)
    return rgb

def q95_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.nanpercentile(np.abs(finite), 95)) if finite.size else float('nan')

def depth_average_indices(depth: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    arr = np.asarray(depth, dtype='f8')
    mask = (arr >= min(dmin, dmax)) & (arr <= max(dmin, dmax))
    idx = np.flatnonzero(mask)
    if idx.size:
        return idx
    target = 0.5 * (float(dmin) + float(dmax))
    return np.array([int(np.nanargmin(np.abs(arr - target)))], dtype=int)

def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    return (math.pi * EARTH_RADIUS_M * math.cos(lat_rad) / 180.0, math.pi * EARTH_RADIUS_M / 180.0)

def lat_label(lat: float) -> str:
    if lat > 0:
        return f'{abs(lat):g}N'
    if lat < 0:
        return f'{abs(lat):g}S'
    return 'EQ'

def load_font(size: int) -> ImageFont.ImageFont:
    for name in ['arial.ttf', 'DejaVuSans.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

def map_xy(box: tuple[int, int, int, int], x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[int, int]:
    px = box[0] + int(round((x - xmin) / max(xmax - xmin, 1e-12) * (box[2] - box[0] - 1)))
    py = box[3] - int(round((y - ymin) / max(ymax - ymin, 1e-12) * (box[3] - box[1] - 1)))
    return (px, py)
