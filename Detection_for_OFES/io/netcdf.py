"""Small NetCDF-grid helpers used by the canonical surface-input stage."""
from __future__ import annotations

import numpy as np


def regrid_scalar_to_velocity(
    field_latlon: np.ndarray,
    lon_source: np.ndarray,
    lat_source: np.ndarray,
    lon_target: np.ndarray,
    lat_target: np.ndarray,
) -> np.ndarray:
    """Linearly remap a periodic scalar grid to the OFES velocity grid."""
    source = np.asarray(field_latlon, dtype="f8")
    lon_source = np.asarray(lon_source, dtype="f8")
    lat_source = np.asarray(lat_source, dtype="f8")
    lon_target = np.asarray(lon_target, dtype="f8")
    lat_target = np.asarray(lat_target, dtype="f8")
    lon_ext = np.concatenate([lon_source, [lon_source[0] + 360.0]])
    source_ext = np.concatenate([source, source[:, :1]], axis=1)
    wrapped_target = np.where(lon_target < lon_source[0], lon_target + 360.0, lon_target)
    zonal = np.empty((source_ext.shape[0], len(lon_target)), dtype="f8")
    for row in range(source_ext.shape[0]):
        zonal[row] = np.interp(wrapped_target, lon_ext, source_ext[row], left=np.nan, right=np.nan)
    output = np.empty((len(lat_target), len(lon_target)), dtype="f8")
    for column in range(zonal.shape[1]):
        output[:, column] = np.interp(lat_target, lat_source, zonal[:, column], left=np.nan, right=np.nan)
    return output.astype("f4")
