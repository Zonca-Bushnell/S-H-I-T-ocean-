import os
from datetime import datetime
from pathlib import Path

DATASET_ID = "cmems_mod_glo_phy-all_my_0.25deg_P1D-m"
DATASET_VERSION = "202311"

RAW_DIR = Path(os.getenv("ACC_RAW_DIR", r"E:\DATA\Copernicus_Data\ACC_raw"))
OUTPUT_DIR = Path(os.getenv("ACC_OUTPUT_DIR", r"E:\DATA\Copernicus_Data\ACC"))
MANIFEST_DIR = Path(os.getenv("ACC_MANIFEST_DIR", "manifests"))
SELECTED_FILE_LIST = MANIFEST_DIR / "selected_files.txt"

GLOBAL_START = datetime(2022, 11, 1)
GLOBAL_END = datetime(2022, 12, 1)

SAMPLE_START = datetime(1993, 1, 1)
SAMPLE_END = datetime(1993, 1, 8)

VARIABLES = [
    "thetao_glor",
    "uo_glor",
    "vo_glor",
    "mlotst_glor",
    "zos_glor",
    "so_glor",
]

MIN_LONGITUDE = -179.0
MAX_LONGITUDE = 180.0
MIN_LATITUDE = -65.0
MAX_LATITUDE = -45.0
MIN_DEPTH = 0.5057600140571594
MAX_DEPTH = 1516.3636474609375

USERNAME_KEYS = [
    "COPERNICUSMARINE_SERVICE_USERNAME",
    "COPERNICUSMARINE_USERNAME",
    "CMEMS_USERNAME",
]
PASSWORD_KEYS = [
    "COPERNICUSMARINE_SERVICE_PASSWORD",
    "COPERNICUSMARINE_PASSWORD",
    "CMEMS_PASSWORD",
]


def selected_bounds(sample=False):
    if sample:
        return SAMPLE_START, min(SAMPLE_END, GLOBAL_END)
    return GLOBAL_START, GLOBAL_END


def build_time_windows(sample=False):
    start, end = selected_bounds(sample)
    if sample:
        return [
            (
                start,
                end,
                f"global_phy_sample_{start:%Y%m%d}_{end:%Y%m%d}.nc",
            )
        ]

    windows = []
    for year in range(start.year, end.year + 1):
        window_start = max(start, datetime(year, 1, 1))
        window_end = min(end, datetime(year + 1, 1, 1))
        if window_start < window_end:
            windows.append((window_start, window_end, f"global_phy_{year}.nc"))
    return windows
