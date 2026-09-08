from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import shutil
import tarfile

import numpy as np


UNDEF_ABS_THRESHOLD = 1.0e30


@dataclass(frozen=True)
class AxisDef:
    count: int
    mode: str
    start: float | None = None
    step: float | None = None
    levels: tuple[float, ...] = ()

    @property
    def values(self) -> np.ndarray:
        if self.mode == "linear":
            if self.start is None or self.step is None:
                raise ValueError("Linear axis is missing start/step metadata.")
            return self.start + self.step * np.arange(self.count, dtype=np.float64)
        if self.mode == "levels":
            return np.asarray(self.levels, dtype=np.float64)
        raise ValueError(f"Unsupported axis mode: {self.mode}")


@dataclass(frozen=True)
class TimeDef:
    count: int
    start: date
    step_days: int

    def date_at(self, index: int) -> date:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return self.start + timedelta(days=index * self.step_days)


@dataclass(frozen=True)
class VarDef:
    name: str
    levels: int
    units: str
    description: str


@dataclass(frozen=True)
class CtlMetadata:
    path: Path
    dataset_template: str
    undef: float
    x: AxisDef
    y: AxisDef
    z: AxisDef
    t: TimeDef
    endian: str
    variable: VarDef

    @property
    def shape_xy(self) -> tuple[int, int]:
        return (self.x.count, self.y.count)

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return (self.x.count, self.y.count, self.z.count)

    @property
    def grid_signature(self) -> dict[str, float | int]:
        return {
            "nx": self.x.count,
            "ny": self.y.count,
            "nz": self.z.count,
            "lon_start": float(self.x.values[0]),
            "lon_step": float(self.x.values[1] - self.x.values[0]) if self.x.count > 1 else 0.0,
            "lat_start": float(self.y.values[0]),
            "lat_step": float(self.y.values[1] - self.y.values[0]) if self.y.count > 1 else 0.0,
        }


def parse_ctl(path: str | Path) -> CtlMetadata:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    dataset_template = ""
    undef = np.nan
    x_axis: AxisDef | None = None
    y_axis: AxisDef | None = None
    z_axis: AxisDef | None = None
    t_def: TimeDef | None = None
    endian = "native"
    variable: VarDef | None = None

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        parts = raw.split()
        key = parts[0].upper()
        if key == "DSET":
            dataset_template = " ".join(parts[1:])
        elif key == "UNDEF":
            undef = float(parts[1])
        elif key == "XDEF":
            x_axis = _parse_axis(parts, lines, i)
        elif key == "YDEF":
            y_axis = _parse_axis(parts, lines, i)
        elif key == "ZDEF":
            z_axis = _parse_axis(parts, lines, i)
            if len(parts) >= 3 and parts[2].lower() == "levels":
                i = _axis_levels_end_index(lines, i, int(parts[1]))
        elif key == "TDEF":
            t_def = _parse_tdef(parts)
        elif key == "OPTIONS":
            lower = {p.lower() for p in parts[1:]}
            if "big_endian" in lower:
                endian = "big"
            elif "little_endian" in lower:
                endian = "little"
        elif key == "VARS":
            if i + 1 >= len(lines):
                raise ValueError(f"VARS section is empty in {path}")
            variable = _parse_var_line(lines[i + 1])
            i += 1
        i += 1

    missing = [
        name
        for name, value in {
            "DSET": dataset_template,
            "UNDEF": None if np.isnan(undef) else undef,
            "XDEF": x_axis,
            "YDEF": y_axis,
            "ZDEF": z_axis,
            "TDEF": t_def,
            "VARS": variable,
        }.items()
        if value is None or value == ""
    ]
    if missing:
        raise ValueError(f"{path} is missing required ctl fields: {', '.join(missing)}")

    return CtlMetadata(
        path=path,
        dataset_template=dataset_template,
        undef=undef,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        t=t_def,
        endian=endian,
        variable=variable,
    )


def data_path(root: str | Path, variable: str, day: date | str) -> Path:
    root = Path(root)
    day = _coerce_date(day)
    name = _canonical_variable(variable)
    filename_prefix = "salinity" if name == "salinity" else name
    return root / "daily" / f"y{day.year:04d}" / name / f"{filename_prefix}.{day:%m.%d.%Y}.dta"


def archive_dir(root: str | Path, variable: str, year: int) -> Path:
    return Path(root) / "compressed" / _canonical_variable(variable) / f"{year:04d}"


def archive_candidates(root: str | Path, variable: str, day: date | str) -> list[Path]:
    day = _coerce_date(day)
    name = _canonical_variable(variable)
    base = archive_dir(root, name, day.year)
    prefixes = {
        "salinity": ["sali", "salinity"],
        "pressur": ["pressur"],
        "eta": ["eta"],
        "temp": ["temp"],
        "prho": ["prho"],
        "u": ["u"],
        "v": ["v"],
        "w": ["w"],
    }[name]
    files: list[Path] = []
    suffix = _archive_suffix_for_day(name, day)
    for prefix in prefixes:
        if suffix:
            files.extend(sorted(base.glob(f"{prefix}_y{day.year:04d}_{suffix}.tgz")))
        files.extend(sorted(base.glob(f"{prefix}_y{day.year:04d}.tgz")))
    if not files:
        for prefix in prefixes:
            files.extend(sorted(base.glob(f"{prefix}_y{day.year:04d}*.tgz")))
    return sorted(set(files))


def expected_archive_member(variable: str, day: date | str) -> str:
    day = _coerce_date(day)
    name = _canonical_variable(variable)
    member_dir = {
        "eta": "eta",
        "pressur": "pressur",
        "u": "u_vel",
        "v": "v_vel",
        "w": "w_vel",
        "temp": "temp",
        "salinity": "salinity",
        "prho": "prho",
    }[name]
    filename_prefix = "salinity" if name == "salinity" else name
    return f"daily/y{day.year:04d}/{member_dir}/{filename_prefix}.{day:%m.%d.%Y}.dta"


def ensure_daily_file(root: str | Path, variable: str, day: date | str, cache_root: str | Path | None = None) -> Path:
    """Return a daily `.dta`, extracting it from OFES `.tgz` archives if needed."""
    root = Path(root)
    day = _coerce_date(day)
    direct = data_path(root, variable, day)
    if direct.exists():
        return direct

    name = _canonical_variable(variable)
    cache_base = Path(cache_root) if cache_root is not None else root / "daily"
    filename_prefix = "salinity" if name == "salinity" else name
    cached = cache_base / f"y{day.year:04d}" / name / f"{filename_prefix}.{day:%m.%d.%Y}.dta"
    if cached.exists():
        return cached

    target_member = expected_archive_member(name, day)
    for archive in archive_candidates(root, name, day):
        with tarfile.open(archive, "r:gz") as tf:
            member = _find_tar_member(tf, target_member)
            if member is None:
                continue
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            tmp.replace(cached)
            return cached
    raise FileNotFoundError(f"No OFES daily file found for {name} on {day.isoformat()}")


def expected_dta_bytes(meta: CtlMetadata, n_levels: int | None = None) -> int:
    nz = meta.z.count if n_levels is None else int(n_levels)
    return int(meta.x.count * meta.y.count * nz * np.dtype(">f4").itemsize)


def require_daily_file(root: str | Path, variable: str, day: date | str, expected_bytes: int | None = None) -> Path:
    """Return an existing daily `.dta` without falling back to `.tgz` extraction."""
    day = _coerce_date(day)
    name = _canonical_variable(variable)
    path = data_path(root, name, day)
    if expected_bytes is None:
        expected_bytes = expected_dta_bytes(parse_ctl(ctl_path(root, name)))
    if _complete_file(path, expected_bytes):
        return path
    if path.exists():
        actual = path.stat().st_size
        raise ValueError(f"{path} has {actual} bytes, expected {expected_bytes}; daily file is incomplete.")
    raise FileNotFoundError(f"Missing extracted OFES daily file for {name} on {day.isoformat()}: {path}")


def prefetch_daily_files(
    root: str | Path,
    variable: str,
    days: list[date] | tuple[date, ...],
    cache_root: str | Path | None = None,
    workers: int = 1,
) -> list[Path]:
    """Cache daily files with one archive scan per `.tgz` where possible.

    OFES monthly 3-D variables are split into multi-day `.tgz` files. Calling
    `ensure_daily_file` once per day reopens and rescans the same gzip stream.
    This helper groups requested days by archive and extracts all required
    members during a single scan of each archive.
    """
    root = Path(root)
    name = _canonical_variable(variable)
    days = [_coerce_date(day) for day in days]
    meta = parse_ctl(ctl_path(root, name))
    expected_bytes = expected_dta_bytes(meta)
    cache_base = Path(cache_root) if cache_root is not None else root / "daily"

    resolved: dict[date, Path] = {}
    grouped: dict[Path, list[tuple[date, str, Path]]] = {}
    for day in days:
        direct = data_path(root, name, day)
        if _complete_file(direct, expected_bytes):
            resolved[day] = direct
            continue

        cached = cached_daily_path(cache_base, name, day)
        if _complete_file(cached, expected_bytes):
            resolved[day] = cached
            continue

        archives = archive_candidates(root, name, day)
        if not archives:
            raise FileNotFoundError(f"No archive found for {name} on {day.isoformat()}")
        grouped.setdefault(archives[0], []).append((day, expected_archive_member(name, day), cached))

    if grouped:
        jobs = [
            {
                "archive": str(archive),
                "items": [(day.isoformat(), member, str(target)) for day, member, target in items],
                "expected_bytes": expected_bytes,
            }
            for archive, items in grouped.items()
        ]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as pool:
                futures = [pool.submit(_extract_archive_members_job, job) for job in jobs]
                for future in as_completed(futures):
                    for day_text, path_text in future.result():
                        resolved[_coerce_date(day_text)] = Path(path_text)
        else:
            for job in jobs:
                for day_text, path_text in _extract_archive_members_job(job):
                    resolved[_coerce_date(day_text)] = Path(path_text)

    missing = [day.isoformat() for day in days if day not in resolved]
    if missing:
        raise FileNotFoundError(f"Could not cache {name} days: {', '.join(missing)}")
    return [resolved[day] for day in days]


def cached_daily_path(cache_base: str | Path, variable: str, day: date | str) -> Path:
    day = _coerce_date(day)
    name = _canonical_variable(variable)
    filename_prefix = "salinity" if name == "salinity" else name
    return Path(cache_base) / f"y{day.year:04d}" / name / f"{filename_prefix}.{day:%m.%d.%Y}.dta"


def ctl_path(root: str | Path, variable: str) -> Path:
    root = Path(root)
    name = _canonical_variable(variable)
    ctl_name = "salinity.ctl" if name == "salinity" else f"{name}.ctl"
    preferred = root / "metadata" / "ctl" / ctl_name
    if preferred.exists():
        return preferred
    legacy = root / ctl_name
    return legacy


def read_dta(path: str | Path, meta: CtlMetadata, level_index: int | None = None) -> np.ndarray:
    path = Path(path)
    dtype = np.dtype(">f4" if meta.endian == "big" else "<f4" if meta.endian == "little" else "f4")
    nx, ny, nz = meta.x.count, meta.y.count, meta.z.count
    expected_values = nx * ny * nz
    expected_bytes = expected_values * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes, expected {expected_bytes} "
            f"for shape ({nx}, {ny}, {nz}) and dtype {dtype}."
        )

    arr = np.fromfile(path, dtype=dtype)
    if nz == 1:
        arr = arr.reshape((nx, ny), order="F")
    else:
        arr = arr.reshape((nx, ny, nz), order="F")
        if level_index is not None:
            arr = arr[:, :, level_index]
    arr = arr.astype(np.float32, copy=False)
    arr[np.abs(arr) > UNDEF_ABS_THRESHOLD] = np.nan
    return arr


def open_dta_memmap(path: str | Path, meta: CtlMetadata) -> np.memmap:
    path = Path(path)
    dtype = np.dtype(">f4" if meta.endian == "big" else "<f4" if meta.endian == "little" else "f4")
    nx, ny, nz = meta.x.count, meta.y.count, meta.z.count
    expected_bytes = nx * ny * nz * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes, expected {expected_bytes} "
            f"for shape ({nx}, {ny}, {nz}) and dtype {dtype}."
        )
    shape = (nx, ny) if nz == 1 else (nx, ny, nz)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, order="F")


def read_dta_layer_latlon(path: str | Path, meta: CtlMetadata, level_index: int | None = None) -> np.ndarray:
    mm = open_dta_memmap(path, meta)
    if meta.z.count == 1:
        arr = np.asarray(mm[:, :].T, dtype=np.float32)
    else:
        if level_index is None:
            raise ValueError("level_index is required for 3-D lat/lon layer reads.")
        arr = np.asarray(mm[:, :, int(level_index)].T, dtype=np.float32)
    arr[np.abs(arr) > UNDEF_ABS_THRESHOLD] = np.nan
    return arr


def read_variable(
    root: str | Path,
    variable: str,
    day: date | str,
    level_index: int | None = None,
    cache_root: str | Path | None = None,
) -> np.ndarray:
    meta = parse_ctl(ctl_path(root, variable))
    return read_dta(ensure_daily_file(root, variable, day, cache_root), meta, level_index=level_index)


def read_variable_latlon(
    root: str | Path,
    variable: str,
    day: date | str,
    level_index: int | None = None,
    cache_root: str | Path | None = None,
) -> np.ndarray:
    meta = parse_ctl(ctl_path(root, variable))
    return read_dta_layer_latlon(ensure_daily_file(root, variable, day, cache_root), meta, level_index=level_index)


def read_variable_latlon_daily_only(
    root: str | Path,
    variable: str,
    day: date | str,
    level_index: int | None = None,
) -> np.ndarray:
    meta = parse_ctl(ctl_path(root, variable))
    return read_dta_layer_latlon(require_daily_file(root, variable, day, expected_dta_bytes(meta)), meta, level_index=level_index)


def read_ssh(root: str | Path, day: date | str, cache_root: str | Path | None = None) -> np.ndarray:
    eta = read_variable(root, "eta", day, cache_root=cache_root)
    pressur = read_variable(root, "pressur", day, cache_root=cache_root)
    return eta - (pressur - 1000.0)


def read_ssh_latlon(root: str | Path, day: date | str, cache_root: str | Path | None = None) -> np.ndarray:
    eta = read_variable_latlon(root, "eta", day, cache_root=cache_root)
    pressur = read_variable_latlon(root, "pressur", day, cache_root=cache_root)
    return eta - (pressur - 1000.0)


def read_ssh_latlon_daily_only(root: str | Path, day: date | str) -> np.ndarray:
    eta = read_variable_latlon_daily_only(root, "eta", day)
    pressur = read_variable_latlon_daily_only(root, "pressur", day)
    return eta - (pressur - 1000.0)


def summarize_array(name: str, arr: np.ndarray) -> dict[str, float | int | str | list[int]]:
    finite = np.isfinite(arr)
    if finite.any():
        values = arr[finite]
        minimum = float(np.nanmin(values))
        maximum = float(np.nanmax(values))
        mean = float(np.nanmean(values))
    else:
        minimum = maximum = mean = float("nan")
    return {
        "name": name,
        "shape": list(arr.shape),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "nan_fraction": float(np.isnan(arr).sum() / arr.size),
        "min": minimum,
        "max": maximum,
        "mean": mean,
    }


def _parse_axis(parts: list[str], lines: list[str], line_index: int) -> AxisDef:
    count = int(parts[1])
    mode = parts[2].lower()
    if mode == "linear":
        return AxisDef(count=count, mode="linear", start=float(parts[3]), step=float(parts[4]))
    if mode == "levels":
        levels: list[float] = []
        levels.extend(float(p) for p in parts[3:])
        cursor = line_index + 1
        while len(levels) < count and cursor < len(lines):
            levels.extend(float(p) for p in lines[cursor].split())
            cursor += 1
        if len(levels) != count:
            raise ValueError(f"Expected {count} levels, found {len(levels)}")
        return AxisDef(count=count, mode="levels", levels=tuple(levels))
    raise ValueError(f"Unsupported axis definition: {' '.join(parts)}")


def _axis_levels_end_index(lines: list[str], line_index: int, count: int) -> int:
    found = len(lines[line_index].split()[3:])
    cursor = line_index
    while found < count and cursor + 1 < len(lines):
        cursor += 1
        found += len(lines[cursor].split())
    return cursor


def _parse_tdef(parts: list[str]) -> TimeDef:
    if parts[2].lower() != "linear":
        raise ValueError(f"Unsupported TDEF: {' '.join(parts)}")
    return TimeDef(count=int(parts[1]), start=_parse_grads_date(parts[3]), step_days=_parse_day_step(parts[4]))


def _parse_var_line(raw: str) -> VarDef:
    match = re.match(r"^(\S+)\s+(\d+)\s+\S+\s+(?:\[([^\]]*)\]\s*)?(.*)$", raw.strip())
    if not match:
        raise ValueError(f"Cannot parse VARS line: {raw}")
    return VarDef(
        name=match.group(1),
        levels=int(match.group(2)),
        units=match.group(3) or "",
        description=(match.group(4) or "").strip(),
    )


def _parse_grads_date(value: str) -> date:
    return datetime.strptime(value.lower(), "%d%b%Y").date()


def _parse_day_step(value: str) -> int:
    match = re.match(r"^(\d+)dy$", value.lower())
    if not match:
        raise ValueError(f"Only day-based TDEF steps are supported, got {value}")
    return int(match.group(1))


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _canonical_variable(value: str) -> str:
    aliases = {
        "sali": "salinity",
        "salt": "salinity",
        "salinity": "salinity",
        "eta": "eta",
        "ssh": "ssh",
        "pressur": "pressur",
        "pressure": "pressur",
        "u": "u",
        "v": "v",
        "w": "w",
        "temp": "temp",
        "temperature": "temp",
        "prho": "prho",
    }
    try:
        return aliases[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown OFES variable alias: {value}") from exc


def _find_tar_member(tf: tarfile.TarFile, target: str) -> tarfile.TarInfo | None:
    normalized = target.replace("\\", "/").lstrip("./")
    for member in tf:
        name = member.name.replace("\\", "/").lstrip("./")
        if member.isfile() and name == normalized:
            return member
    return None


def _complete_file(path: Path, expected_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_bytes
    except OSError:
        return False


def _extract_archive_members_job(job: dict[str, object]) -> list[tuple[str, str]]:
    archive = Path(str(job["archive"]))
    expected_bytes = int(job["expected_bytes"])
    items = [(str(day), str(member), Path(str(target))) for day, member, target in job["items"]]  # type: ignore[index]
    wanted = {member.replace("\\", "/").lstrip("./"): (day, target) for day, member, target in items}
    extracted: list[tuple[str, str]] = []

    completed = []
    for day_text, target in wanted.values():
        if _complete_file(target, expected_bytes):
            completed.append((day_text, str(target)))
    extracted.extend(completed)

    remaining = {member: pair for member, pair in wanted.items() if not _complete_file(pair[1], expected_bytes)}
    if not remaining:
        return extracted

    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name.replace("\\", "/").lstrip("./")
            if not member.isfile() or name not in remaining:
                continue
            day_text, target = remaining.pop(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
            if tmp.stat().st_size != expected_bytes:
                raise ValueError(
                    f"{archive} member {name} extracted {tmp.stat().st_size} bytes, expected {expected_bytes}."
                )
            tmp.replace(target)
            extracted.append((day_text, str(target)))
            if not remaining:
                break

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise FileNotFoundError(f"{archive} is missing expected members: {missing}")
    print(f"[prefetch-archive] {archive.name} ready={len(extracted)}", flush=True)
    return extracted


def _archive_suffix_for_day(variable: str, day: date) -> str:
    name = _canonical_variable(variable)
    if name in {"eta", "pressur"}:
        return ""
    if day.month != 1:
        return ""
    if day.day <= 9:
        return "jan0"
    if day.day <= 19:
        return "jan1"
    if day.day <= 29:
        return "jan2"
    return "jan3"
