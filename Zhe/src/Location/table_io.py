from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


def fallback_pickle_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".pkl")


def fallback_csv_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".csv")


def table_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() or fallback_pickle_path(path).exists() or fallback_csv_path(path).exists()


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pkl = fallback_pickle_path(path)
    csv = fallback_csv_path(path)
    marker = path.with_suffix(".parquet.unavailable.txt")
    try:
        df.to_parquet(path, index=index, engine="fastparquet")
        df.to_pickle(pkl)
        df.to_csv(csv, index=index)
        if marker.exists():
            marker.unlink()
    except Exception as exc:
        df.to_pickle(pkl)
        df.to_csv(csv, index=index)
        marker.write_text(
            "Parquet output was not written because the parquet engine failed.\n"
            f"Fallback files: {pkl.name}, {csv.name}\n"
            f"Original error: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )


def write_table_fast(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index, engine="fastparquet")


def write_parquet_parts_to_single(parts: list[str | Path], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pq is None:
        frames = [pd.read_parquet(part, engine="fastparquet") for part in parts if Path(part).exists()]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False, engine="fastparquet")
        else:
            pd.DataFrame().to_parquet(out_path, index=False, engine="fastparquet")
        return
    writer: pq.ParquetWriter | None = None
    wrote = False
    try:
        for part in parts:
            part = Path(part)
            if not part.exists():
                continue
            table = pq.read_table(part)
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            wrote = True
    finally:
        if writer is not None:
            writer.close()
    if not wrote:
        pd.DataFrame().to_parquet(out_path, index=False, engine="fastparquet")


def discover_parquet_parts(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def read_partitioned_table(root: str | Path) -> pd.DataFrame:
    parts = discover_parquet_parts(root)
    if not parts:
        raise FileNotFoundError(root)
    return pd.read_parquet(root, engine="fastparquet")


def read_table_or_partitions(path: str | Path, parts_root: str | Path | None = None) -> pd.DataFrame:
    if table_exists(path):
        return read_table(path)
    if parts_root is not None and Path(parts_root).exists():
        return read_partitioned_table(parts_root)
    raise FileNotFoundError(path)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception:
            pass
    pkl = fallback_pickle_path(path)
    if pkl.exists():
        return pd.read_pickle(pkl)
    csv = fallback_csv_path(path)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(path)
