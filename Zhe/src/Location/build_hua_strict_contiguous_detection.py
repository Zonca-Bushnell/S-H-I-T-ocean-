from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["date", "hua_object_id", "depth_index"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="fastparquet")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, engine="fastparquet")
    tmp.replace(path)


def _strict_centers(centers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if centers.empty:
        return centers.copy(), {
            "objects_total": 0,
            "objects_kept": 0,
            "objects_dropped_no_surface": 0,
            "rows_total": 0,
            "rows_kept": 0,
            "passed_rows_total": 0,
            "passed_rows_kept": 0,
            "isolated_pass_rows_removed": 0,
        }

    keep_frames: list[pd.DataFrame] = []
    objects_kept = 0
    objects_dropped_no_surface = 0
    passed_rows_total = int(centers["hua_pass"].astype(bool).sum())
    isolated_removed = 0

    for _, group in centers.groupby("hua_object_id", sort=False):
        group = group.sort_values("depth_index")
        passed = group[group["hua_pass"].astype(bool)].copy()
        passed_indices = set(passed["depth_index"].astype(int).tolist())
        if 0 not in passed_indices:
            objects_dropped_no_surface += 1
            isolated_removed += len(passed)
            continue
        max_contiguous = -1
        while (max_contiguous + 1) in passed_indices:
            max_contiguous += 1
        kept = passed[passed["depth_index"].astype(int).le(max_contiguous)].copy()
        removed = passed[passed["depth_index"].astype(int).gt(max_contiguous)]
        isolated_removed += len(removed)
        if not kept.empty:
            objects_kept += 1
            keep_frames.append(kept)

    if keep_frames:
        strict = pd.concat(keep_frames, ignore_index=True)
        strict = strict.sort_values(["date", "hua_object_id", "depth_index"]).reset_index(drop=True)
    else:
        strict = centers.iloc[0:0].copy()

    return strict, {
        "objects_total": int(centers["hua_object_id"].nunique()),
        "objects_kept": int(objects_kept),
        "objects_dropped_no_surface": int(objects_dropped_no_surface),
        "rows_total": int(len(centers)),
        "rows_kept": int(len(strict)),
        "passed_rows_total": int(passed_rows_total),
        "passed_rows_kept": int(len(strict)),
        "isolated_pass_rows_removed": int(isolated_removed),
    }


def _filter_by_keys(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    if df.empty or keys.empty:
        return df.iloc[0:0].copy()
    cols = [col for col in KEY_COLUMNS if col in df.columns]
    if len(cols) != len(KEY_COLUMNS):
        return df.iloc[0:0].copy()
    return df.merge(keys[KEY_COLUMNS].drop_duplicates(), on=KEY_COLUMNS, how="inner")


def _copy_aux_files(input_dir: Path, output_dir: Path) -> None:
    for name in (
        "method_alignment_zh.md",
        "hua_acc_replication_summary_zh.md",
        "rejection_reasons.csv",
    ):
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _empty_totals() -> dict[str, int]:
    return {
        "days_processed": 0,
        "objects_total": 0,
        "objects_kept": 0,
        "objects_dropped_no_surface": 0,
        "rows_total": 0,
        "rows_kept": 0,
        "passed_rows_total": 0,
        "passed_rows_kept": 0,
        "isolated_pass_rows_removed": 0,
        "structure_rows_kept": 0,
        "circle_rows_kept": 0,
        "voxel_rows_kept": 0,
    }


def _process_day_part(center_path_text: str, input_dir_text: str, output_dir_text: str) -> tuple[dict[str, int], list[dict[str, object]]]:
    input_dir = Path(input_dir_text)
    output_dir = Path(output_dir_text)
    center_path = Path(center_path_text)
    date_key = center_path.stem.split("=", 1)[-1]
    centers = _read(center_path)
    strict, stats = _strict_centers(centers)
    totals = _empty_totals()
    for key, value in stats.items():
        totals[key] += int(value)
    totals["days_processed"] = 1

    keys = strict[KEY_COLUMNS].drop_duplicates() if not strict.empty else pd.DataFrame(columns=KEY_COLUMNS)
    _write(strict, output_dir / "parts" / "centers" / center_path.name)

    circle = _read(input_dir / "parts" / "circle" / center_path.name)
    circle_strict = _filter_by_keys(circle, keys)
    totals["circle_rows_kept"] += int(len(circle_strict))
    _write(circle_strict, output_dir / "parts" / "circle" / center_path.name)

    structures = _read(input_dir / "parts" / "structures" / center_path.name)
    structures_strict = _filter_by_keys(structures, keys)
    totals["structure_rows_kept"] += int(len(structures_strict))
    _write(structures_strict, output_dir / "parts" / "structures" / center_path.name)

    day = pd.to_datetime(date_key).date()
    voxel_path = input_dir / "object_voxels_parts" / f"year={day.year}" / f"date={date_key}.parquet"
    voxels = _read(voxel_path)
    voxels_strict = _filter_by_keys(voxels, keys)
    totals["voxel_rows_kept"] += int(len(voxels_strict))
    _write(voxels_strict, output_dir / "object_voxels_parts" / f"year={day.year}" / f"date={date_key}.parquet")

    examples: list[dict[str, object]] = []
    if not centers.empty:
        for object_id, group in centers.groupby("hua_object_id", sort=False):
            passed = sorted(group.loc[group["hua_pass"].astype(bool), "depth_index"].astype(int).tolist())
            if not passed or 0 not in passed:
                continue
            k = 0
            while k in passed:
                k += 1
            removed = [idx for idx in passed if idx >= k]
            if removed:
                examples.append(
                    {
                        "date": str(group["date"].iloc[0])[:10],
                        "hua_object_id": str(object_id),
                        "passed_depth_indices": passed,
                        "kept_depth_indices": list(range(k)),
                        "removed_depth_indices": removed,
                    }
                )
                if len(examples) >= 3:
                    break
    return totals, examples


def build_strict_detection(input_dir: Path, output_dir: Path, *, force: bool = False, workers: int = 1) -> dict[str, object]:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_aux_files(input_dir, output_dir)

    center_parts = sorted((input_dir / "parts" / "centers").glob("date=*.parquet"))
    if not center_parts:
        raise FileNotFoundError(input_dir / "parts" / "centers")

    totals = _empty_totals()
    examples: list[dict[str, object]] = []

    if workers <= 1:
        iterator = (
            _process_day_part(str(path), str(input_dir), str(output_dir))
            for path in center_parts
        )
        for day_totals, day_examples in iterator:
            for key, value in day_totals.items():
                totals[key] += int(value)
            if len(examples) < 25:
                examples.extend(day_examples[: 25 - len(examples)])
            if totals["days_processed"] % 100 == 0:
                print(
                    f"[strict] days={totals['days_processed']} kept_pass={totals['passed_rows_kept']} "
                    f"removed_isolated={totals['isolated_pass_rows_removed']}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_process_day_part, str(path), str(input_dir), str(output_dir))
                for path in center_parts
            ]
            for future in as_completed(futures):
                day_totals, day_examples = future.result()
                for key, value in day_totals.items():
                    totals[key] += int(value)
                if len(examples) < 25:
                    examples.extend(day_examples[: 25 - len(examples)])
                if totals["days_processed"] % 100 == 0:
                    print(
                        f"[strict] days={totals['days_processed']} kept_pass={totals['passed_rows_kept']} "
                        f"removed_isolated={totals['isolated_pass_rows_removed']}",
                        flush=True,
                    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "policy": "Keep only surface-connected contiguous Hua-passed layers depth_index=0..k.",
        "workers": int(workers),
        "totals": totals,
        "examples_removed_isolated_pass_layers": examples,
    }
    (output_dir / "strict_contiguous_detection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Strict-contiguous Hua detection manifest",
        "",
        f"- Input: `{input_dir}`",
        f"- Output: `{output_dir}`",
        "- Policy: keep only `depth_index=0..k` contiguous Hua-passed layers for each `date + hua_object_id`.",
        "",
        "## Totals",
        "",
    ]
    for key, value in totals.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Example removed isolated layers", ""])
    for item in examples[:10]:
        lines.append(
            f"- `{item['date']} {item['hua_object_id']}` removed `{item['removed_depth_indices']}` "
            f"from passed `{item['passed_depth_indices']}`"
        )
    (output_dir / "strict_contiguous_detection_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strict-contiguous Hua detection directory from existing per-day detection parts.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    build_strict_detection(Path(args.input_dir), Path(args.output_dir), force=bool(args.force), workers=max(1, int(args.workers)))


if __name__ == "__main__":
    main()
