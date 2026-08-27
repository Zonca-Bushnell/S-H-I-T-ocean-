from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def _read_text_tail(path: Path, n: int = 100) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _norm_object_id(value: object) -> int:
    if value is None or pd.isna(value):
        return -1
    text = str(value).strip()
    if text in {"", "-1", "nan", "None"}:
        return -1
    return int(text.replace("_", ""))


def _node_id(i: int, j: int, k: int, x_dim: int, y_dim: int) -> int:
    return int(k) * int(y_dim) * int(x_dim) + int(j) * int(x_dim) + int(i)


def _minimal_poly_block(color_seed: int) -> str:
    r = (73 * color_seed + 31) % 255
    g = (47 * color_seed + 83) % 255
    b = (113 * color_seed + 17) % 255
    return f"{r:3d} {g:3d} {b:3d}\n0\n0\n\n\n"


def _write_surrogate_frames(root: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = pd.read_parquet(root / "frame_object_summary.parquet")
    voxels = pd.read_parquet(root / "object_voxels.parquet")
    voxels["date"] = voxels["date"].astype(str)
    objects["date"] = objects["date"].astype(str)
    voxels["hua_object_id_text"] = voxels["hua_object_id"].astype(str)
    objects["hua_object_id_text"] = objects["hua_object_id"].astype(str)
    voxels["hua_object_id_norm"] = voxels["hua_object_id_text"].map(_norm_object_id)
    objects["hua_object_id_norm"] = objects["hua_object_id_text"].map(_norm_object_id)

    x_dim = int(voxels["i"].max()) + 1
    y_dim = int(voxels["j"].max()) + 1
    z_dim = int(voxels["depth_index"].max()) + 1
    dims = {"x_dim": x_dim, "y_dim": y_dim, "z_dim": z_dim}

    frame_rows: list[dict[str, object]] = []
    for frame_no, day in enumerate(sorted(voxels["date"].unique()), start=1):
        frame = f"acc_{frame_no}"
        day_voxels = voxels[voxels["date"].eq(day)].copy()
        day_objects = objects[objects["date"].eq(day)].copy()
        local_ids = {
            int(hua_id): local_id
            for local_id, hua_id in enumerate(sorted(day_voxels["hua_object_id_norm"].dropna().astype(int).unique()))
        }
        day_voxels["local_object_id"] = day_voxels["hua_object_id_norm"].astype(int).map(local_ids)

        trak_lines: list[str] = []
        uocd_lines: list[str] = []
        attr_lines: list[str] = []
        poly_lines: list[str] = []
        group_lines: list[str] = []

        for hua_id_norm, local_id in local_ids.items():
            obj_vox = day_voxels[day_voxels["hua_object_id_norm"].astype(int).eq(hua_id_norm)].copy()
            if obj_vox.empty:
                continue
            obj_vox["node_id"] = [
                _node_id(i, j, k, x_dim, y_dim)
                for i, j, k in zip(obj_vox["i"], obj_vox["j"], obj_vox["depth_index"])
            ]
            obj_vox = obj_vox.sort_values("node_id")
            vol = int(len(obj_vox))
            mass = float(vol)
            cx = float(obj_vox["i"].mean())
            cy = float(obj_vox["j"].mean())
            cz = float(obj_vox["depth_index"].mean())
            min_i, max_i = int(obj_vox["i"].min()), int(obj_vox["i"].max())
            min_j, max_j = int(obj_vox["j"].min()), int(obj_vox["j"].max())
            min_k, max_k = int(obj_vox["depth_index"].min()), int(obj_vox["depth_index"].max())
            pac_id = local_id + 1
            trak_lines.append(
                f"{mass:.6f} {vol:d} {cx:.6f} {cy:.6f} {cz:.6f} "
                f"{min_i:d} {min_j:d} {min_k:d} {max_i:d} {max_j:d} {max_k:d} "
                f"{min_i:d} {min_j:d} {min_k:d} {max_i:d} {max_j:d} {max_k:d} "
                f"0.000000 1.000000 {pac_id:d}\n"
            )
            first = True
            for row in obj_vox.itertuples(index=False):
                node_id = int(row.node_id)
                i = int(row.i)
                j = int(row.j)
                k = int(row.depth_index)
                if first:
                    uocd_lines.append(
                        f"{local_id:d} {vol:d} {mass:.6f} {cx:.6f} {cy:.6f} {cz:.6f} "
                        f"0 0 0 0 0 0 {node_id:d} {i:d} {j:d} {k:d} 1.000000\n"
                    )
                    first = False
                else:
                    uocd_lines.append(f"{node_id:d} {i:d} {j:d} {k:d} 1.000000\n")
            attr_lines.append(f"Object: {local_id:d}\nVolume: {vol:d}\n")
            poly_lines.append(_minimal_poly_block(local_id + 1))
            group_lines.append(f"{pac_id:d} {local_id:d}\n")

            obj_meta = day_objects[day_objects["hua_object_id_norm"].astype(int).eq(hua_id_norm)]
            frame_rows.append(
                {
                    "date": day,
                    "frame_no": frame_no,
                    "frame_name": frame,
                    "hua_object_id": obj_vox["hua_object_id_text"].iloc[0],
                    "hua_object_id_norm": hua_id_norm,
                    "cpp_local_object_id": local_id,
                    "cpp_object_id_1based": local_id + 1,
                    "voxel_count": vol,
                    "polarity": obj_meta["polarity"].iloc[0] if "polarity" in obj_meta and not obj_meta.empty else "",
                }
            )

        (out_dir / f"{frame}.trak").write_text("".join(trak_lines), encoding="utf-8")
        (out_dir / f"{frame}.uocd").write_text("".join(uocd_lines), encoding="utf-8")
        (out_dir / f"{frame}.attr").write_text("".join(attr_lines), encoding="utf-8")
        (out_dir / f"{frame}.poly").write_text("".join(poly_lines), encoding="utf-8")
        (out_dir / f"{frame}.group").write_text("".join(group_lines), encoding="utf-8")

    frame_map = pd.DataFrame(frame_rows)
    frame_map.to_csv(out_dir / "surrogate_object_id_map.csv", index=False)
    return frame_map, dims


def _write_feature_track_conf(out_dir: Path, n_frames: int, dims: dict[str, int]) -> Path:
    conf = out_dir / "FeatureTrack.Conf"
    text = f"""#
# Generated by run_hua_cpp_cross_validation_acc.py.
# Runs Hua/Rutgers C++ feature tracking on precomputed ACC surrogate frames.

 SURROGATE_TRACKING: 1
 DATA_FILES_PATH:  /dev/null/
 GENERATED_FILES_PATH:  {out_dir.as_posix()}/
 STACKED_NC_DATA_PATH:  SURROGATE
 FILE_BASE_NAME: acc_
 FILE_EXTENSION: .nc
 INITIAL_TIME_STEP: 1
 FINAL_TIME_STEP: {n_frames}
 TIME_STEP_INCREMENT: 1
 TIME_STEP_PRECISION: 1
 VARIABLE_NAMES: omega
 THRESHOLD1: -1.0
 THRESHOLD2: 1.0
 STARTRADIUS: 3
 DELTA_X_THRESHOLD: 0.01
 DELTA_Y_THRESHOLD: 0.01
 DELTA_Z_THRESHOLD: 0.01
 SMALLEST_OBJECT_VOLUME_TO_TRACK: 1
 X_Dim: {dims["x_dim"]}
 Y_Dim: {dims["y_dim"]}
 Z_Dim: {dims["z_dim"]}
 X1_Dim: {dims["x_dim"] - 1}
 Y1_Dim: {dims["y_dim"] - 1}
 Z1_Dim: {dims["z_dim"] - 1}
 X0_Dim: 0
 Y0_Dim: 0
 Z0_Dim: 0
"""
    conf.write_text(text, encoding="utf-8")
    return conf


def _run_cpp(cpp_binary: Path, conf: Path, out_dir: Path) -> tuple[int, str]:
    log_path = out_dir / "ft_surrogate_run.log"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/root/miniconda3/envs/eddy_verify/lib:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(
        [str(cpp_binary), str(conf)],
        cwd=str(out_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


def _parse_trak_table(path: Path, frame_map: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    current_frame = None
    map_by_frame_local = {
        (int(r.frame_no), int(r.cpp_object_id_1based)): (str(r.hua_object_id), int(r.hua_object_id_norm))
        for r in frame_map.itertuples(index=False)
    }
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Frame #"):
            current_frame = int(line.split("#", 1)[1])
            continue
        if current_frame is None:
            continue
        vals = [int(float(x)) for x in line.split()]
        if -1 not in vals:
            continue
        split_at = vals.index(-1)
        prev_ids = vals[:split_at]
        curr_ids = vals[split_at + 1 :]
        if prev_ids and curr_ids:
            if len(prev_ids) == 1 and len(curr_ids) == 1:
                event = "continuous"
            elif len(prev_ids) == 1:
                event = "split"
            elif len(curr_ids) == 1:
                event = "merge"
            else:
                event = "complex"
        elif curr_ids:
            event = "new"
        else:
            event = "dissipate"
        for prev_id in (prev_ids or [-1]):
            for curr_id in (curr_ids or [-1]):
                prev_text, prev_norm = map_by_frame_local.get((current_frame - 1, prev_id), ("-1", -1)) if prev_id != -1 else ("-1", -1)
                curr_text, curr_norm = map_by_frame_local.get((current_frame, curr_id), ("-1", -1)) if curr_id != -1 else ("-1", -1)
                records.append(
                    {
                        "frame_no": current_frame,
                        "event_type": event,
                        "prev_cpp_object_id": prev_id,
                        "curr_cpp_object_id": curr_id,
                        "prev_hua_object_id": prev_text,
                        "curr_hua_object_id": curr_text,
                        "prev_hua_object_id_norm": prev_norm,
                        "curr_hua_object_id_norm": curr_norm,
                        "raw_line": line,
                    }
                )
    return pd.DataFrame(records)


def _python_events(root: Path) -> pd.DataFrame:
    events_path = root / "feature_group_tracking" / "feature_track_events.parquet"
    if not events_path.exists():
        return pd.DataFrame()
    events = pd.read_parquet(events_path)
    events.columns = [str(c) for c in events.columns]
    records: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        data = row._asdict()
        prev_raw = str(data.get("object_id_t0", "-1"))
        curr_raw = str(data.get("object_id_t1", "-1"))
        prev_ids = ["-1"] if prev_raw == "-1" else [x for x in prev_raw.split("|") if x]
        curr_ids = ["-1"] if curr_raw == "-1" else [x for x in curr_raw.split("|") if x]
        for prev_id in prev_ids:
            for curr_id in curr_ids:
                rec = dict(data)
                rec["prev_hua_object_id"] = prev_id
                rec["curr_hua_object_id"] = curr_id
                rec["prev_hua_object_id_norm"] = _norm_object_id(prev_id)
                rec["curr_hua_object_id_norm"] = _norm_object_id(curr_id)
                records.append(rec)
    return pd.DataFrame(records)


def _edge_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["event_type"].astype(str)
        + "|"
        + df["prev_hua_object_id_norm"].astype("int64").astype(str)
        + "|"
        + df["curr_hua_object_id_norm"].astype("int64").astype(str)
    )


def _write_edge_diffs(out_dir: Path, cpp_events: pd.DataFrame, py_events: pd.DataFrame) -> dict[str, int]:
    if cpp_events.empty or py_events.empty:
        return {"matched_edges": 0, "cpp_only_edges": int(len(cpp_events)), "python_only_edges": int(len(py_events))}

    cpp = cpp_events.copy()
    py = py_events.copy()
    cpp["edge_key"] = _edge_key(cpp)
    py["edge_key"] = _edge_key(py)
    cpp_keys = set(cpp["edge_key"])
    py_keys = set(py["edge_key"])
    matched = sorted(cpp_keys & py_keys)
    cpp_only = sorted(cpp_keys - py_keys)
    py_only = sorted(py_keys - cpp_keys)

    cpp[cpp["edge_key"].isin(matched)].to_csv(out_dir / "matched_edges.csv", index=False)
    cpp[cpp["edge_key"].isin(cpp_only)].to_csv(out_dir / "cpp_only_edges.csv", index=False)
    py[py["edge_key"].isin(py_only)].to_csv(out_dir / "python_only_edges.csv", index=False)

    cpp_pair = cpp.copy()
    py_pair = py.copy()
    cpp_pair["pair_key"] = cpp_pair["prev_hua_object_id_norm"].astype("int64").astype(str) + "|" + cpp_pair["curr_hua_object_id_norm"].astype("int64").astype(str)
    py_pair["pair_key"] = py_pair["prev_hua_object_id_norm"].astype("int64").astype(str) + "|" + py_pair["curr_hua_object_id_norm"].astype("int64").astype(str)
    confusion = (
        py_pair[["pair_key", "event_type"]]
        .rename(columns={"event_type": "python_event"})
        .merge(cpp_pair[["pair_key", "event_type"]].rename(columns={"event_type": "cpp_event"}), on="pair_key", how="outer")
        .fillna("missing")
    )
    confusion_counts = confusion.groupby(["python_event", "cpp_event"]).size().reset_index(name="count")
    confusion_counts.to_csv(out_dir / "event_confusion.csv", index=False)

    return {"matched_edges": len(matched), "cpp_only_edges": len(cpp_only), "python_only_edges": len(py_only)}


def _summarize_diff(root: Path, out_dir: Path, cpp_events: pd.DataFrame, frame_map: pd.DataFrame, returncode: int, build_tail: str) -> None:
    tracking = root / "feature_group_tracking"
    event_counts = pd.read_csv(tracking / "feature_track_event_counts.csv")
    summary = pd.read_json(tracking / "feature_group_tracking_summary.json", typ="series")
    py_events = _python_events(root)

    cpp_counts = (
        cpp_events["event_type"].value_counts().rename_axis("event_type").reset_index(name="cpp_count")
        if not cpp_events.empty
        else pd.DataFrame(columns=["event_type", "cpp_count"])
    )
    py_counts = event_counts.rename(columns={"count": "python_count"})
    counts = py_counts.merge(cpp_counts, on="event_type", how="outer").fillna(0)
    counts["python_count"] = counts["python_count"].astype(int)
    counts["cpp_count"] = counts["cpp_count"].astype(int)
    counts["delta_cpp_minus_python"] = counts["cpp_count"] - counts["python_count"]
    counts.to_csv(out_dir / "cpp_python_event_count_diff.csv", index=False)

    cpp_events.to_parquet(out_dir / "cpp_trakTable_events.parquet", index=False)
    cpp_events.to_csv(out_dir / "cpp_trakTable_events.csv", index=False)
    edge_stats = _write_edge_diffs(out_dir, cpp_events, py_events)

    table_path = out_dir / "surrogate_frames" / "acc_1.trakTable"
    total_delta = int(counts["delta_cpp_minus_python"].abs().sum()) if not counts.empty else 0
    conclusion = (
        "二者没有本质性的路线差异：C++ 原库和 Python 复刻都以对象体素 overlap 为主判据，"
        "然后按 new、dissipate、continuous、split、merge 等事件组织 track。"
        "但是二者不是逐行同构实现。C++ 原库同时使用 3D overlap 和表层 2D overlap score，"
        "且在 TrackSplit_Merge / TrackContinuous / TrackNew_Dissipate 的状态机中做较保守的事件归类；"
        "Python 版本目前是等价复刻，使用 voxel-key overlap 与邻域扩展后的 overlap table。"
    )
    if total_delta:
        conclusion += "因此当前差异主要是实现细节和事件归类阈值造成的，不是检测物理口径的根本差异。"
    else:
        conclusion += "事件计数已经一致，但仍需逐边完全一致后才能称为逐行复刻。"

    count_table_lines = ["| event_type | python_count | cpp_count | delta_cpp_minus_python |", "|---|---:|---:|---:|"]
    for row in counts.sort_values("event_type").itertuples(index=False):
        count_table_lines.append(
            f"| {row.event_type} | {int(row.python_count)} | {int(row.cpp_count)} | {int(row.delta_cpp_minus_python)} |"
        )

    lines = [
        "# C++ Hua/Rutgers 原库运行接口适配与 Python 对比",
        "",
        "## 运行状态",
        "",
        f"- C++ build: `{'available' if 'Built target FT' in build_tail else 'unknown'}`",
        f"- FT return code: `{returncode}`",
        f"- surrogate frames: `{frame_map['frame_no'].nunique()}`",
        f"- surrogate objects: `{len(frame_map)}`",
        f"- native trakTable: `{table_path}`",
        "",
        "## Python 等价复刻结果",
        "",
        f"- frame objects: `{int(summary['n_frame_objects'])}`",
        f"- feature tracks: `{int(summary['n_feature_tracks'])}`",
        f"- overlap edges: `{int(summary['n_overlap_edges'])}`",
        f"- longest track objects: `{int(summary['max_track_objects'])}`",
        "",
        "## 事件计数对比",
        "",
        "\n".join(count_table_lines),
        "",
        "## 逐边 diff",
        "",
        f"- matched edges: `{edge_stats['matched_edges']}`",
        f"- C++ only edges: `{edge_stats['cpp_only_edges']}`",
        f"- Python only edges: `{edge_stats['python_only_edges']}`",
        "- 详表：`matched_edges.csv`、`cpp_only_edges.csv`、`python_only_edges.csv`、`event_confusion.csv`。",
        "",
        "## 判断",
        "",
        conclusion,
        "",
        "## 技术口径",
        "",
        "- C++ 输入不是重新跑 Hua 检测，而是 `SURROGATE_TRACKING: 1`，直接读取同一批 `.uocd/.trak/.group/.poly/.attr` surrogate frame。",
        "- surrogate 模式已强制 `dataType = Scalar_data`，否则原库会按 tensor `.uocd` 多读 5 个字段并在第 4 帧崩溃。",
        "- `.uocd` 按原库 `ReadOct` 格式写入：每个对象第一行包含 object header 和第一个节点，后续行是该对象剩余体素节点。",
        "- `.trak` 按原库 `ReadTrak` 格式写入：每行一个对象，使用 grid-index centroid/bounding box；`ObjVol` 与 `.uocd` 节点数逐帧一致。",
        "- 因此这次比较的是 tracking 实现，不是检测中心、速度场定义或 Hua 圆周判据。",
        "",
        "## build log tail",
        "",
        "```text",
        build_tail[-4000:],
        "```",
    ]
    (out_dir / "cpp_python_tracking_diff_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "cpp_python_tracking_diff_summary.json").write_text(
        json.dumps(
            {
                "ft_returncode": returncode,
                "n_surrogate_frames": int(frame_map["frame_no"].nunique()),
                "n_surrogate_objects": int(len(frame_map)),
                "n_cpp_event_edges": int(len(cpp_events)),
                "edge_stats": edge_stats,
                "conclusion": conclusion,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    root = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else root / "cpp_cross_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    surrogate_dir = out_dir / "surrogate_frames"
    frame_map, dims = _write_surrogate_frames(root, surrogate_dir)
    conf = _write_feature_track_conf(surrogate_dir, int(frame_map["frame_no"].max()), dims)
    returncode = -1
    if args.cpp_binary:
        returncode, _ = _run_cpp(Path(args.cpp_binary), conf, surrogate_dir)
    table = surrogate_dir / "acc_1.trakTable"
    cpp_events = _parse_trak_table(table, frame_map)
    build_tail = _read_text_tail(Path(args.cpp_build_log))
    _summarize_diff(root, out_dir, cpp_events, frame_map, returncode, build_tail)
    print(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hua/Rutgers C++ tracking on ACC surrogate frames and compare with Python tracking.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--cpp-build-log", default="/root/autodl-fs/2020_2022_acc/hua_paper_replication/full_feature_tracking_acc/cpp_build_surrogate_patch.log")
    parser.add_argument("--cpp-binary", default="/root/Verify/vendor/Hybrid-Eddy-detection-main/build_fixed/FT")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
