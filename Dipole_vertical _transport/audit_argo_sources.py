from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


DEFAULT_ARGO_ROOT = Path(r"F:\Argo_data")
DEFAULT_BBOX = "120,145,20,35"
DEFAULT_OUTPUT = Path(__file__).with_name("ARGO_SOURCE_AUDIT_ZH.md")


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise argparse.ArgumentTypeError("--bbox ranges must be increasing")
    return lon_min, lon_max, lat_min, lat_max


def matlab_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "''")


def run_matlab_audit(argo_root: Path, bbox: tuple[float, float, float, float]) -> dict:
    with tempfile.TemporaryDirectory(prefix="argo_audit_") as tmp:
        tmp_dir = Path(tmp)
        json_path = tmp_dir / "argo_audit.json"
        script_path = tmp_dir / "audit_argo_sources_matlab.m"
        script_path.write_text(_matlab_script(argo_root, bbox, json_path), encoding="utf-8")
        cmd = ["matlab", "-batch", f"run('{matlab_quote(script_path)}')"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "MATLAB audit failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _matlab_script(
    argo_root: Path,
    bbox: tuple[float, float, float, float],
    json_path: Path,
) -> str:
    lon_min, lon_max, lat_min, lat_max = bbox
    root = matlab_quote(argo_root)
    out = matlab_quote(json_path)
    return f"""
argo_root = '{root}';
bbox = [{lon_min:.12g} {lon_max:.12g} {lat_min:.12g} {lat_max:.12g}];
result = struct();
result.generated_at = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
result.argo_root = argo_root;
result.bbox = bbox;
result.sources = struct();

teos_path = fullfile(argo_root, 'ArgoData_SA_CT_PT_PDen_sigma.mat');
legacy_path = fullfile(argo_root, 'Argo1000m_UVW_TSDen_199601_202306.mat');
ehf_path = fullfile(argo_root, 'Argo1000m_EHF_199601_202306.mat');

result.sources.teos = inspect_file(teos_path);
if exist(teos_path, 'file') == 2
    load(teos_path, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', 'Depth');
    result.sources.teos.stats = profile_stats(I_Time, I_Lon, I_Lat, I_ParkDepth, I_PF, bbox);
    if exist('Depth', 'var')
        result.sources.teos.depth_levels = numel(Depth);
        result.sources.teos.depth_min_m = min(Depth);
        result.sources.teos.depth_max_m = max(Depth);
    end
    clear I_Time I_Lon I_Lat I_ParkDepth I_PF Depth
end

result.sources.legacy_uvw = inspect_file(legacy_path);
if exist(legacy_path, 'file') == 2
    load(legacy_path, 'I_Time', 'I_Lon', 'I_Lat', 'I_ParkDepth', 'I_PF', 'I_Upk', 'I_Vpk', 'I_Wpk', 'Depth1');
    result.sources.legacy_uvw.stats = profile_stats(I_Time, I_Lon, I_Lat, I_ParkDepth, I_PF, bbox);
    in_bbox = bbox_mask(I_Lon, I_Lat, bbox);
    result.sources.legacy_uvw.valid_u_in_bbox = sum(in_bbox & ~isnan(I_Upk));
    result.sources.legacy_uvw.valid_v_in_bbox = sum(in_bbox & ~isnan(I_Vpk));
    result.sources.legacy_uvw.valid_w_in_bbox = sum(in_bbox & ~isnan(I_Wpk));
    result.sources.legacy_uvw.valid_w_1000m_in_bbox = sum(in_bbox & I_ParkDepth >= 900 & I_ParkDepth <= 1100 & ~isnan(I_Wpk));
    result.sources.legacy_uvw.median_w_1000m_m_s = median(I_Wpk(in_bbox & I_ParkDepth >= 900 & I_ParkDepth <= 1100), 'omitnan');
    if exist('Depth1', 'var')
        result.sources.legacy_uvw.depth_levels = numel(Depth1);
        result.sources.legacy_uvw.depth_min_m = min(Depth1);
        result.sources.legacy_uvw.depth_max_m = max(Depth1);
    end
    clear I_Time I_Lon I_Lat I_ParkDepth I_PF I_Upk I_Vpk I_Wpk Depth1 in_bbox
end

result.sources.ehf = inspect_file(ehf_path);
result.sources.boa = inspect_directory(fullfile(argo_root, 'BOA_Argo'));
result.sources.isas = inspect_directory(fullfile(argo_root, 'ISAS_Argo'));
result.sources.easyoneargo = inspect_directory(fullfile(argo_root, 'EasyOneArgo'));
result.sources.gdac_like = inspect_directory(fullfile(argo_root, 'ARGO'));
result.sources.argos = inspect_directory(fullfile(argo_root, 'Argos'));
result.sources.self_boa_pden = inspect_directory(fullfile(argo_root, 'Self_BOA_Argo_PotentialDensity'));

text = jsonencode(result);
fid = fopen('{out}', 'w');
fwrite(fid, text, 'char');
fclose(fid);

function out = inspect_file(path)
    out = struct();
    out.path = path;
    out.exists = exist(path, 'file') == 2;
    if out.exists
        d = dir(path);
        out.size_bytes = d.bytes;
        info = whos('-file', path);
        vars = repmat(struct('name', '', 'size', [], 'class', '', 'bytes', 0), numel(info), 1);
        for k = 1:numel(info)
            vars(k).name = info(k).name;
            vars(k).size = info(k).size;
            vars(k).class = info(k).class;
            vars(k).bytes = info(k).bytes;
        end
        out.variables = vars;
    else
        out.size_bytes = 0;
        out.variables = [];
    end
end

function out = inspect_directory(path)
    out = struct();
    out.path = path;
    out.exists = exist(path, 'dir') == 7;
    out.file_count = 0;
    out.mat_file_count = 0;
    out.nc_file_count = 0;
    out.total_size_bytes = 0;
    if out.exists
        files = dir(fullfile(path, '**', '*'));
        is_file = ~[files.isdir];
        files = files(is_file);
        out.file_count = numel(files);
        for k = 1:numel(files)
            out.total_size_bytes = out.total_size_bytes + files(k).bytes;
            [~,~,ext] = fileparts(files(k).name);
            if strcmpi(ext, '.mat')
                out.mat_file_count = out.mat_file_count + 1;
            end
            if strcmpi(ext, '.nc')
                out.nc_file_count = out.nc_file_count + 1;
            end
        end
    end
end

function in_bbox = bbox_mask(lon, lat, bbox)
    lon = double(lon);
    lat = double(lat);
    lon(lon < 0) = lon(lon < 0) + 360;
    in_bbox = lon >= bbox(1) & lon <= bbox(2) & lat >= bbox(3) & lat <= bbox(4);
end

function out = profile_stats(time, lon, lat, park_depth, platform, bbox)
    in_bbox = bbox_mask(lon, lat, bbox);
    paired = false(size(time));
    dt_days = nan(size(time));
    for i = 2:numel(time)-1
        if platform(i-1) == platform(i+1)
            paired(i) = true;
            dt_days(i) = time(i+1) - time(i-1);
        end
    end
    out = struct();
    out.total_profiles = numel(time);
    out.in_bbox_profiles = sum(in_bbox);
    out.valid_parking_depth_in_bbox = sum(in_bbox & ~isnan(park_depth));
    out.park_900_1100_in_bbox = sum(in_bbox & park_depth >= 900 & park_depth <= 1100);
    out.park_1400_1600_in_bbox = sum(in_bbox & park_depth >= 1400 & park_depth <= 1600);
    out.park_ge_1800_in_bbox = sum(in_bbox & park_depth >= 1800);
    out.paired_center_profiles_in_bbox = sum(in_bbox & paired);
    out.median_pair_dt_days = median(dt_days(in_bbox & paired), 'omitnan');
    out.unique_platforms_in_bbox = numel(unique(platform(in_bbox)));
end
"""


def format_int(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "-"


def size_gb(value: object) -> str:
    try:
        return f"{float(value) / 1024**3:.2f} GB"
    except (TypeError, ValueError):
        return "-"


def var_names(source: dict) -> set[str]:
    return {item.get("name", "") for item in source.get("variables", [])}


def stats(source: dict) -> dict:
    return source.get("stats", {}) if isinstance(source.get("stats"), dict) else {}


def build_report(audit: dict) -> str:
    generated = audit.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    bbox = audit.get("bbox", [])
    bbox_text = ", ".join(format_float(item, 6) for item in bbox)
    sources = audit.get("sources", {})
    teos = sources.get("teos", {})
    legacy = sources.get("legacy_uvw", {})
    teos_stats = stats(teos)
    legacy_stats = stats(legacy)
    teos_vars = var_names(teos)
    legacy_vars = var_names(legacy)

    lines = [
        "# Argo 垂直速度数据源审计",
        "",
        f"- 生成时间：`{generated}`",
        f"- Argo 数据根目录：`{audit.get('argo_root', '')}`",
        f"- 审计区域：`{bbox_text}`，即黑潮区域 `120E-145E, 20N-35N`",
        "- 第一阶段目标：判断哪类 Argo 数据适合后续估算 `w = c * dz_rho/dx + u · grad(z_rho)`，不做全量垂直速度计算。",
        "",
        "## 结论",
        "",
        "- 主数据源推荐：`ArgoData_SA_CT_PT_PDen_sigma.mat`。它保留逐 profile 序列、float 编号、时间、经纬度、parking depth，并包含 TEOS-10 派生密度变量，最适合后续反插值得到等密面深度 `z_rho`。",
        "- 历史验证源推荐：`Argo1000m_UVW_TSDen_199601_202306.mat`。它已经包含 `I_Upk/I_Vpk/I_Wpk`，可用来复现和检查旧的 parking-drift / isopycnal-displacement 方法，但密度口径应作为旧口径对照。",
        "- 背景辅助源推荐：`BOA_Argo` 和 `ISAS_Argo`。它们是网格化温盐/密度背景产品，适合辅助估计背景场或气候态梯度，不适合作为 parking drift 主数据源。",
        "",
        "## 黑潮区域核心统计",
        "",
        "| 数据源 | 总 profile | 区域内 profile | 有效 parking depth | 同 float 相邻配对 | 配对中位间隔 | 900-1100 m | 1400-1600 m | >=1800 m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| TEOS 派生逐 profile | {format_int(teos_stats.get('total_profiles'))} | {format_int(teos_stats.get('in_bbox_profiles'))} | {format_int(teos_stats.get('valid_parking_depth_in_bbox'))} | {format_int(teos_stats.get('paired_center_profiles_in_bbox'))} | {format_float(teos_stats.get('median_pair_dt_days'))} 天 | {format_int(teos_stats.get('park_900_1100_in_bbox'))} | {format_int(teos_stats.get('park_1400_1600_in_bbox'))} | {format_int(teos_stats.get('park_ge_1800_in_bbox'))} |",
        f"| 旧 UVW/T/S/Density | {format_int(legacy_stats.get('total_profiles'))} | {format_int(legacy_stats.get('in_bbox_profiles'))} | {format_int(legacy_stats.get('valid_parking_depth_in_bbox'))} | {format_int(legacy_stats.get('paired_center_profiles_in_bbox'))} | {format_float(legacy_stats.get('median_pair_dt_days'))} 天 | {format_int(legacy_stats.get('park_900_1100_in_bbox'))} | {format_int(legacy_stats.get('park_1400_1600_in_bbox'))} | {format_int(legacy_stats.get('park_ge_1800_in_bbox'))} |",
        "",
        "## 候选数据源判断",
        "",
        "### `ArgoData_SA_CT_PT_PDen_sigma.mat`",
        "",
        f"- 文件大小：`{size_gb(teos.get('size_bytes'))}`",
        f"- 深度层数：`{format_int(teos.get('depth_levels'))}`，范围 `{format_float(teos.get('depth_min_m'))}` 到 `{format_float(teos.get('depth_max_m'))}` m。",
        f"- 关键变量存在性：`I_Time`={str('I_Time' in teos_vars).lower()}，`I_Lon/I_Lat`={str({'I_Lon', 'I_Lat'} <= teos_vars).lower()}，`I_ParkDepth`={str('I_ParkDepth' in teos_vars).lower()}，`I_PF`={str('I_PF' in teos_vars).lower()}，`I_sigma1/I_PDen1`={str({'I_sigma1', 'I_PDen1'} <= teos_vars).lower()}。",
        "- 适用性：适合作为正式主源。后续可按 float 相邻 cycle 估计 parking drift，并用 TEOS-10 密度剖面反插值得到 `z_rho`。",
        "",
        "### `Argo1000m_UVW_TSDen_199601_202306.mat`",
        "",
        f"- 文件大小：`{size_gb(legacy.get('size_bytes'))}`",
        f"- 已有区域内有效速度：`U={format_int(legacy.get('valid_u_in_bbox'))}`，`V={format_int(legacy.get('valid_v_in_bbox'))}`，`W={format_int(legacy.get('valid_w_in_bbox'))}`。",
        f"- 1000 m parking 区域内有效 `Wpk`：`{format_int(legacy.get('valid_w_1000m_in_bbox'))}`，中位数 `{format_float(legacy.get('median_w_1000m_m_s'))}` m/s。",
        "- 适用性：适合做历史方法复现和数量级检查，不建议作为新模块正式密度口径。",
        "",
        "### 网格化或辅助产品",
        "",
        "| 数据源 | 文件数 | MAT 文件 | NetCDF 文件 | 总大小 | 判断 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, label, judgement in [
        ("boa", "BOA_Argo", "背景温盐/密度场辅助；不含逐 float parking drift。"),
        ("isas", "ISAS_Argo", "背景温盐场辅助；不含逐 float parking drift。"),
        ("easyoneargo", "EasyOneArgo", "网格产品候选；先不作为主源。"),
        ("gdac_like", "ARGO", "可能含历史 GDAC/逐 profile 材料，可作为追溯来源。"),
        ("argos", "Argos", "非本任务主源，暂不用于等密面垂直速度。"),
        ("self_boa_pden", "Self_BOA_Argo_PotentialDensity", "月度位密网格辅助；不含 float drift。"),
    ]:
        item = sources.get(key, {})
        lines.append(
            f"| `{label}` | {format_int(item.get('file_count'))} | {format_int(item.get('mat_file_count'))} | "
            f"{format_int(item.get('nc_file_count'))} | {size_gb(item.get('total_size_bytes'))} | {judgement} |"
        )

    lines.extend(
        [
            "",
            "## 后续实现建议",
            "",
            "- 用 `ArgoData_SA_CT_PT_PDen_sigma.mat` 建立正式 profile reader，读取 `I_Time/I_Lon/I_Lat/I_ParkDepth/I_PF/I_sigma1` 或指定密度变量。",
            "- 对每个中心 profile，用同一 float 的前后 profile 估计 parking drift：`u = dx/dt`，`v = dy/dt`。",
            "- 对指定密度面 `rho`，在每条密度剖面上单调清洗后反插值得到 `z_rho`。",
            "- 第一项使用涡旋传播速度 `c` 和局地东西向 `dz_rho/dx`；第二项使用 Argo parking drift 与 `grad(z_rho)`。",
            "- 旧 `I_Wpk` 结果只用于 sanity check：数量级、符号分布、区域中位数和有效样本数。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Argo data sources for dipole vertical transport.")
    parser.add_argument("--argo-root", type=Path, default=DEFAULT_ARGO_ROOT)
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox(DEFAULT_BBOX))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = run_matlab_audit(args.argo_root, args.bbox)
    report = build_report(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
