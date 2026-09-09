# META4 Core Argo W 性能审计

本文档记录 3D thermal-wind W 管线在启动全球 `60S-60N / 10deg` 全量前的性能审计结论。

## 审计口径

- Worktree：main worktree `S-H-I-T-ocean-`
- Branch：`Dipole`
- 测试入口：`meta4_core_argo_vertical_transport.py`
- 测试模式：`20N crossing, 1R, match-mode all, thermal_wind_depth_stack`
- 推荐参数：`grid_n=61, Rc=1.0R, min_obs=8, smooth=4`
- 计算设备：`--compute-device auto`，MATLAB 检测到 GPU Cressman 可用。

## 阶段耗时观察

一次 profiler smoke 使用 `depth-levels=100:100:1000`、`max-matches-per-group=3000`。在原逐层映射口径下，主要耗时为：

| 阶段 | 典型耗时 | 判断 |
| --- | ---: | --- |
| Argo/META 初始载入、历史速度匹配 | 固定成本，约十几秒量级 | 全量时应在单 MATLAB 进程内复用 |
| profile BOA/QC cache 首次构建 | cyclonic 约 `6.6 s`，anticyclonic 约 `3.1 s` | 全量首次运行的重要成本，缓存命中后接近 0 |
| 逐层 Cressman 映射 10 层 | 约 `2.3-2.7 s` | 随深度层线性放大，是主要可优化项 |
| thermal-wind velocity stack | 约 `0.1 s` | 不是瓶颈 |
| W terms 梯度/合成 | 约 `0.1 s` | 不是瓶颈 |
| matched MAT/grid MAT/NC | 小样本约 `0.0-0.5 s` | 正式全量时 NC 会增加写盘压力 |
| PNG 输出 | cyclonic 约 `7.6 s`，anticyclonic 约 `3.4 s` | 小样本中占比较高，正式全量应谨慎控制图像数量 |

MATLAB profiler 的函数级热点显示：

| 排名 | 函数 | TotalTime | NumCalls | 解释 |
| ---: | --- | ---: | ---: | --- |
| 1 | `build_group_3d` / `composite_grid_3d` | 约 `16 s` | 2 | 核心计算入口 |
| 2 | `write_group_outputs_3d` | 约 `12 s` | 2 | 输出和绘图 |
| 3 | `profile_depth_stack_cache` | 约 `10 s` | 2 | BOA/profile 逐深度 QC |
| 4 | `exportgraphics` | 约 `7.6 s` | 4 | PNG 输出 |
| 5 | `track_cx` | 约 `5.6 s` | 2 | META 轨迹传播速度 |
| 6 | `smooth2_supported` | 约 `4.3 s` | 40 | 逐层平滑 |
| 7 | `match_history_argo1000m` | 约 `3.8 s` | 1 | 历史速度匹配 |
| 8 | `cressman_map_multi` / GPU backend | 约 `1.2 s` | 22 | profiler 中 GPU 异步耗时偏低，但日志显示逐层映射仍显著 |

没有发现一个“很小但被异常频繁调用”的自写函数单独吞掉大部分时间。高频 `mean` 主要来自平滑/绘图/内部统计。

## 已完成的提速改造

### 1. 可选 MATLAB profiler

新增 Python 参数：

```powershell
--matlab-profile
```

默认关闭。开启后输出：

- `matlab_profile_info.mat`
- `matlab_profile_html/`

注意：`profsave` 本身会生成大量 HTML 文件并明显拖慢收尾，因此正式全量不得开启。

### 2. 关键阶段日志

3D 管线现在会记录：

- matching 耗时
- profile depth cache 耗时
- base velocity Cressman 耗时
- depth-stack Cressman 耗时
- thermal-wind velocity stack 耗时
- W terms 耗时
- matched MAT / grid MAT / NetCDF / PNG 写出耗时

### 3. 3D Cressman 批量矩阵化

原口径：每个深度层调用一次 Cressman，200 层约调用 200 次，每次只映射 `z_anom/rho_anom` 两个变量。

新口径：把 `z_anom(depth)` 和 `rho_anom(depth)` 组合为缺测感知的样本矩阵，一次分块映射：

```text
V = [z_anom_1 ... z_anom_nz rho_anom_1 ... rho_anom_nz]
```

GPU backend 使用矩阵乘法批量计算：

```text
mapped = W' * V / W' * valid
```

这样减少了 MATLAB 循环、GPU kernel 调度和 CPU-GPU 往返。小样本 10 层对比中，depth-stack Cressman 从约 `2.3-2.7 s` 降到约 `0.5 s`。

## 全量前建议

1. 正式全量不打开 `--matlab-profile`。
2. 第一轮全球 3D 建议使用 MAT-only，即加 `--no-grid-nc`；NetCDF 可在后处理阶段从 MAT 批量转换。理由是 MAT 保存完整 MATLAB 结构更快，且目前科学检查主要靠 PNG 和 MAT。
3. 不输出 CSV/JSON，继续使用默认 MAT 表和 MAT grid。
4. 保留 PNG，但只输出必要的 `section` 和 `depth_slices`；若后续全量仍慢，再增加 `--no-plots` 或 `--plot-summary-only`。
5. 若全量首次运行瓶颈转为 profile BOA/QC cache，应进一步把 profile cache 从“每条 crossing 纬线独立”升级为“全局 Argo profile × depth × month/lon/lat 一次缓存”，避免同一个 Argo 在相邻纬线重复做反插值。
6. `track_cx` 可预计算并缓存到 META 派生轻量 MAT，避免每次按极性重新扫全 track。
7. 当前没有改 `x/R,y/R` 定义、`meshgrid` 顺序或 `gradient(dx,dy)` 顺序；本轮只改变量维度批处理，避免再次出现东西/南北偶极方向被转置的问题。

## 推荐全量命令

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --crossing-lats -60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60 `
  --intersect-radius-r 1 `
  --bbox 0,360,-60,60 `
  --match-mode all `
  --vertical-mode thermal_wind_depth_stack `
  --depth-levels 10:10:2000 `
  --grid-n 61 `
  --cressman-radius-r 1.0 `
  --cressman-min-obs 8 `
  --smooth-passes 4 `
  --compute-device auto `
  --workers 8 `
  --no-grid-nc `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_W_3D_crossing_global_60S60N_10deg_repeat_all_recommended_thermalwind"
```
