# META4.0 + Core Argo 垂直速度重建方法

本文档记录 `Dipole_vertical _transport/meta4_core_argo_vertical_transport.py`
的当前科学口径和工程假定。

## 数据源

- Argo 主数据源：`F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat`
- 历史速度/观测 W 校准源：`F:\Argo_data\Argo1000m_UVW_TSDen_199601_202306.mat`
- META4.0 涡旋源：`F:\Eddy\Eddy\META4.0_DT_allsat`
- 结果根目录：`E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_crossing_global_60S60N_10deg`

Argo 使用 TEOS-10 派生密度变量，默认读取 `I_sigma1`。历史
`Argo1000m_UVW_TSDen_199601_202306.mat` 不作为正式密度口径，但修正版默认
匹配其中的 `I_Upk/I_Vpk/I_Wpk`：`I_Upk/I_Vpk` 作为 parking drift，`I_Wpk`
作为观测 W sanity check。

## 样本选择

- 区域：全球 Argo 常规有效覆盖带 `0E-360E, 60S-60N`。
- 默认 crossing：`--selection-mode crossing_lat`，默认纬线为
  `60S,50S,40S,30S,20S,10S,00N,10N,20N,30N,40N,50N,60N`。
- 每条纬线选择涡旋本体 `1R` 跨过该纬线的 META 涡旋；此时 Argo 不按纬度带预筛，
  只由 bbox、Core parking depth、时间窗和 `0-4R` 空间匹配决定。
- 若只想跑单条纬线，可使用 `--target-lat 20`；若显式给出多条纬线，
  使用 `--crossing-lats -60,-50,...,60`，其优先级高于 `--target-lat`。
- 纬度带模式保留为对照：`--selection-mode lat_band --lat-bands 20:25`。
- 可选重复匹配：`--match-mode all` 会让同一条 Argo profile 在所有满足时间窗和
  `0-4R` 的涡旋坐标系中重复投影；默认 `--match-mode nearest` 仍只归属最近
  `r/R` 涡旋，避免重复计数。
- Argo 类型：只取 Core Argo，第一版定义为 `I_ParkDepth` 在 `900-1100 m`。
- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `1 day`。
- 空间匹配：以 META 涡心为原点，以 `final_radius` 为 `R`，保留 `0-4R`
  内 profile，并标记 `0-1R`、`1-2R`、`2-4R`。
- 极性：cyclonic 和 anticyclonic 分开计算；不再输出 combined，因为两种极性合并后不保留清晰物理意义。
- 目录标签：crossing 使用 `cross_20N_1R`、`cross_50S_1R`；lat-band 对照使用
  `60S_55S`、`05S_00N`、`00N_05N`、`55N_60N` 这类半球显式标签。

## 速度与垂直速度分解

修正版默认重建：

```text
rebuild_W_up = +c_x_rel * dz'_rho/dx - (u_pk - c_x_raw, v_pk) · grad(z'_rho)
```

其中：

- `c_x_raw`：从 META track 相邻轨迹点中央差分得到的局地东西向传播速度。
- `u_bg`：同 crossing 组、同极性、匹配 Core Argo 的 parking drift 纬向均值。
- `c_x_rel = mean(c_x_raw) - mean(u_bg)`。
- 默认 `--velocity-source argo1000m_match`：使用历史文件的 `I_Upk/I_Vpk`
  作为 parking drift；旧的相邻 profile 位置差近似仅保留为
  `--velocity-source profile_diff` 对照。
- `rho0`：默认取每个 crossing 组/极性内，匹配 Core Argo 在实际 parking depth
  处 `I_sigma1` 的中位数，作为共同目标等密面。
- `z_rho`：每条 profile 上共同 `rho0` 对应的等密面深度。
- 默认推荐 `--z-mode anomaly_boa_climatology`：使用
  `F:\Argo_data\Self_BOA_Argo_PotentialDensity\PDen1000_YYYYMM.mat`
  构建 BOA 多年同月位密气候态，对每条 profile 的 `lon/lat/month`
  双线性插值得到背景密度剖面，并在同一 `rho0` 上反插值得到
  `z_bg`，最终对 `z'_rho = z_rho - z_bg` 求梯度。
- `anomaly_farfield_plane` 保留为对照：使用 `2-4R` 远场样本拟合
  `z_bg = a + b x/R + c y/R`。旧的 `anomaly_farfield` 只扣一个远场中位数，
  已证实会残留强南北背景坡度。
- BOA `Den` 是约 `1032 kg/m^3` 的位密，而默认 `I_sigma1` 是约
  `32 kg/m^3` 的 sigma 口径；脚本会自动做 `+/-1000` 单位对齐后再
  寻找背景 crossing。
- `z_rho` 反插值只允许显式 bracket crossing，不再 fallback 到全剖面
  `interp1(profile, depth)`；CSV 输出 `rho_crossing_count`、
  `rho_bracket_dz_m`、`local_drho_dz`，BOA 背景则输出
  `boa_rho_crossing_count`、`boa_rho_bracket_dz_m`、`boa_local_drho_dz`
  和 `boa_bg_valid`。
- `z_rho` 有效窗口：默认只保留 `900-1100 m`，且要求
  `abs(local_drho_dz) >= 1e-5`、bracket 厚度 `<=150 m`。

注意：旧的逐 profile parking-density 口径会使 `z_rho` 几乎退化为每条
profile 自身的 parking depth，导致 `dz_rho/dx` 和重建 W 接近 0。脚本保留
`--rho0-mode profile` 作为历史对照，但正式 composite 默认使用共同
`rho0`，更符合“同一等密面深度起伏”的物理定义。

## 网格覆盖与空白区

复合图默认先把 profile 投影到涡心归一化坐标，再用 Cressman objective
mapping 生成连续 `z_rho/u/v` composite 场。Cressman 权重为
`w=(R_c^2-r^2)/(R_c^2+r^2)`，只使用映射半径 `R_c` 内样本；脚本默认
`--cressman-radius-r 0.5`、`--cressman-min-obs 3`。脚本同时在
`composite_grid.json/.npz` 中写出原始 bin 计数 `sample_count` 和 Cressman
支撑样本数 `mapped_support`。

`dz_rho/dx` 的数值梯度在连续 composite 场上计算；白色表示当前格点没有达到
最小样本支撑。若需要查看完全不插值的原始散点格点结果，可加
`--grid-mapping bin`；若需要诊断无约束凸包插值，可加
`--grid-mapping scattered`。如果使用 `--max-matches-per-group 200` 做 smoke run，
覆盖和插值支撑都会偏低；正式结果应使用默认 `0` 读取全部匹配样本。

符号约定：W 统一向上为正，与历史 `I_Wpk` 一致；但 `z_rho_m`、`z_rho_bg_m`
和 `z_rho_anom_m` 仍按正深度向下保存和标识。JSON/NPZ 里同时保留
`rebuild_w_raw_depth_positive_m_s`，用于检查深度向下正公式的原始符号。

脚本还输出 `gradient_order_comparison.png`：左图是主口径“先 Cressman 合成
`z'_rho` 后求梯度”，右图是轻量诊断“逐样本局地平面梯度后再合成 W”。
为避免诊断项拖慢全样本，后者默认最多使用 `--sample-gradient-max-profiles 1000`
个确定性抽样 profile，不作为主结果。

## 三维 W 模式

`--vertical-mode isopycnal_depth_stack` 会把单一约 1000 m 等密面 W 扩展为
`W(x/R,y/R,z)`。每个名义深度 `z0` 使用 BOA 多年同月局地背景密度
`rho_bg(lon,lat,month,z0)` 作为目标密度，在 Argo profile 和 BOA 背景剖面上分别
严格 bracket 反插值得到 `z_rho_profile` 与 `z_rho_bg`，再计算
`z'_rho = z_rho_profile - z_rho_bg`。

本模式默认深度层为 `100:100:1900 m`，横截面用 `--section-axis x`，即沿
`x/R=-4..4`、`|y/R|<=0.25` 做中位数截面。纵坐标显示正深度数值，W 仍向上为正。
输出为 `matched_core_argo_3d.csv`、`w_3d_grid.npz/json`、`w_3d_section_x.png` 和
`w_3d_depth_slices.png`。

## 2D 快速参数敏感度

`--fast-sensitivity-2d` 用于快速判断 20N crossing W 图像碎片/锯齿是否来自
Cressman 半径、最小支撑样本、网格分辨率和平滑强度。该模式每个极性只做一次
Argo-META 匹配、`rho0/z_rho` 反插值和 BOA 背景 QC，然后把缓存样本表复用于 6 组
预设参数：`baseline`、`recommended`、`smoother`、`strong_support`、
`low_res_smooth`、`high_smooth`。

加速默认使用 `--compute-device auto`：若 MATLAB 能访问 GPU，则 Cressman 的距离矩阵
和权重求和走 `gpuArray` 分块计算；若 GPU 不可用则自动回到 CPU。CPU 模式下参数组合
可使用 MATLAB 并行池，`--workers` 默认取 CPU 核心数的一半、最多 8。GPU 模式和
参数组合 `parfor` 不同时启用，避免多个 worker 抢同一张 GPU。此模式仍只输出
cyclonic 和 anticyclonic，不输出 combined。

## BOA_Argo 假定

BOA_Argo / Self_BOA_Argo_PotentialDensity 当前只作为 gridded 密度背景产品使用，
不从 BOA 直接推导背景速度；涡旋相对传播速度的背景扣除仍使用匹配 Core Argo
的 parking drift 纬向均值。

## 运行入口

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py"
```

默认入口即全球 `60S-60N`、每 10 度 crossing、全样本。调试或 smoke run 可限制每组
匹配数，并建议输出到单独 smoke 目录：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --crossing-lats -20,0,20 `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_crossing_global_60S60N_10deg_smoke" `
  --max-matches-per-group 200
```

若希望调整 Cressman 客观分析半径或最小样本数，可使用：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --cressman-radius-r 0.5 `
  --cressman-min-obs 3
```

若需要复现原始散点格点图，可使用：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --grid-mapping bin
```

若需要复现旧的逐 profile `rho0` 口径，可使用：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --rho0-mode profile
```

若需要显式跑全量 crossing，可使用：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --selection-mode crossing_lat `
  --crossing-lats -60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60 `
  --intersect-radius-r 1 `
  --bbox 0,360,-60,60 `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_crossing_global_60S60N_10deg"
```

若需要测试一条 Argo 重复参与多个涡旋 composite，可加：

```powershell
--match-mode all
```

重复匹配结果的 `SUMMARY.csv` 会额外记录 `unique_argo_count` 和
`duplicate_match_count`。

20N crossing 修正版验证命令：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --selection-mode crossing_lat `
  --target-lat 20 `
  --intersect-radius-r 1 `
  --match-mode nearest `
  --bbox 0,360,-60,60 `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_fixed_crossing_20N_1R"
```

该目录额外输出 `wpk_validation.png` 和 `velocity_sign_sensitivity.png`，用于
比较正式重建 W、历史观测 `I_Wpk` 和速度/符号敏感性。旧的单独临时
`I_Wpk` 绘图入口已删除；`I_Wpk` 现在只作为正式程序里的观测校验字段保存和出图。

## 参考依据

本地 PDF 依据见 `ARGO_EDDY_W_LITERATURE_CHECK_ZH.md` 和
`D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\PDF\Dipole_vertical _transport`。
未能下载为有效 PDF 的论文只作为待补充条目，不计入本地依据。
