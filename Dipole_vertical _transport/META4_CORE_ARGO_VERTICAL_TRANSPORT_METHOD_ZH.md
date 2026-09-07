# META4.0 + Core Argo 垂直速度重建方法

本文档记录 `Dipole_vertical _transport/meta4_core_argo_vertical_transport.py`
的当前科学口径和工程假定。

## 数据源

- Argo 主数据源：`F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat`
- 历史速度/观测 W 校准源：`F:\Argo_data\Argo1000m_UVW_TSDen_199601_202306.mat`
- META4.0 涡旋源：`F:\Eddy\Eddy\META4.0_DT_allsat`
- 结果根目录：`E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_global_60S60N_5deg`

Argo 使用 TEOS-10 派生密度变量，默认读取 `I_sigma1`。历史
`Argo1000m_UVW_TSDen_199601_202306.mat` 不作为正式密度口径，但修正版默认
匹配其中的 `I_Upk/I_Vpk/I_Wpk`：`I_Upk/I_Vpk` 作为 parking drift，`I_Wpk`
作为观测 W sanity check。

## 样本选择

- 区域：全球 Argo 常规有效覆盖带 `0E-360E, 60S-60N`。
- 纬度带：默认按 5 度分带，从 `60S-55S` 到 `55N-60N`。
- 可选 crossing 模式：`--selection-mode crossing_lat --target-lat 20 --intersect-radius-r 1`
  会选择涡旋本体 `1R` 跨过目标纬线的 META 涡旋；此时 Argo 不按纬度带预筛，
  只由 bbox、Core parking depth、时间窗和 `0-4R` 空间匹配决定。
- 可选重复匹配：`--match-mode all` 会让同一条 Argo profile 在所有满足时间窗和
  `0-4R` 的涡旋坐标系中重复投影；默认 `--match-mode nearest` 仍只归属最近
  `r/R` 涡旋，避免重复计数。
- Argo 类型：只取 Core Argo，第一版定义为 `I_ParkDepth` 在 `900-1100 m`。
- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `1 day`。
- 空间匹配：以 META 涡心为原点，以 `final_radius` 为 `R`，保留 `0-4R`
  内 profile，并标记 `0-1R`、`1-2R`、`2-4R`。
- 极性：cyclonic 和 anticyclonic 分开计算，并输出 combined 汇总。
- 目录标签：纬度带使用 `60S_55S`、`05S_00N`、`00N_05N`、`55N_60N`
  这类半球显式标签，避免南半球被误标为 `N`。

## 速度与垂直速度分解

修正版默认重建：

```text
rebuild_W = -c_x_rel * dz'_rho/dx + (u_pk - c_x_raw, v_pk) · grad(z'_rho)
```

其中：

- `c_x_raw`：从 META track 相邻轨迹点中央差分得到的局地东西向传播速度。
- `u_bg`：同纬度带、同极性、匹配 Core Argo 的 parking drift 纬向均值。
- `c_x_rel = mean(c_x_raw) - mean(u_bg)`。
- 默认 `--velocity-source argo1000m_match`：使用历史文件的 `I_Upk/I_Vpk`
  作为 parking drift；旧的相邻 profile 位置差近似仅保留为
  `--velocity-source profile_diff` 对照。
- `rho0`：默认取每个纬度带/极性内，匹配 Core Argo 在实际 parking depth
  处 `I_sigma1` 的中位数，作为共同目标等密面。
- `z_rho`：每条 profile 上共同 `rho0` 对应的等密面深度。
- 默认 `--z-mode anomaly_farfield`：使用 `2-4R` 样本中位数估计背景，
  对 `z'_rho = z_rho - median(z_rho in 2-4R)` 求梯度。
- `z_rho` 反插值只允许显式 bracket crossing，不再 fallback 到全剖面
  `interp1(profile, depth)`；CSV 输出 `rho_crossing_count`、
  `rho_bracket_dz_m`、`local_drho_dz`。
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

## BOA_Argo 假定

BOA_Argo 可作为 gridded 温盐/密度背景产品的科学依据，适合后续估计背景
密度面或气候态密度结构。第一版不从 BOA_Argo 直接推导背景速度；涡旋相对
传播速度的背景扣除使用匹配 Core Argo 的 parking drift 纬向均值。

## 运行入口

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py"
```

默认入口即全球 `60S-60N`、5 度分带、全样本。调试或 smoke run 可限制每组
匹配数，并建议输出到单独 smoke 目录：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_global_60S60N_5deg_smoke" `
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

若需要按“涡旋本体跨过某条纬线”而不是涡心纬度带选样本，可使用：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --selection-mode crossing_lat `
  --target-lat 20 `
  --intersect-radius-r 1 `
  --bbox 0,360,-60,60 `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_crossing_20N_1R"
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
比较正式重建 W、历史观测 `I_Wpk` 和速度/符号敏感性。

## 参考依据

本地 PDF 依据见 `ARGO_EDDY_W_LITERATURE_CHECK_ZH.md` 和
`D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\PDF\Dipole_vertical _transport`。
未能下载为有效 PDF 的论文只作为待补充条目，不计入本地依据。
