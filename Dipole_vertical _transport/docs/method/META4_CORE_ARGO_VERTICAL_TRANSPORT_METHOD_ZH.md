# META4.0 + Core Argo 垂直速度重建方法

本文档记录 `Dipole_vertical _transport/meta4_core_argo_vertical_transport.py`
当前正式口径。早期 `lat_band`、非 BOA 背景、非 Cressman 映射、
`profile_diff` 速度口径，以及 `fast-sensitivity/zgeometry/threeway`
诊断入口已从正式入口删除。

## 数据源

- Argo 主数据源：`F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat`
- 历史速度/观测 W 校验源：`F:\Argo_data\Argo1000m_UVW_TSDen_199601_202306.mat`
- META4.0 涡旋源：`F:\Eddy\Eddy\META4.0_DT_allsat`
- BOA 背景位密源：`F:\Argo_data\Self_BOA_Argo_PotentialDensity`
- 默认输出根目录：`E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_crossing_global_60S60N_10deg`

Argo 使用 TEOS-10 派生密度，默认变量为 `I_sigma1`。历史文件中的
`I_Upk/I_Vpk` 用作 1000 m parking drift；`I_Wpk` 只作为观测校验字段，
不参与 `rebuild_W`。

## 正式样本选择

- 正式主流程固定为 crossing：涡旋本体 `1R` 跨过目标纬线。
- 默认纬线：`60S,50S,40S,30S,20S,10S,00N,10N,20N,30N,40N,50N,60N`。
- Argo profile 不再按纬度带预筛，只受 bbox、Core parking depth、时间窗和
  `0-4R` 空间匹配控制。
- Core Argo：`I_ParkDepth = 900-1100 m`。
- 时间窗：Argo profile 与 META 轨迹点相差不超过 `1 day`。
- 空间坐标：`x/R` 是局地东西向，`y/R` 是局地南北向，归一化半径使用
  META `final_radius`。
- 极性：只输出 cyclonic 和 anticyclonic，不生成 combined。
- 推荐匹配：`--match-mode all`，允许同一 Argo profile 对所有满足条件的涡旋
  各计一次；`nearest` 仅保留为显式保守选项。

## 正式等密面几何

正式几何固定为：

```text
BOA monthly climatology -> D'_rho -> Cressman composite -> isopycnal geometry
```

对每条匹配 profile：

```text
rho0 = rho_Argo(parking_depth)
D_rho = D_Argo(rho0)
D_rho_bg = D_BOA_monthly_clim(lon, lat, month, rho0)
D'_rho = D_rho - D_rho_bg
```

`D_rho` 和 `D_rho_bg` 都必须由严格 bracket crossing 反插值得到；无 crossing、
弱层结或 bracket 过厚的样本会被剔除。深度变量保存和图像标识均为正深度向下。
正式流程禁止使用 `eta_rho ≈ rho' / (dρ/dD)` 这类密度导数反推等密面位移；
密度导数只允许出现在热成风速度剪切等物理计算中，不作为等密面几何来源。

Composite analysis 的边界停在等密面几何：它只输出 `D_rho/D_rho_bg/D'_rho`、
合成速度、样本支撑和 QC。`term1/term2/rebuild_W` 由独立 physics rebuild
模块在几何量之后计算。

## W 计算

W 使用向上为正：

```text
term1 = + c_x_rel * dD'_rho/dx
term2 = - [(u_pk - c_x_raw), v_pk] · grad(D'_rho)
rebuild_W = term1 + term2
```

其中 `c_x_raw` 来自 META track 相邻点中央差分；`u_bg` 是同 crossing 组、
同极性、匹配 Core Argo 的 parking drift 纬向均值；`c_x_rel =
mean(c_x_raw) - mean(u_bg)`。

## Cressman 与输出

正式映射只使用 Cressman objective mapping。权重为
`w=(R_c^2-r^2)/(R_c^2+r^2)`，只使用 `R_c` 内样本；推荐平滑后的 20N 2D
口径为 `grid_n=61, Rc=1.0R, min_obs=8, smooth=4`。

默认输出以二进制科学格式为主：匹配表写 `.mat`，网格写 `.mat` 和 `.nc`，
SUMMARY 写 `.mat`。大 CSV 和网格 JSON 默认关闭，需要时显式使用
`--write-matched-csv`、`--write-summary-csv`、`--write-grid-json`。

二维图使用 `contourf(..., 'LineStyle', 'none')`，只显示填色块，不画等值线描边。

## 三维模式

`--vertical-mode thermal_wind_depth_stack` 会在每个名义深度层重复上述 BOA
背景密度反插值，得到 `W(x/R,y/R,z)`。热成风只用于把 1000 m parking drift
延拓成随深度变化的 `u(z),v(z)`：

```text
du/dD =  g/(f rho_ref) * d rho'/dy
dv/dD = -g/(f rho_ref) * d rho'/dx
```

这里 `D` 为正深度向下。速度以匹配到的 1000 m parking drift 为锚点，从
1000 m 向上、向下积分。W 仍向上为正；截面图纵坐标显示正深度数字。

## 保留口径

当前 main worktree 保留两个科学口径的命名：

- `--geometry-mode boa_anomaly`：正式生产口径，已在 main worktree 可运行。
- `--geometry-mode predecessor_hybrid`：前辈兼容口径，定义为
  `term1` 用 Argo composite absolute density 等密面斜率、热成风速度优先从
  Argo composite absolute density 积分、`term2` 用 ISAS/background density
  等密面斜率。该口径目前保留在 Original 验证 worktree，尚未提升为 main
  worktree 的生产入口。

`META3.2 allsat validation` 是 `predecessor_hybrid` 的轨迹源验证方案，不作为
独立科学口径。

## 默认运行

```powershell
D:\Util\lever\02_miniforge\envs\Dipole_vertical_transport\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --match-mode all
```

20N 推荐 2D 运行：

```powershell
D:\Util\lever\02_miniforge\envs\Dipole_vertical_transport\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --target-lat 20 `
  --match-mode all `
  --grid-n 61 `
  --cressman-radius-r 1.0 `
  --cressman-min-obs 8 `
  --smooth-passes 4
```

20N 三维热成风 smoke：

```powershell
D:\Util\lever\02_miniforge\envs\Dipole_vertical_transport\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --target-lat 20 `
  --match-mode all `
  --vertical-mode thermal_wind_depth_stack `
  --depth-levels 100:100:500 `
  --max-matches-per-group 300
```
