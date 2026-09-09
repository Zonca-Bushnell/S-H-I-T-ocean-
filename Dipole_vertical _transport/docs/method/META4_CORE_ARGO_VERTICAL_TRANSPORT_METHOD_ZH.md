# META4.0 + Core Argo 垂直速度重建方法

本文档记录 `Dipole_vertical _transport/meta4_core_argo_vertical_transport.py`
当前正式口径。早期 `lat_band`、非 BOA 背景、非 Cressman 映射和
`profile_diff` 速度口径已从正式入口删除；相关科学疑问只保留在 diagnostics
入口中。

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

## 正式几何与 W

正式几何固定为：

```text
BOA monthly climatology -> z'_rho -> Cressman composite -> gradient/W
```

对每条匹配 profile：

```text
rho0 = rho_Argo(parking_depth)
z_rho = z_Argo(rho0)
z_rho_bg = z_BOA_monthly_clim(lon, lat, month, rho0)
z'_rho = z_rho - z_rho_bg
```

`z_rho` 和 `z_rho_bg` 都必须由严格 bracket crossing 反插值得到；无 crossing、
弱层结或 bracket 过厚的样本会被剔除。深度变量保存和图像标识均为正深度向下。

W 使用向上为正：

```text
term1 = + c_x_rel * dz'_rho/dx
term2 = - [(u_pk - c_x_raw), v_pk] · grad(z'_rho)
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

## 诊断入口

以下入口保留为诊断，不改变正式生产口径：

- `--fast-sensitivity-2d`：复用匹配/QC 缓存，对少量 Cressman 参数做 2D 敏感性。
- `--diagnose-reversal-factors`：检查等密面斜率、热成风速度和 term2 口径对深层反转的影响。
- `--compare-z-geometry-modes`：对比正式 BOA `z'_rho` 几何与前辈式
  `composite density -> isosurface` 几何。
- `--z-geometry-mode`：只在 zgeometry diagnostics 中有效，不允许影响正式生产结果。

## 默认运行

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --match-mode all
```

20N 推荐 2D 诊断：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --target-lat 20 `
  --match-mode all `
  --fast-sensitivity-2d `
  --sensitivity-configs recommended `
  --grid-n 61 `
  --cressman-radius-r 1.0 `
  --cressman-min-obs 8 `
  --smooth-passes 4
```

20N 三维热成风 smoke：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --target-lat 20 `
  --match-mode all `
  --vertical-mode thermal_wind_depth_stack `
  --depth-levels 100:100:500 `
  --max-matches-per-group 300
```
