# META4.0 + Core Argo 垂直速度重建方法

本文档记录 `Dipole_vertical _transport/meta4_core_argo_vertical_transport.py`
的第一版科学口径和工程假定。

## 数据源

- Argo 主数据源：`F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat`
- META4.0 涡旋源：`F:\Eddy\Eddy\META4.0_DT_allsat`
- 结果根目录：`E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_global_60S60N_5deg`

Argo 使用 TEOS-10 派生密度变量，默认读取 `I_sigma1`。旧
`Argo1000m_UVW_TSDen_199601_202306.mat` 仅作为历史结果和数量级 sanity
check，不作为正式密度口径。

## 样本选择

- 区域：全球 Argo 常规有效覆盖带 `0E-360E, 60S-60N`。
- 纬度带：默认按 5 度分带，从 `60S-55S` 到 `55N-60N`。
- Argo 类型：只取 Core Argo，第一版定义为 `I_ParkDepth` 在 `900-1100 m`。
- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `1 day`。
- 空间匹配：以 META 涡心为原点，以 `final_radius` 为 `R`，保留 `0-4R`
  内 profile，并标记 `0-1R`、`1-2R`、`2-4R`。
- 极性：cyclonic 和 anticyclonic 分开计算，并输出 combined 汇总。
- 目录标签：纬度带使用 `60S_55S`、`05S_00N`、`00N_05N`、`55N_60N`
  这类半球显式标签，避免南半球被误标为 `N`。

## 速度与垂直速度分解

第一版重建：

```text
rebuild_W = c_x_rel * dz_rho/dx + u_Argo · grad(z_rho)
```

其中：

- `c_x_raw`：从 META track 相邻轨迹点中央差分得到的局地东西向传播速度。
- `u_bg`：同纬度带、同极性、匹配 Core Argo 的 parking drift 纬向均值。
- `c_x_rel = mean(c_x_raw) - mean(u_bg)`。
- `rho0`：默认取每个纬度带/极性内，匹配 Core Argo 在实际 parking depth
  处 `I_sigma1` 的中位数，作为共同目标等密面。
- `z_rho`：每条 profile 上共同 `rho0` 对应的等密面深度。
- `z_rho` 有效窗口：默认只保留 `900-1100 m`，避免共同密度面在个别
  profile 中跳到浅层或深层交点，造成不合理的大梯度和 W 量级。

注意：旧的逐 profile parking-density 口径会使 `z_rho` 几乎退化为每条
profile 自身的 parking depth，导致 `dz_rho/dx` 和重建 W 接近 0。脚本保留
`--rho0-mode profile` 作为历史对照，但正式 composite 默认使用共同
`rho0`，更符合“同一等密面深度起伏”的物理定义。

## 网格覆盖与空白区

复合图默认先把 profile 投影到涡心归一化坐标，再用 MATLAB
`scatteredInterpolant` 生成连续 `z_rho/u/v` composite 场。脚本同时在
`composite_grid.json/.npz` 中写出原始 bin 计数 `sample_count` 和插值支撑
`mapped_support`。

`dz_rho/dx` 的数值梯度在连续 composite 场上计算；白色表示当前插值方式没有
空间支撑。若需要查看完全不插值的原始散点格点结果，可加
`--grid-mapping bin`。如果使用 `--max-matches-per-group 200` 做 smoke run，
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

若希望提高单格可靠性，可增加最小格点样本数，例如：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --min-bin-count 3
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

## 参考依据

- NOAA AOML Argo overview: https://www.aoml.noaa.gov/two-decades-argo-program/
- NOAA Argo best practices PDF: https://repository.library.noaa.gov/view/noaa/70164/noaa_70164_DS1.pdf
- Lin et al. 2019 Remote Sensing: https://www.mdpi.com/2072-4292/11/24/2989
- Zhou et al. 2023 JGR Oceans: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JC019386
- JAMSTEC Argo gridded products: https://www.jamstec.go.jp/PARC/product
