# META4.0 + Core Argo 垂直速度重建方法

本文档记录 `Dipole_vertical _transport/meta4_core_argo_vertical_transport.py`
的第一版科学口径和工程假定。

## 数据源

- Argo 主数据源：`F:\Argo_data\ArgoData_SA_CT_PT_PDen_sigma.mat`
- META4.0 涡旋源：`F:\Eddy\Eddy\META4.0_DT_allsat`
- 结果根目录：`E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport`

Argo 使用 TEOS-10 派生密度变量，默认读取 `I_sigma1`。旧
`Argo1000m_UVW_TSDen_199601_202306.mat` 仅作为历史结果和数量级 sanity
check，不作为正式密度口径。

## 样本选择

- 区域：黑潮 `120E-145E, 20N-35N`。
- 纬度带：`20-25N`、`25-30N`、`30-35N`。
- Argo 类型：只取 Core Argo，第一版定义为 `I_ParkDepth` 在 `900-1100 m`。
- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `1 day`。
- 空间匹配：以 META 涡心为原点，以 `final_radius` 为 `R`，保留 `0-4R`
  内 profile，并标记 `0-1R`、`1-2R`、`2-4R`。
- 极性：cyclonic 和 anticyclonic 分开计算，并输出 combined 汇总。

## 速度与垂直速度分解

第一版重建：

```text
rebuild_W = c_x_rel * dz_rho/dx + u_Argo · grad(z_rho)
```

其中：

- `c_x_raw`：从 META track 相邻轨迹点中央差分得到的局地东西向传播速度。
- `u_bg`：同纬度带、同极性、匹配 Core Argo 的 parking drift 纬向均值。
- `c_x_rel = mean(c_x_raw) - mean(u_bg)`。
- `rho0`：每条 Core Argo 在实际 parking depth 处的 `I_sigma1`。
- `z_rho`：该 profile 上 `rho0` 对应的等密面深度。

注意：第一版使用每条 profile 自身 parking-depth density 定义 `rho0`，所以
`z_rho` 在观测点会接近该 profile 的实际 parking depth。这个口径保留了用户
示意图与历史脚本的可追溯性；若后续需要更强的等密面 heave 解释，应扩展为
固定 `sigma1` 面或 BOA 背景 `z_rho` 面。

## 网格覆盖与空白区

复合图中的白色格点表示该 `x/R, y/R` 网格没有 Argo 样本支撑。脚本会在
`composite_grid.json/.npz` 中写出 `sample_count`，并在运行摘要里记录每组
有效网格覆盖率。

`dz_rho/dx` 的数值梯度需要先对稀疏 `z_rho` 网格做局部补洞，但默认输出会再
按 `sample_count >= --min-bin-count` 掩膜，只显示有观测支撑的 `term1`、
`term2` 和 `rebuild_W`。如果使用 `--max-matches-per-group 200` 做 smoke
run，图上大面积空白是预期现象；正式结果应使用默认 `0` 读取全部匹配样本。

## BOA_Argo 假定

BOA_Argo 可作为 gridded 温盐/密度背景产品的科学依据，适合后续估计背景
密度面或气候态密度结构。第一版不从 BOA_Argo 直接推导背景速度；涡旋相对
传播速度的背景扣除使用匹配 Core Argo 的 parking drift 纬向均值。

## 运行入口

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py"
```

调试或 smoke run 可限制每组匹配数：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --max-matches-per-group 200
```

若希望提高单格可靠性，可增加最小格点样本数，例如：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --min-bin-count 3
```

## 参考依据

- NOAA AOML Argo overview: https://www.aoml.noaa.gov/two-decades-argo-program/
- NOAA Argo best practices PDF: https://repository.library.noaa.gov/view/noaa/70164/noaa_70164_DS1.pdf
- Lin et al. 2019 Remote Sensing: https://www.mdpi.com/2072-4292/11/24/2989
- Zhou et al. 2023 JGR Oceans: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JC019386
- JAMSTEC Argo gridded products: https://www.jamstec.go.jp/PARC/product
