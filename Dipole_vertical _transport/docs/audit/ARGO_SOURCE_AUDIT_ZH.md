# Argo 垂直速度数据源审计

- 生成时间：`2026-09-07 19:18:19`
- Argo 数据根目录：`F:\\Argo_data`
- 审计区域：`120, 145, 20, 35`，即黑潮区域 `120E-145E, 20N-35N`
- 第一阶段目标：判断哪类 Argo 数据适合后续估算 `w = c * dz_rho/dx + u · grad(z_rho)`，不做全量垂直速度计算。

## 结论

- 主数据源推荐：`ArgoData_SA_CT_PT_PDen_sigma.mat`。它保留逐 profile 序列、float 编号、时间、经纬度、parking depth，并包含 TEOS-10 派生密度变量，最适合后续反插值得到等密面深度 `z_rho`。
- 历史验证源推荐：`Argo1000m_UVW_TSDen_199601_202306.mat`。它已经包含 `I_Upk/I_Vpk/I_Wpk`，可用来复现和检查旧的 parking-drift / isopycnal-displacement 方法，但密度口径应作为旧口径对照。
- 背景辅助源推荐：`BOA_Argo` 和 `ISAS_Argo`。它们是网格化温盐/密度背景产品，适合辅助估计背景场或气候态梯度，不适合作为 parking drift 主数据源。

## 黑潮区域核心统计

| 数据源 | 总 profile | 区域内 profile | 有效 parking depth | 同 float 相邻配对 | 配对中位间隔 | 900-1100 m | 1400-1600 m | >=1800 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TEOS 派生逐 profile | 2,715,467 | 68,072 | 65,505 | 67,073 | 10 天 | 54,634 | 6,004 | 1,524 |
| 旧 UVW/T/S/Density | 2,596,522 | 65,505 | 65,505 | 64,557 | 10 天 | 54,634 | 6,004 | 1,524 |

## 候选数据源判断

### `ArgoData_SA_CT_PT_PDen_sigma.mat`

- 文件大小：`13.65 GB`
- 深度层数：`81`，范围 `0` 到 `2000` m。
- 关键变量存在性：`I_Time`=true，`I_Lon/I_Lat`=true，`I_ParkDepth`=true，`I_PF`=true，`I_sigma1/I_PDen1`=true。
- 适用性：适合作为正式主源。后续可按 float 相邻 cycle 估计 parking drift，并用 TEOS-10 密度剖面反插值得到 `z_rho`。

### `Argo1000m_UVW_TSDen_199601_202306.mat`

- 文件大小：`5.98 GB`
- 已有区域内有效速度：`U=64,357`，`V=64,353`，`W=41,309`。
- 1000 m parking 区域内有效 `Wpk`：`38,841`，中位数 `6.749e-07` m/s。
- 适用性：适合做历史方法复现和数量级检查，不建议作为新模块正式密度口径。

### 网格化或辅助产品

| 数据源 | 文件数 | MAT 文件 | NetCDF 文件 | 总大小 | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `BOA_Argo` | 245 | 238 | 0 | 5.39 GB | 背景温盐/密度场辅助；不含逐 float parking drift。 |
| `ISAS_Argo` | 1,416 | 0 | 1,368 | 257.40 GB | 背景温盐场辅助；不含逐 float parking drift。 |
| `EasyOneArgo` | 27 | 0 | 0 | 7.26 GB | 网格产品候选；先不作为主源。 |
| `ARGO` | 50 | 27 | 0 | 39.21 GB | 可能含历史 GDAC/逐 profile 材料，可作为追溯来源。 |
| `Argos` | 172 | 77 | 0 | 58.48 GB | 非本任务主源，暂不用于等密面垂直速度。 |
| `Self_BOA_Argo_PotentialDensity` | 227 | 225 | 0 | 2.56 GB | 月度位密网格辅助；不含 float drift。 |

## 后续实现建议

- 用 `ArgoData_SA_CT_PT_PDen_sigma.mat` 建立正式 profile reader，读取 `I_Time/I_Lon/I_Lat/I_ParkDepth/I_PF/I_sigma1` 或指定密度变量。
- 对每个中心 profile，用同一 float 的前后 profile 估计 parking drift：`u = dx/dt`，`v = dy/dt`。
- 对指定密度面 `rho`，在每条密度剖面上单调清洗后反插值得到 `z_rho`。
- 第一项使用涡旋传播速度 `c` 和局地东西向 `dz_rho/dx`；第二项使用 Argo parking drift 与 `grad(z_rho)`。
- 旧 `I_Wpk` 结果只用于 sanity check：数量级、符号分布、区域中位数和有效样本数。
