# 前辈 eddy_DenField 程序清理与前处理说明

本目录保存前辈新提供的密度场前处理程序。原始包里存在两类重复：

- macOS 资源叉文件 `._*.m`、`.DS_Store`，不是 MATLAB 程序，已删除。
- 按南北半球、气旋/反气旋、ISAS/WOA 数据源复制出来的脚本。多数只改了输入文件、
  `lat_eddy > 0 / < 0`、输出目录，算法主体相同。本次只保留每类算法的代表文件，
  并在下方记录被删除副本对应的参数含义。

## 保留程序清单

| 文件 | 功用 | 数据源 | 主要输出 | 备注 |
|---|---|---|---|---|
| `prepare_Argo_DensityField.m` | 将 ISAS 的 TEMP/PSAL NetCDF 转成原位密度 | `ISAS_Argo/field/*.nc` | `Den_YYYYMM.mat` | 使用 TEOS-10 `gsw_SA_from_SP`、`gsw_CT_from_t`、`gsw_rho(SA,CT,p)`；压力随深度和纬度变化。 |
| `prepare_Argo_PotentialDensityField_1000ref_ISAS.m` | 将 ISAS TEMP/PSAL 转成 1000 dbar 参考密度 | `ISAS_Argo/field/*.nc` | `PDen1000_YYYYMM.mat` | 和上一个不同：最终密度用 `gsw_rho(SA,CT,1000)`，是后续等密面/热成风诊断更常用的密度口径。 |
| `prepare_Argo_PotentialDensityField_1000ref_BOA.m` | 将 BOA 温盐 MAT 转成 1000 dbar 参考密度 | `BOA_Argo/MAT/*.mat` | `PDen1000_YYYYMM.mat` | BOA 版本；保存 `Den, lat_in, lon_in, pres`。 |
| `prepare_Argo_PotentialDensityField_1000ref_WOA.m` | 将 WOA2023 季节温盐场转成 1000 dbar 参考密度 | `WOA2023/seasonal/*.nc` | `PDen1000_13..16.mat` | 四季背景场，月份先被映射到 13-16 的季节编号。 |
| `point_Density_field_composiiton_North_ae_ISAS.m` | 逐涡旋抽取 ISAS 密度点云并投影到涡旋归一化坐标 | META3.2 AE + `Self_ISAS_Argo_PotentialDensity` | `I_x_all, I_y_all, I_Den_all, E_lat, E_lon` | 代表 `point_*_ISAS` 系列；保留 North/AE 作为模板。 |
| `point_Density_field_composiiton_North_ae_WOA.m` | 逐涡旋抽取 WOA 季节密度点云并投影到涡旋归一化坐标 | META3.2 AE + WOA seasonal density | `I_x_all, I_y_all, I_Den_all` | 代表 `point_*_WOA` 系列；月份先转季节。 |
| `average_composition_point_DenField.m` | 将点云文件合成为 81 x 81 x depth 的三维平均密度场 | `DenField_*/*.mat` 点云 | `Den_compound_all` | 使用 `dd = 1.2R` 方形邻域平均；为了速度预分配 `8e7` 行。 |
| `average_Density_field_composiiton_North_ae.m` | 直接围绕每个涡旋切三维密度块并月平均/中位合成 | META3.2 AE + ISAS PDen1000 | `eddy_DenField_monthly` | 代表直接 51 x 51 x depth 体块合成路线。 |
| `average_composition_DenField.m` | 将 51 x 51 合成密度场重插值到 81 x 81 | 已有 `eddy_DenField_monthly` | `eddy_DenField` | 只做 `interp3(...,'spline')` 展示/后处理。 |

## 已删除重复副本

| 删除文件 | 删除原因 | 可由哪个保留文件代表 |
|---|---|---|
| `point_Density_field_composiiton_North_ce_ISAS.m` | 与 North/AE/ISAS 版算法相同，只是读取 CE 七采样涡旋文件并输出到 `DenField_ce_ISAS` | `point_Density_field_composiiton_North_ae_ISAS.m` |
| `point_Density_field_composiiton_South_ae_ISAS.m` | 与 North/AE/ISAS 版算法相同，只是 `lat_eddy < 0` 和输出目录不同 | `point_Density_field_composiiton_North_ae_ISAS.m` |
| `point_Density_field_composiiton_South_ce_ISAS.m` | 与 North/AE/ISAS 版算法相同，只是 CE + 南半球参数不同 | `point_Density_field_composiiton_North_ae_ISAS.m` |
| `point_Density_field_composiiton_North_ce_WOA.m` | 与 North/AE/WOA 版算法相同，只是 CE 输入文件不同 | `point_Density_field_composiiton_North_ae_WOA.m` |
| `average_Density_field_composiiton_North_ce.m` | 与 North/AE 直接体块合成算法高度相似，只是 CE 输入、是否重插值、mean/median 和循环范围不同 | `average_Density_field_composiiton_North_ae.m` |
| `test_average_Density_field_composiiton_North_ae_BOA.m` | 测试脚本，和直接体块合成路线重复；只用于 BOA 单月/单区域试验 | `average_Density_field_composiiton_North_ae.m` + `prepare_Argo_PotentialDensityField_1000ref_BOA.m` |
| `test_average_Density_field_composiiton_North_ae_WOA.m` | 测试脚本，和直接体块合成路线重复；只用于 WOA 单月/单区域试验 | `average_Density_field_composiiton_North_ae.m` + `prepare_Argo_PotentialDensityField_1000ref_WOA.m` |

## 前期处理到底做了什么

### 1. ISAS/BOA/WOA 密度预处理

前辈并不是直接使用温盐做后续 W 计算，而是先把网格化温盐场转成密度场：

```text
SP/T -> SA/CT -> rho
```

其中：

- `prepare_Argo_DensityField.m` 对 ISAS 使用实际压力 `p(z,lat)`，得到原位密度。
- `prepare_Argo_PotentialDensityField_1000ref_ISAS.m` 对 ISAS 使用固定 `1000 dbar`
  参考压力，得到 `PDen1000`。
- `prepare_Argo_PotentialDensityField_1000ref_BOA.m` 对 BOA 温盐场做同样的
  `PDen1000`。
- `prepare_Argo_PotentialDensityField_1000ref_WOA.m` 对 WOA2023 季节场做
  `PDen1000`，并把四季命名为 `13,14,15,16`。

这一步的关键作用是统一后续使用的密度口径。对于我们当前问题，最重要的是
`PDen1000`，因为前辈 term1/term2 的等密面几何和热成风速度都围绕密度场展开。

### 2. 涡旋筛选与时间处理

前辈使用 META3.2 twosat eddy 数据：

```text
latitude, longitude, time, speed_radius, track
```

时间换算为：

```text
timeday = time_eddy + datenum('1950-01-01')
```

并筛选：

```text
2003 < year < 2021
```

部分脚本还对每条涡旋轨迹只取 7 个 snapshot，做法是对同一 `track` 按生命周期等间隔取
7 个点。这一点会显著减少后续样本量和重复涡旋快照。

### 3. 南北半球和极性分组

原始包把半球和极性拆成许多脚本：

```text
North: lat_eddy > 0
South: lat_eddy < 0
ae: anticyclonic
ce: cyclonic
```

算法主体没有必要复制多份；后续我们应改为参数化入口。

### 4. 涡旋坐标投影

对每个入选涡旋，程序在涡心附近先切一个经纬度窗口，常用范围为涡心正负 10 度：

```text
lon_min = lon_c - 10
lon_max = lon_c + 10
lat_min = lat_c - 10
lat_max = lat_c + 10
```

然后用 `distance` 计算相对涡心的东西/南北距离，并除以涡旋半径：

```text
I_x = distance(lat_c, I_lon, lat_c, lon_c) * R_earth / radius_eddy
I_y = distance(I_lat, lon_c, lat_c, lon_c) * R_earth / radius_eddy
```

符号约定是：

```text
I_x < 0: west of eddy center
I_x > 0: east of eddy center
I_y < 0: south of eddy center
I_y > 0: north of eddy center
```

这一点和我们后来严查的 x/y 坐标问题直接相关：前辈代码里 x 是东西向，y 是南北向。

### 5. 点云输出

`point_Density_field_composiiton_*` 系列不是直接生成规则三维场，而是先把每个涡旋附近的
网格密度点摊平成点云：

```text
I_x_all
I_y_all
I_Den_all
E_lat
E_lon
```

其中 `I_Den_all` 是每个相对位置点上的整条密度剖面。随后删除整条剖面中存在 NaN 的点。

### 6. 点云到规则三维合成场

`average_composition_point_DenField.m` 把许多点云文件合并，再映射到统一的
`x/R, y/R = -4:0.1:4` 网格，即 `81 x 81`：

```text
dd = 1.2
index = x in [x0-dd, x0+dd] and y in [y0-dd, y0+dd]
Den_compound_temp(x0,y0,:) = mean(I_Den_in(index,:), 1, 'omitnan')
```

这不是 Cressman，而是方形邻域内简单平均。它有较强平滑作用，也解释了为什么前辈图像
通常比我们早期 Cressman 逐层结果规整。

### 7. 直接三维体块合成路线

`average_Density_field_composiiton_North_ae.m` 走另一条路线：不是先存点云，而是对每个
涡旋直接把局地密度块插值到 `51 x 51 x depth` 的归一化涡旋坐标盒里，再对同月所有涡旋
做 median：

```text
eddy_DenField(:,:,:,i) = interp3(...)
eddy_DenField_monthly(:,:,:,k) = median(eddy_DenField,4,'omitmissing')
```

然后 `average_composition_DenField.m` 可将 51 x 51 的结果用 spline 插值到 81 x 81。

## 你可能没特别注意、但很关键的程序/细节

1. `prepare_Argo_DensityField.m` 和 `prepare_Argo_PotentialDensityField_1000ref_ISAS.m`
   看起来重复，但物理量不同：前者是原位密度，后者是 1000 dbar 参考密度。
2. `average_composition_point_DenField.m` 使用 `dd = 1.2R` 的方形邻域平均，不是
   Cressman。这个会天然让图更规整，也可能让小尺度结构被抹平。
3. 点云脚本会删除任意深度存在 NaN 的整条点剖面，这个 QC 很强，会改变深层可用区域。
4. 直接体块合成脚本有 `median` 和 `mean` 两种变体；AE 测试版偏向 median，CE 版偏向
   mean。这可能导致极性之间图像平滑度不同。
5. 有些脚本只跑单月或单个索引，例如 `for k = 1`、`for k = 100`、`for i = 36000`，
   属于测试残留，不能当成正式全量流程。
6. 原始路径全部是 macOS `/Users/Root/...`，直接在当前 Windows 环境运行会失败；
   后续如果要复现，应先统一路径配置。

## 与我们当前改版的关系

当前改版已经吸收了前辈的关键思想：Argo composite absolute density 给 term1 和热成风
速度，ISAS/background density 给 term2 背景等密面斜率。但本目录提醒我们还有两个差异：

- 前辈点云合成是 `dd = 1.2R` 方形邻域平均，不是 Cressman。
- 前辈可能先经过强 QC 和分批随机/分组平均，图像天然更规整。

因此，若后续继续追求前辈式图像形态，需要比较 `Cressman` 与 `1.2R box mean`，
以及 `mean/median`、整剖面 NaN 剔除、7 snapshot 采样对结果的影响。
