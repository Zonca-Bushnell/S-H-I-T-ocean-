# ArgoData 重建 W 系统性诊断

更新时间：2026-09-07  
当前分支：`Dipole`  
当前判断：正式程序的问题主要不在 eddy-Argo composite 投影方式，而在 `w = c dz_rho/dx + u · grad(z_rho)` 的物理量构造链条。

## 本地 PDF 状态

已下载并用 PDF 解析确认可读：

- `PDF/Dipole_vertical _transport/JGR Oceans - 2026 - Zhang - Dipole Structure of Vertical Velocity Induced by Mesoscale Eddies.pdf`
- `PDF/Dipole_vertical _transport/Hou_etal_2022_Frontiers_Eddy_beta_spiral.pdf`
- `PDF/Dipole_vertical _transport/Wang_etal_2020_JGR_Eddy_Induced_Acceleration_of_Argo_Floats.pdf`
- `PDF/Dipole_vertical _transport/CortesMorales_etal_2026_ESSD_Global_Thermocline_Vertical_Velocities.pdf`
- `PDF/Dipole_vertical _transport/Roemmich_etal_2020_Frontiers_Argo_Data_1999_2019.pdf`
- `PDF/Dipole_vertical _transport/Gaube_etal_2019_Frontiers_Mesoscale_Eddy_Impacts_Review.pdf`
- `PDF/Dipole_vertical _transport/Liu_etal_2013_JOUC_Eddies_Argo_STMW.pdf`

已检索但尚未拿到本地有效 PDF，因此不作为“已下载文献”计数：

- Christensen et al. 2024, JGR Oceans, `Global Estimates of Mesoscale Vertical Velocity Near 1,000 m From Argo Observations`, DOI `10.1029/2023JC020003`。网页全文可读并已用于核对方法，但 UW 镜像连接失败，ESSOAr 预印本入口被 Cloudflare 拦截，因此尚未计入本地 PDF。
- Freeland 2013, Deep Sea Research II, `Vertical velocity estimates in the North Pacific using Argo floats`, DOI `10.1016/j.dsr2.2012.07.019`。目前只确认题名/DOI，未取得 PDF。
- Colin de Verdiere and Ollitrault 2016, JPO, DOI `10.1175/JPO-D-15-0046.1`。Ifremer 直链返回 403，未取得 PDF。
- Colin de Verdiere et al. 2019, JGR Oceans, DOI `10.1029/2018JC014565`。Wiley 链路未取得 PDF。
- Mason et al. 2017, DSR, Argo/eddy composite 相关论文。SciSpace 链接为人机验证页，未取得 PDF。
- Zhou et al. 2023 BOA_Argo 相关论文。Wiley 链接返回 HTML 拦截页，已删除无效文件。

## 文献方法要点

### 1. 直接 Argo W 不是由剖面等密面坡度直接重建

Cortes-Morales et al. 2026 在综述中把 Freeland 2013 和 Christensen et al. 2024 归入 Argo 观测垂直速度路线。Christensen et al. 2024 的网页全文显示，其核心公式使用 park phase 的温度异常 `T'`、压力异常 `P'`、1000 dbar 附近的温度梯度 `(dT/dP)_1000` 和密度来估计 W；并且数据条件要求使用 trajectory/profile/technical/metadata 等文件，park phase 温度和压力采样少于每 6 小时的 cycle 不能用于估计。关键点是：它依赖 float 停泊段的高频温压轨迹，而不是只用单条上升剖面的 T/S 密度结构。

对我们的影响：

- `ArgoData_SA_CT_PT_PDen_sigma.mat` 提供 TEOS-10 密度剖面和 `I_ParkDepth`，但当前正式脚本没有使用真实 parking segment 的起止位置、停泊压力时间序列或已计算 `I_Wpk`。
- 临时 `Argo1000m_UVW_TSDen_199601_202306.mat` 的 `I_Wpk` 能出现偶极，说明数据中已有的 W 量更接近文献中的直接 Argo W 观测路线。
- 因此正式程序若只从 profile 密度和近似水平速度重建 W，不能预期自动复现 `I_Wpk` 的偶极结构。

### 2. Argo parking drift 应是停泊段位移速度，不是相邻 profile 位置差的粗代理

Roemmich et al. 2020 说明标准 Argo 有 park-and-profile cycle，profile 通常包含停泊、下潜、上升和表面通讯阶段。Wang et al. 2020 讨论 Argo-like trajectory 时明确区分 parking depth trajectory 的起点/终点，并用 ANDRO 或模拟轨迹研究 1000 m 水平速度变化。

当前正式脚本：

```matlab
argo_u(i) = (lon(i+1) - lon(i-1)) / (time(i+1) - time(i-1))
argo_v(i) = (lat(i+1) - lat(i-1)) / (time(i+1) - time(i-1))
```

这个量混入了两个 cycle 之间的全部漂移和表面通讯位置误差，不能严格代表 1000 m parking drift。它会直接污染：

- `u_bg = mean(u_Argo)`
- `c_x_rel = mean(c_x_raw) - mean(u_bg)`
- `term2 = u_Argo · grad(z_rho)`

### 3. `rho0` 不能只用一个纬度带/极性中位数就结束

当前默认 `rho0_mode = band_median`，即每个 group 用一个共同 `rho0`。这能避免逐 profile `rho0` 导致 `z_rho` 全部贴近 parking depth，但也会把跨流系、跨经度、跨水团的背景密度差强行压到一个面上。

文献路线更常见的是：

- 在共同密度层或共同深度层上先构造背景态，再看 anomaly。
- 对 eddy composite，通常合成的是剖面异常或等密面位移异常，而不是绝对 `z_rho`。
- 对大尺度 LVB 方法，Cortes-Morales et al. 2026 是在 isopycnal levels 上建立三维速度/散度闭合，并明确只信任大尺度平滑后的 W。

对我们的影响：

- 当前 `z_rho` 是绝对深度，直接在 eddy 坐标上 Cressman 后求 `dz_rho/dx`。
- 这个梯度同时包含背景纬向/经向密度坡度、水团差异、季节差异和涡旋信号。
- 因此 `term1` 很容易被背景梯度或插值噪声主导，而不是涡旋偶极结构。

### 4. `z_rho` 反插值需要更严格的物理约束

当前 `isopycnal_depth` 会寻找剖面中最接近 parking depth 的 crossing；如果找不到，再用唯一 density 值做 `interp1(profile, depth)`。这里有几个风险：

- 密度剖面存在逆温/噪声/多重 crossing 时，`interp1(profile, depth)` 的密度坐标并不等价于“沿深度单调”的反插值。
- 只限制 `900-1100 m` 可以抑制极端跳点，但也可能把真实等密面起伏截断。
- 没有剔除弱层结、密度范围过窄、profile 垂向分辨率不足、或 crossing bracket 过宽的剖面。

推荐修正是：只做 bracket-based crossing，不再 fallback 到全剖面 `interp1(profile, depth)`；先对密度剖面按深度排序、去重并可选单调化；记录每条剖面的 crossing 数、bracket 厚度、局地 `d rho/dz`，低质量样本不进入梯度计算。

### 5. `term1` 的符号和速度参考系需要重新推导

用户给出的目标式是：

```text
w = c * dz_rho/dx + u · grad(z_rho)
```

但如果 `z_rho(x - c t, y)` 表示随涡旋平移的形态，则固定点时间变化会给出 `partial z/partial t = -c * partial z/partial x`。到底使用 `+c` 还是 `-c` 取决于：

- `x/R` 是否定义为 Argo 相对涡心的 east-positive 坐标。
- `c_x_raw` 是否定义为涡心 eastward positive 传播速度。
- `u` 是地理坐标速度、涡旋随体坐标速度，还是背景扣除后的相对速度。

当前程序没有把这些符号约定写成可测试开关，也没有用 `I_Wpk` 做符号校验。

### 6. `term2` 不应直接用未校验的 `u_Argo` 乘同一个 composite 梯度

`term2 = u_Argo · grad(z_rho)` 当前使用 Cressman 后的 `grid.u/grid.v` 与同一张 `grid.z` 的梯度相乘。风险有三层：

- `u_Argo/v_Argo` 不是严格 parking drift。
- `grid.u/grid.v` 和 `grid.z` 的样本支撑可能不同；即便来自同一 profile，Cressman 后的速度场和密度面场在空白区/边缘的支撑不完全一致。
- 公式里的速度应使用绝对速度、相对涡旋传播速度、背景异常速度，还是只作为 stirring 诊断项，当前没有由文献或 `I_Wpk` 校验固定下来。

### 7. Cressman 合成本身不是主要错误，但参数应成为敏感性测试

Wang et al. 2020 和 Liu et al. 2013 都使用 Cressman/objective interpolation 将不规则 Argo 样本映射到网格。Wang et al. 2020 的权重形式为：

```text
omega = (R^2 - r^2) / (R^2 + r^2)
```

当前脚本采用相同权重形式，并用 `cressman_radius_r = 0.5R`、`min_obs = 3`。这比“把空白补洞后乱画”可靠，但不解决 W 构造变量错误的问题。Cressman 半径、最小样本数和平滑次数应作为敏感性测试，而不是首要改动。

## 当前正式程序差异表

| 环节 | 当前实现 | 文献/物理要求 | 诊断结论 |
| --- | --- | --- | --- |
| Argo 速度 | 相邻 profile 位置差估计 `u/v` | parking-depth displacement velocity 或 ANDRO/Argo1000m 已处理速度 | 必须改 |
| W 观测量 | 从 `z_rho` 和速度重建 | 直接 Argo W 研究多使用 parking pressure/isopycnal displacement；LVB 是大尺度散度闭合 | 必须先校验 |
| `rho0` | group 中位数或逐 profile | 共同密度面应结合背景态/水团，推荐用 anomaly | 必须改 |
| `z_rho` | 绝对等密面深度 | eddy composite 应优先合成等密面位移异常 | 必须改 |
| 反插值 | crossing + fallback `interp1(profile, depth)` | 单调/分段 crossing、质量控制、剔除多重不稳定 crossing | 必须改 |
| `c_x_rel` | `mean(c_x_raw)-mean(u_bg)` | 需明确移动坐标符号和背景速度定义 | 必须敏感性测试 |
| `term1` 梯度 | composite `z_rho` 后求梯度 | 应对 `z_rho_anomaly` 求梯度，并测试正负号 | 必须改 |
| `term2` 配准 | Cressman 后 `u/v` 乘 `grad(z)` | 速度来源/参考系/支撑一致性需独立校验 | 必须改 |
| 合成方式 | eddy-centered `x/R,y/R`，nearest/all 可选 | Argo-eddy composite 常用此框架 | 暂不认为是一阶错误 |
| Cressman | 支持样本掩膜，白色为空 | 文献常用 objective mapping，但参数需敏感性 | 可测试 |

## 推荐修改优先级

### 必须改

1. 将正式程序的 `u/v` 来源改为真实 parking drift：优先尝试把 `Argo1000m_UVW_TSDen_199601_202306.mat` 的 `I_Upk/I_Vpk/I_Wpk` 与 TEOS 数据按 `I_PF + I_Time + I_Lon/I_Lat` 匹配；匹配成功后用 TEOS 密度算 `z_rho`，用历史文件的 `I_Upk/I_Vpk` 做速度，并用 `I_Wpk` 做形态校验。
2. 增加 `z_rho_anomaly`：先建立同密度面在背景场中的深度，例如按纬度、经度、月份或远离涡心 `2-4R` 的背景均值，然后 composite `z_rho - z_rho_bg`。
3. 重写 `isopycnal_depth`：只允许显式 bracket crossing；输出 crossing 个数、bracket 厚度、局地层结强度，弱层结和多重异常 crossing 剔除。
4. 给 `term1` 加符号敏感性：至少输出 `+c dzdx` 和 `-c dzdx` 两套，并用临时 `I_Wpk` 偶极方向决定符号。
5. `term2` 使用相对速度敏感性：测试 `u_abs · grad(z_anom)`、`(u_abs-c) · grad(z_anom)`、`u_anom · grad(z_anom)`，不要默认直接相加为最终 W。

### 可敏感性测试

1. `rho0_mode`：共同密度面、纬度带中位数、每 profile parking density、以及固定 sigma level。
2. 背景定义：远场 `2-4R`、同纬度带 climatology、BOA/ISAS 背景密度面。
3. Cressman 参数：`0.3R/0.5R/0.75R`、`min_obs=3/5/10`、是否平滑。
4. 匹配方式：`nearest` 与 `all` 只作为样本统计敏感性，不作为 W 异常的主要解释。
5. 选择方式：`lat_band` 与 `crossing_lat` 可保留；当前结果说明选样不是一阶问题。

### 暂不改

1. 不立刻恢复全球全样本运行。
2. 不把 BOA_Argo 用来直接推导背景速度；BOA/ISAS 目前只适合作为背景密度/温盐场候选。
3. 不把 `I_Wpk` 临时程序替代正式方法；它只作为观测形态 sanity check 和符号/量级校验。

## 下一步最小验证

建议先做 20N crossing 的小样本闭环：

1. 用临时 `I_Wpk` 程序生成观测 W composite，作为目标偶极方向。
2. 在正式程序中合并历史 `I_Upk/I_Vpk/I_Wpk` 到 TEOS profile。
3. 只改 `u/v` 后跑一次，判断是否恢复偶极。
4. 再改 `z_rho_anomaly` 和反插值 QC，逐项比较 `term1/term2/rebuild_W`。
5. 每一步输出 `SUMMARY` 中的 `match_count/unique_argo_count/duplicate_match_count`、有效格点、`q95_abs_w`，并保存对照图。

如果第一步速度替换后仍无偶极，则优先怀疑 `rho0/z_rho_anomaly/term1 sign`；如果速度替换后形态明显接近 `I_Wpk`，则说明主要问题是当前 `u/v` 代理和 `c_x_rel/u_bg` 背景扣除。

## 2026-09-08 20N crossing 修正验证

已按 20N crossing、`nearest` 匹配和全样本重新运行正式程序，输出目录：

```text
E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_fixed_crossing_20N_1R
```

本次修正内容：

- `u/v` 改用 `Argo1000m_UVW_TSDen_199601_202306.mat` 的 `I_Upk/I_Vpk`，并保存 `I_Wpk` 作为观测对照。
- `z_rho` 改为严格 bracket crossing，保存 crossing 个数、bracket 厚度和局地 `drho/dz`。
- 主图改用 `z_rho_anom = z_rho - median(z_rho in 2-4R)` 后求梯度。
- 同时保存 `term1_plus/minus`、`term2_abs/rel` 和 `wpk_validation.png`。

结果摘要：

| polarity | matches | valid grid cells | corr(rebuild_W, I_Wpk) | q95 rebuild | q95 I_Wpk |
| --- | ---: | ---: | ---: | ---: | ---: |
| cyclonic | 50460 | 5025 | 0.0024 | 16.59 | 7.46 |
| anticyclonic | 47734 | 5025 | 0.0233 | 13.94 | 6.97 |
| combined | 98194 | 5025 | 0.0805 | 12.21 | 2.10 |

结论：速度来源、样本覆盖、Cressman 支撑和基础 `z_rho` QC 已不再是一阶问题；同一批样本下 `I_Wpk` 仍显示中心附近东西偶极，而 `z_rho` 公式重建仍偏带状/斑块状。因此下一步不应继续扩大样本，而应重新审查 `w = c dz_rho/dx + u · grad(z_rho)` 是否能由单时刻 profile 的等密面深度异常闭合到 Argo parking vertical velocity。优先方向是：使用 Christensen/Freeland 类型的 parking-phase pressure/temperature W 算法，或将当前 `z_rho` 项定位为动力诊断项而非直接替代 `I_Wpk`。

补充量化：标量远场背景扣除后，`z_rho_anom` 与 `y/R` 的空间相关很高：
cyclonic `0.720`、anticyclonic `0.823`、combined `0.826`；而 `I_Wpk` 与
`y/R` 的相关接近 0。说明 `z_rho` 的南北背景坡度没有被标量背景扣除清掉，
`term1/term2` 会继承这个背景坡度。因此新增 `--z-mode anomaly_farfield_plane`，
用 `2-4R` 远场样本拟合平面背景 `z_bg=a+b x/R+c y/R`，作为下一步判断
“背景变量处理问题”与“公式本身问题”的分界测试。

平面背景测试结果：

| polarity | corr(z'_rho, y/R) | corr(rebuild_W, I_Wpk) | q95 rebuild | q95 I_Wpk |
| --- | ---: | ---: | ---: | ---: |
| cyclonic | 0.0249 | 0.0084 | 15.72 | 7.46 |
| anticyclonic | 0.0103 | 0.0208 | 12.09 | 6.97 |
| combined | 0.0227 | 0.0892 | 9.87 | 2.10 |

解释：平面背景已经基本去掉了 `z_rho` 中的南北背景坡度，但 `rebuild_W` 与
`I_Wpk` 仍不相关，且形态仍非中心东西偶极。因此当前证据支持“两层结论”：

1. 背景场/中间变量问题确实存在：标量远场背景不够，必须至少用平面或更物理的
   local climatology/BOA/ISAS 背景密度面。
2. 即便改正这一层，`z_rho` 坡度项重建仍不能复现 direct parking `I_Wpk`。这说明
   主要剩余问题在计算方式/观测量定义：单时刻 profile 的等密面深度异常与
   Christensen/Freeland 类型的 parking-phase vertical velocity 不是同一个观测量。

## 2026-09-08 BOA 多年同月气候态背景测试

本轮按文献式“局地/季节/水团背景”方向，将正式程序新增为：

```text
BOA monthly climatology density profile -> z_rho_bg -> z_rho_anom -> Cressman composite -> gradient/W
```

使用的数据源为 `F:\Argo_data\Self_BOA_Argo_PotentialDensity\PDen1000_YYYYMM.mat`。
该产品的 `Den` 是约 `1032 kg/m^3` 的位密，而正式程序默认 `I_sigma1`
是约 `32 kg/m^3` 的 sigma 口径；因此脚本已加入自动单位对齐：
当 BOA 背景剖面约为 `rho` 而目标 `rho0` 为 `sigma` 时，先对 BOA 剖面减
`1000`，再做同一 `rho0` 的 bracket crossing。

新增有效 PDF：

- `Lin_etal_2019_RemoteSensing_Argo_Eddy_Bay_of_Bengal.pdf`：Argo 与卫星涡旋配准后，
  使用温盐/密度异常和涡旋坐标 composite，支持先扣背景再合成的路线。
- `Cressman_1959_MWR_Operational_Objective_Analysis_System.pdf`：确认 Cressman
  权重形式和客观分析思想，当前脚本继续使用同类有限半径加权。

仍未作为有效 PDF 使用：

- `Sandalyuk_etal_2020_Lofoten_Basin_Argo_Eddy_Structure.pdf` 下载后 PDF 解析失败，
  已从有效文献清单排除。

20N crossing、`nearest`、全样本输出目录：

```text
E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_vertical_transport_boa_clim_crossing_20N_1R
```

结果摘要：

| polarity | matches | BOA bg valid | valid grid cells | corr(rebuild_W, I_Wpk) | corr(sample-gradient W, I_Wpk) | q95 rebuild | q95 sample-gradient | q95 I_Wpk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cyclonic | 48555 | 100% | 5025 | -0.0063 | -0.5400 | 13.29 | 19.78 | 7.56 |
| anticyclonic | 45788 | 100% | 5025 | -0.0409 | -0.4954 | 9.88 | 19.83 | 6.89 |
| combined | 94343 | 100% | 5025 | 0.0596 | -0.0922 | 5.35 | 9.43 | 2.23 |

解释：

1. BOA 多年同月背景没有造成样本损失，说明 `lon/lat/month` 背景密度面技术上可行。
2. `I_Wpk` 仍显示中心附近东西偶极，但 BOA 背景后的 `rebuild_W` 仍不是稳定东西偶极，
   与 `I_Wpk` 的空间相关接近 0。
3. “逐样本局地梯度后合成”的诊断没有改善，两个极性反而与 `I_Wpk` 呈明显负相关；
   因此当前差异不主要来自“先合成再求梯度”的数值顺序。
4. 结论进一步收敛：密度异常背景处理确实必须改为 BOA/气候态口径，但即便如此，
   单时刻 `z_rho_anom` 坡度公式仍不能直接复现历史 direct parking `I_Wpk` 偶极。
   下一步应重点检查 `rho0` 是否应改为固定 sigma 面/多 sigma 层，并重新推导
   `w = c dz/dx + u·grad(z)` 与 `I_Wpk = Dz_rho/Dt` 的观测量对应关系。
