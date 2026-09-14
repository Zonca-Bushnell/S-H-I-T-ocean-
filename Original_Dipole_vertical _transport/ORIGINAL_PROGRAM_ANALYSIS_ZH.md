# 前辈原始 Dipole 垂直输送程序审计

审计对象为本目录中的三个 MATLAB 程序：

- `rebuild_eddy_W.m`
- `test01_dzdt_induced_W.m`
- `test02_UV_induced_W.m`

这些程序并不是从原始 Argo/META 数据开始的完整端到端管线，而是基于若干已经预处理好的中间产品继续计算：

```text
Argo/META 配准样本 -> eddy composite density/velocity/background density
                 -> thermal-wind velocity
                 -> isopycnal slope
                 -> W_dzdt + W_is
```

## 程序职责

| 程序 | 主要作用 | 输出 |
|---|---|---|
| `test01_dzdt_induced_W.m` | 只计算涡旋平移诱导项 `W_dzdt = c0 * dzdx`。密度场来自 `Den_compound`，速度 `c0` 来自 `E_mspeed` 中位数。 | `CE_North_dzdt_W_smooth15.mat`，含 `dzdx/dzdy/W_dzdt/Depth1/c0/grid_dis` |
| `test02_UV_induced_W.m` | 只计算水平流过等密面斜率诱导项 `W_is = U_thw*dzdx + V_thw*dzdy`。密度斜率来自 ISAS 背景密度合成场，速度来自已保存的热成风速度。 | `CE_North_BackgroundDen_W_smooth2.mat`，含 `dzdx/dzdy/W_is/W_uis/W_vis/Depth1/grid_dis` |
| `rebuild_eddy_W.m` | 总控式脚本：重新计算 thermal-wind `U_thw/V_thw`，分别计算 `UV induced W` 和 `dzdt induced W`，最后 `W = W_dzdt + W_is`。 | `test_AE_North_rebuild_W.mat`，含 `W_is/W_dzdt/W/Depth1` |

## 数据来源

| 类别 | 前辈程序使用 | 说明 |
|---|---|---|
| Argo 涡旋配准/合成点 | `/Users/Root/Output/Eddy Heat Flux/Argo02_data_point_ce_North_twosat.mat`、`Argo02_data_point_ae_North_twosat.mat` | 已经是按涡旋类别、半球、twosat 数据处理后的点数据；包含 `E_radius/E_mspeed/E_lat/Depth1` 等。 |
| Argo 合成密度和 1000 m 速度 | `/Users/Root/Output/Eddy Heat Flux/Argo03_compound_ce_North_res0.6_median.mat` | 包含 `Den_compound/U1000_compound/V1000_compound/W1000_compound`；说明前置程序已经完成 composite。 |
| 背景密度场 | `/Users/Root/Output/Eddy Heat Flux/Argo04_data_DenField_ce_North_twosat_ISAS_7Sample.mat` | 包含 `Den_compound_all`，来自 ISAS gridded Argo 背景场，后续取 `mean(...,4)`。 |
| ISAS 深度坐标 | `/Users/Root/Data/Argo_Data/ISAS_Argo/field/2004/ISAS20_ARGO_20040615_fld_TEMP.nc` | 只读取 `depth`，取前 152 层并将第一层设为 0。 |
| META 涡旋半径 | `/Users/Root/Data/AVISO_Eddy/META3.2_DT_twosat/META3.2_DT_twosat_Cyclonic_long_19930101_20220209.nc` | 在 `rebuild_eddy_W.m` 中读取 `speed_radius/latitude`，筛选北半球。 |
| 涡旋传播速度 | `/Users/Root/Output/Eddy Heat Flux/eddy_features/eddy_moving_speed/Twosat_CE_eddy_zonal_moving_speed.mat` 或 `E_mspeed` | `rebuild_eddy_W.m` 用 `abs(median(moving_speed_zonal))`；`test01` 用 `median(E_mspeed)`。 |
| 热成风速度中间结果 | `/Users/Root/Output/Eddy Heat Flux/rebuild_W/UV_induced_W/AE_North_ThermalWind_UV_smooth1.mat` | `test02` 直接读取 `U_thw/V_thw`，没有在本脚本中重新计算。 |

## 前辈计算链条

### 1. 网格和尺度

前辈程序统一采用 `-4R:0.1R:4R`，即 `81 x 81` 的涡旋归一化网格。物理距离定义为：

```matlab
grid_dis = 8 * radius_eddy / 80;
```

其中 `radius_eddy` 来自匹配涡旋半径的均值或中位数。也就是说，所有涡旋先被归一化到 `R` 坐标，再使用平均/中位半径换算成米。

### 2. 热成风速度

`rebuild_eddy_W.m` 中，热成风以 `1000 m` 的合成 parking drift 为锚点：

```matlab
U_thw(:,:,41) = U1000_compound;
V_thw(:,:,41) = V1000_compound;
```

然后由合成密度场水平梯度积分：

```text
dU/dz =  g/f/rho_ref * d rho/dy
dV/dz = -g/f/rho_ref * d rho/dx
```

程序中深度层间隔为 `25 m`，向上积分时加 `+25`，向下积分时加 `-25`。该实现实际使用的是合成密度场 `Den_compound` 或其平滑版本。

### 3. 等密面斜率

前辈方法的关键不是先构造 `z'_rho = z_rho - z_rho_bg`，而是在合成密度场中直接计算整体等密面集合的斜率：

```text
对中心格点 (i,j,z0)，取 rho0 = rho_comp(i,j,z0)
在左右相邻整条 rho(z) 剖面中反插同一 rho0 得到 z_right, z_left
dzdx = 0.5 * (z_right - z_left) / grid_dis
```

南北向 `dzdy` 同理。

这正是我们后续 `zgeometry comparison` 中 B/C/D panel 所复现的核心差异来源。

### 4. 垂直速度公式

前辈程序拆成两项：

```text
W_dzdt = c0 * dzdx
W_is   = U_thw * dzdx + V_thw * dzdy
W      = W_dzdt + W_is
```

其中 `W_is` 对应水平流穿过等密面斜率诱导的 vertical velocity，`W_dzdt` 对应涡旋平移穿过等密面坡度诱导的 vertical velocity。

注意：脚本中使用 `griddedInterpolant(den, -Depth1)`，因此斜率计算中的 `z` 是负深度坐标；这与我们后续统一“W 向上为正、深度显示正向下”的符号口径需要逐项换算。

## 与我们当前正式程序的差异

| 环节 | 前辈程序 | 我们当前正式程序 | 影响 |
|---|---|---|---|
| 原始数据阶段 | 不从原始 Argo/META 开始，依赖已生成的 `Argo02/03/04` 中间产品。 | 从 `ArgoData_SA_CT_PT_PDen_sigma.mat`、META4.0、BOA 背景和历史 `Argo1000m` 速度文件重新匹配。 | 我们可追溯性更强；前辈程序更像后处理脚本。 |
| 涡旋数据 | META3.2 DT twosat，示例为 North CE/AE。 | META4.0 DT allsat，crossing latitude，cyclonic/anticyclonic 分开。 | 数据版本、涡旋半径、生命周期和轨迹速度都可能不同。 |
| 样本选择 | 已经在前置文件里完成；本脚本看不到 Argo-META 时间窗、空间窗和重复匹配规则。 | 显式使用 crossing、`0-4R`、`within 1 day`、`match-mode all/nearest`。 | 前辈结果无法仅从这三个脚本复现样本选择。 |
| 网格 | 固定 `81 x 81`，`0.1R` 间隔。 | 推荐 2D 为 `61 x 61`，Cressman 半径 `1R`，3D 可为 `61 x 61 x depth`。 | 前辈图更细，但不代表样本支撑更高。 |
| 映射方式 | 这些脚本不做 Cressman；输入已经是合成网格。 | 从散点 Argo-META matches 经 Cressman/objective mapping 到网格。 | 前辈的实际 mapping 发生在缺失的前置程序中，需要另查。 |
| 密度场 | `Den_compound` 或 ISAS `Den_compound_all` 的合成密度场。 | 正式默认 BOA monthly climatology 背景下的 `z'_rho` 异常几何；诊断中可做 composite density isosurface。 | 这是造成深层反转差异的最大环节。 |
| `z_rho` 几何 | 整体合成密度场中的绝对等密面集合。 | 正式生产是 BOA 背景等密面异常 `z'_rho`；前辈式只在 diagnostics。 | 前辈式更容易保留深层背景/水团结构并出现反转。 |
| 反插值 | 对相邻柱密度剖面做 `griddedInterpolant(den, -Depth1)`；若剖面非单调，用局部替换强制修正。 | 严格 bracket crossing，剔除弱层结/过厚 bracket；不再使用无 crossing fallback。 | 我们 QC 更严格；前辈单调修正更激进，可能保留更多深层结构，也可能引入人工斜率。 |
| 平滑 | 多处 `ndnanfilter(rectwin)`，窗口分别有 `[2 2]`、`[4 4]`、`[6 6]`、`[8 8]`。 | Cressman 后可平滑，推荐 `smooth=4`；3D/诊断有独立平滑。 | 前辈对密度和斜率平滑很强，图像更规整。 |
| 热成风速度 | 从合成密度场梯度积分，锚定 `U1000_compound/V1000_compound`。 | 从 BOA/Argo 密度异常梯度积分，锚定匹配历史 `I_Upk/I_Vpk`。 | 前辈速度与整体密度几何更自洽；我们速度与异常几何更自洽。 |
| 涡旋传播速度 `c` | `abs(median(moving_speed_zonal))` 或 `median(E_mspeed)`。 | `c_x_rel = mean(c_x_raw) - mean(u_bg)`，分 crossing/极性计算。 | 前辈取绝对中位速度，可能消去东西向符号；我们保留相对传播速度定义。 |
| `term1` | `W_dzdt = c0 * dzdx`。 | `term1 = +c_x_rel * dz'_rho/dx`。 | 前辈用整体等密面斜率；我们用异常等密面斜率。 |
| `term2` | `W_is = U_thw*dzdx + V_thw*dzdy`。 | `term2 = -[(u_tw-c_x_raw, v_tw) · grad(z'_rho)]`。 | 符号、速度参考系、斜率几何都不同。 |
| W 符号 | 使用 `-Depth1` 参与反插值，最终变量未显式声明向上/向下。 | 明确 `W` 向上为正，深度变量正向下。 | 对照时必须统一符号，否则会出现方向误读。 |
| 输出 | 只保存 MAT，变量较少。 | MAT/NC/PNG/Markdown，可关闭 CSV/JSON。 | 我们产物更可审计。 |

## 当前最重要结论

前辈程序产生深层反转，最相关的环节是：

```text
在整体合成密度场 Den_compound 或 Den_compound_all 中计算绝对等密面集合的斜率。
```

它不是用 BOA 气候态先扣掉背景再求异常等密面起伏，而是把合成后的绝对密度结构直接转成 `dzdx/dzdy`。因此深层背景斜率、水团结构、ISAS 背景密度场的垂向变化都会进入 `W_dzdt` 和 `W_is`。

这与我们当前正式生产口径的哲学不同：

```text
我们：BOA monthly climatology -> z'_rho anomaly -> Cressman composite -> gradient/W
前辈：composite density field -> absolute isopycnal slope -> thermal wind/W
```

所以如果目标是复现前辈图像，必须保留 `composite density isosurface` 作为正式候选或至少作为强诊断口径；如果目标是只解释“涡旋相对于背景”的异常输送，则 BOA anomaly 口径更干净，但深层反转可能被背景几何一并扣掉。

## 需要进一步确认的缺口

仅凭这三个程序，还不能确认以下前置步骤：

1. `Argo02_data_point_*` 是如何从 Argo 与 META3.2 匹配生成的：时间窗、空间窗、是否重复匹配、是否筛选生命周期。
2. `Argo03_compound_*` 的 composite 方法：bin mean/median、Cressman/objective mapping、样本数阈值、异常定义。
3. `Argo04_data_DenField_*_ISAS_7Sample.mat` 的背景场构造：7Sample 的含义、月份/季节处理、是否按涡旋日期取 ISAS。
4. `U1000_compound/V1000_compound` 是否来自 Argo parking drift、geostrophic velocity，还是其他速度产品。
5. 前辈最终 W 的符号约定：变量名没有明确说明向上为正或深度向下为正，需要通过图和物理解释反推。

## 扩展逐项差异清单

下表按实际计算链条列出前辈程序与当前 Dipole 正式管线的差异。这里的“当前正式管线”指 `Dipole_vertical _transport/matlab/` 下的 crossing + BOA anomaly + Cressman + thermal-wind W 口径；整体等密面集合只作为 diagnostics 存在。

| # | 环节 | 前辈程序 | 当前正式管线 | 可能影响 |
|---|---|---|---|---|
| 1 | 入口形态 | 三个脚本式 MATLAB 文件，依赖工作区和固定绝对路径。 | Python CLI 调 MATLAB 后端，参数显式，MATLAB 后端模块化。 | 前辈脚本复现性依赖本机中间文件；我们可追踪参数但流程更复杂。 |
| 2 | 数据版本 | META3.2 DT twosat。 | META4.0 DT allsat。 | 涡旋半径、轨迹、生命周期和传播速度样本不同。 |
| 3 | 极性命名 | 文件名中 CE/AE 混用较明显；`rebuild_eddy_W.m` 加载 CE 数据但输出名含 AE。 | cyclonic/anticyclonic 分开，目录和 summary 显式。 | 前辈脚本需确认 CE/AE 是否在某些输出名中写反。 |
| 4 | 半球/纬度范围 | 示例为 North，筛选 `lat_eddy > 0` 或加载 `*_North_*`。 | 默认 crossing latitude，可为 20N、全球 60S-60N 等。 | 前辈是半球平均口径；我们是纬线穿越口径。 |
| 5 | Argo-META 匹配 | 不在这三个脚本内完成，已封装进 `Argo02/03/04` 中间文件。 | 明确用 `within 1 day`、`r/R<=4`、crossing、本体 `1R` 过纬线、`match-mode all/nearest`。 | 前辈匹配规则不可见，是最大不可审计前置差异之一。 |
| 6 | Argo 是否重复匹配 | 三个脚本看不到；取决于前置 `Argo02/03/04`。 | 可选 `match-mode all`；满意 20N 口径使用 all。 | 若前辈前置是重复投影，则与我们 all 更接近。 |
| 7 | Core Argo 筛选 | 三个脚本没有显式 `900-1100 m` 筛选，可能在前置文件中完成。 | 显式 Core Argo：`I_ParkDepth=900-1100 m`。 | 前辈若混入非 1000 m parking float，会改变速度/密度样本。 |
| 8 | 密度主源 | `Den_compound`：已合成 Argo 密度场；另有 `Den_compound_all`：ISAS 背景密度合成场。 | `ArgoData_SA_CT_PT_PDen_sigma.mat` 的 TEOS-10 profile + BOA monthly climatology 背景。 | 前辈密度源是“合成后场”；我们从逐 profile 和 BOA 背景构造。 |
| 9 | 背景密度来源 | ISAS gridded Argo 背景场，读 `ISAS20_ARGO_20040615_fld_TEMP.nc` 只为 depth 坐标；实际密度在 `Argo04...ISAS_7Sample.mat`。 | `Self_BOA_Argo_PotentialDensity/PDen1000_YYYYMM.mat` 多年同月气候态。 | ISAS 与 BOA 的客观分析、时空分辨率、季节处理不同。 |
| 10 | 背景密度算法 | `eddy_DenField_in = mean(Den_compound_all,4,'omitnan')`，即对第 4 维直接平均。 | 对所有 `PDen1000_YYYYMM.mat` 按月份累加，形成 12 个月多年同月 climatology，再按 profile lon/lat/month 双线性插值。 | 前辈更像合成涡旋背景密度场；我们是局地月气候态背景。 |
| 11 | 异常密度定义 | 三个脚本没有显式 `rho' = rho - rho_bg` 作为 W 主几何；直接使用绝对 `Den_compound` 或 ISAS 背景密度场。 | `rho_anom = rho_profile(z0) - rho_BOA(lon,lat,month,z0)`，并保存 `rho_abs`。 | 我们去掉背景密度结构；前辈保留背景/水团/合成绝对结构。 |
| 12 | `rho0` 定义 | 对每个网格点/深度，`den0 = Den_compound_smooth(i,j,:)`，即中心柱每层密度本身作为目标密度集合。 | 对每个 profile 和名义深度 `z0`，`rho0 = rho_BOA(lon,lat,month,z0)`。 | 前辈是“合成场自洽等密面”；我们是“局地 BOA 背景目标密度”。 |
| 13 | `z_rho` 定义 | 在合成密度场相邻柱中寻找同一个 `den0` 的深度，得到整体等密面斜率。 | 在单条 Argo profile 和 BOA profile 中分别找同一个 `rho0` 的深度，取 `z_anom = z_profile - z_bg`。 | 这是深层反转是否出现的核心差异。 |
| 14 | `z_rho` 是否异常化 | 不异常化，直接 `z2-z1` 得到绝对等密面坡度。 | 先异常化 `z'_rho = z_rho - z_bg`，再 Cressman、平滑、求梯度。 | 我们可能去掉了导致深层反转的背景等密面几何。 |
| 15 | 单调化方式 | 发现 `diff(den)<0` 后，用相邻层外推/替换；这是局部强制修复。 | 正式 `isopycnal_depth_qc` 不强制改剖面，只找 bracket crossing；diagnostic `monotonic_density_profile` 是逐层最小递增修复。 | 前辈更激进，保留更多点但可能人工改变深层斜率。 |
| 16 | 反插值函数 | `griddedInterpolant(den, -Depth1, 'linear','none')`，用密度作自变量、负深度作因变量。 | `isopycnal_depth_qc(depth, profile, rho0, z0)`，在相邻深度 bracket 中线性反插正深度。 | 前辈 z 轴为负深度；我们保存正深度向下，W 再显式换成向上为正。 |
| 17 | 多重 crossing | 前辈单调化后 `interp`，基本不显式记录 crossing 数。 | 记录 crossing_count，并选择距离目标深度最近的 crossing；弱层结/大 bracket 剔除。 | 我们 QC 更严格；前辈更平滑连续但不易审计。 |
| 18 | bracket 厚度/QC | 没有显式 bracket 厚度阈值。 | `max_rho_bracket_dz_m` 和 `min_drho_dz` 控制。 | 我们会丢弃深层弱层结或粗 bracket 样本；前辈可能保留。 |
| 19 | 垂向层数 | `Depth1=(0:25:2000)'`，共 81 层。 | 2D 为单/若干层；3D 可 `10:10:2000`，共 200 层。 | 我们垂向分辨率更高，但可能更敏感于噪声。 |
| 20 | 水平网格 | 固定 `-4:0.1:4`，81x81。 | 推荐 `grid_n=61`，也可 81；x/y = `linspace(-4,4,grid_n)`。 | 前辈水平网格更细；我们 recommended 更平滑。 |
| 21 | 网格物理距离 | `grid_dis = 8*radius_eddy/80`；radius 用 mean 或 median。 | `dx_m=mean(diff(x_vec))*mean_radius_m`，`dy_m` 同理；mean_radius 来自匹配涡旋。 | 半径均值/中位数、样本来源差异会改变所有梯度量级。 |
| 22 | 散点映射 | 三个脚本看不到；输入已经是 `Den_compound` 网格。 | Cressman：`w=(Rc^2-r^2)/(Rc^2+r^2)`，支撑数阈值，外圈 `r>4` 掩膜。 | 前辈前置 mapping 未知；我们映射可控可诊断。 |
| 23 | 合成统计 | 文件名含 `median`，说明前置可能使用 median composite。 | Cressman 是加权平均；部分截面用 median 取 `|y/R|<=0.25`。 | median 与 weighted mean 会改变异常峰值和斑块结构。 |
| 24 | 平滑对象 | 先平滑密度场，再算斜率；term2 斜率再平滑。窗口 `[8 8]`、`[6 6]`、`[4 4]`、`[2 2]` 不同。 | Cressman 后 `smooth2_supported` 平滑 `z_anom/rho_anom/rho_abs`；速度和 W 主要由映射/支撑掩膜控制。 | 前辈先密度平滑再反插，图更规整；我们保留更多局地异常。 |
| 25 | 梯度算法：密度水平梯度 | 手写中心差分：列方向为 x，行方向为 y；边界用一阶差分。 | `gradient(rho_grid, dx_m, dy_m)` 或相关栈函数。 | MATLAB `gradient` 的行列间距语义必须谨慎；若参数/输出顺序误用，会造成 x/y 对调。 |
| 26 | 梯度算法：等密面斜率 | 前辈不是 `gradient(z_rho)`，而是“左右/南北相邻整条密度柱反插同一 rho0 后差分”。 | 正式是先得到 `z_anom(x,y,z)`，再 `gradient(fillmissing2(z_grid), dx_m, dy_m)`。 | 这是算法层面最大差异；二者不等价。 |
| 27 | 前后差分形式 | 内点：`0.5*(right-left)/grid_dis`；边界：复制内侧或一阶差分。 | MATLAB `gradient` 自动中心差分和边界差分；部分 mapping 先 fillmissing。 | 边界和缺测处理不同，深层/外围差异会放大。 |
| 28 | 缺测处理 | `ndnanfilter` 处理 NaN；反插时 `none`，超范围为 NaN；边界斜率复制邻格。 | `fillmissing2` 先补洞用于求梯度，然后用 support mask 重新掩膜。 | 我们梯度可能受补洞影响；前辈受平滑和边界复制影响。 |
| 29 | 热成风密度 | 主要用 `Den_compound_smooth` 的绝对密度梯度；向下部分有一处用未平滑 `Den_compound(:,:,n-1)`。 | 用 Cressman 后的 `rho_anom` 梯度积分热成风。 | 前辈速度剪切含绝对背景斜压结构；我们只含异常斜压结构。 |
| 30 | 热成风积分锚点 | 1000 m，第 41 层，`U1000_compound/V1000_compound`。 | 1000 m 附近层，`u_base/v_base` 来自匹配历史 `I_Upk/I_Vpk` Cressman。 | 合成速度源、样本支撑和插值方法不同。 |
| 31 | 热成风积分格式 | 层间距固定 25 m，显式 Euler：向上 `+dudz*25`，向下 `+dudz*(-25)`。 | 梯形积分：`0.5*(shear_k+shear_{k+1})*dD`，上下方向符号按正深度处理。 | 我们数值积分更平滑稳定；前辈可能更强或更相位偏移。 |
| 32 | Coriolis 参数 | `f=2*7.292e-5*sind(median(E_lat))`。 | `f=2*7.2921159e-5*sind(mean(argo_lat_match))`。 | 中位/均值、样本纬度不同，量级小差异。 |
| 33 | 参考密度 | `mean(Den_compound(:),'omitnan')`。 | 固定 `rho_ref=1025`。 | 前辈按样本密度自适应；我们固定常数。 |
| 34 | 1000 m W 校验 | `W1000_compound` 被加载，但在脚本中没有直接参与重建。 | `I_Wpk` 只作为验证字段，不参与 W。 | 二者都不把观测 W 直接放入重建。 |
| 35 | 传播速度 `c` | `abs(median(moving_speed_zonal))` 或 `median(E_mspeed)`。 | `c_x_rel = mean(c_x_raw)-mean(u_bg)`，`c_x_raw` 来自 META track 相邻点。 | 前辈常取正值/绝对值；我们保留东西向相对传播速度。 |
| 36 | `term1` 符号 | `W_dzdt = c0 * dzdx`，斜率来自负深度坐标。 | `term1 = +c_x_rel * dz'_rho/dx`，W 向上为正。 | 即使公式同形，z 坐标和 c 符号不同，实际可反号。 |
| 37 | `term2` 速度参考系 | `W_is = U_thw*dzdx + V_thw*dzdy`，未显式扣 `c`。 | `term2 = -((u_tw-mean_cx_raw)*dzdx + v_tw*dzdy)`。 | 前辈用绝对热成风速度；我们用相对 x 速度并整体取负。 |
| 38 | `term2` 斜率 | 可来自 ISAS 背景密度场整体等密面斜率。 | 正式来自 BOA anomaly `z'_rho` 梯度。 | 前辈 `term2` 直接包含背景等密面坡度贡献。 |
| 39 | W 合成 | `W = W_dzdt + W_is`。 | `W = term1 + term2`，但 term2 定义和符号已不同。 | 表面同为两项相加，内部物理口径差异很大。 |
| 40 | W 符号声明 | 未在脚本/输出中明说。 | 明确 `W` 向上为正，深度变量正向下。 | 与前辈图对比必须先做符号校准。 |
| 41 | 水平坐标方向 | 行 `i` 是 y，列 `j` 是 x；手写差分清楚区分。 | `meshgrid(x_vec,y_vec)`，理论上行 y、列 x；但部分 `gradient` 包装函数需审查输出顺序。 | x/y 对调曾导致南北偶极风险，是必须单元测试的点。 |
| 42 | 背景速度扣除 | 未见 `u_bg` 或传播速度背景扣除。 | `mean_u_bg = mean(u)`，`cx_rel=mean_cx_raw-mean_u_bg`。 | 我们试图做相对运动；前辈更接近观测/合成绝对运动框架。 |
| 43 | 支撑掩膜 | 三个脚本不保存每格样本支撑。 | 保存 `count/mapped_support/valid_profile_count/boa_bg_valid_count`。 | 我们能解释空白/噪声；前辈图不易评估支撑。 |
| 44 | 输出格式 | 只存 MAT，变量较少，没有方法 Markdown。 | MAT/NC/PNG/Markdown，CSV/JSON 默认可关。 | 我们更适合汇报和复查。 |
| 45 | 可复现性 | 缺少前置生成 `Argo02/03/04` 的代码时，不能完全复现。 | 原始匹配、QC、缓存、mapping、输出均在仓库内。 | 前辈三脚本只能解释后处理，不足以审计完整流程。 |

## 对当前问题最有解释力的差异排序

1. **`z_rho` 几何定义**：前辈用整体合成密度场反插绝对等密面斜率；我们正式用 BOA 背景异常等密面 `z'_rho`。这是 A 与 B/C/D 图像差异最大的原因。
2. **密度/背景来源**：前辈 `Den_compound/Den_compound_all(ISAS)` 已经是合成场；我们逐 profile 对 BOA 多年同月局地背景做异常。这会改变深层水团和背景坡度是否保留。
3. **term2 参考系和符号**：前辈 `U_thw*dzdx + V_thw*dzdy`；我们 `-[(u_tw-cx)*dzdx + v_tw*dzdy]`。这会改变相位和幅度，但诊断显示不如 `z_rho` 几何主导。
4. **热成风速度来源**：前辈用绝对合成密度梯度；我们用异常密度梯度。这会改变深层剪切，尤其是是否出现深层反转。
5. **平滑/前置合成方式**：前辈强平滑且可能 median composite；我们 Cressman 加支撑阈值。它控制图像是否规整，但不是物理符号差异的唯一来源。
6. **梯度实现和 x/y 语义**：前辈手写中心差分；我们用 MATLAB `gradient`。当前代码必须持续用已知偶极方向做回归测试，防止行列方向误用。
