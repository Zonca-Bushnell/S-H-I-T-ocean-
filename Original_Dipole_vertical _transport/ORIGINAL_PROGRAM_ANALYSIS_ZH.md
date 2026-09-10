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

