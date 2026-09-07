# Argo-Eddy W 合成方法文献核查

## 本地有效 PDF

PDF 目录：`D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\PDF\Dipole_vertical _transport`

- `JGR Oceans - 2026 - Zhang - Dipole Structure of Vertical Velocity Induced by Mesoscale Eddies.pdf`
- `Hou_etal_2022_Frontiers_Eddy_beta_spiral.pdf`
- `Wang_etal_2020_JGR_Eddy_Induced_Acceleration_of_Argo_Floats.pdf`
- `CortesMorales_etal_2026_ESSD_Global_Thermocline_Vertical_Velocities.pdf`
- `Roemmich_etal_2020_Frontiers_Argo_Data_1999_2019.pdf`
- `Gaube_etal_2019_Frontiers_Mesoscale_Eddy_Impacts_Review.pdf`
- `Liu_etal_2013_JOUC_Eddies_Argo_STMW.pdf`

## 已找到但未计入本地依据

以下论文很相关，但当前自动下载只拿到了 Cloudflare/站点拦截 HTML，不是有效 PDF，
因此不作为“已下载论文”引用。后续若手动保存到上述目录并通过 PDF 解析，再纳入依据。

- Christensen et al. (2024), *Global Estimates of Mesoscale Vertical Velocity Near 1,000 m From Argo Observations*, JGR Oceans, DOI: `10.1029/2023JC020003`。网页全文可读并已核对方法，但本地 PDF 下载仍失败。
- Chaigneau et al. (2011), *Vertical structure of mesoscale eddies in the eastern South Pacific Ocean: A composite analysis from altimetry and Argo profiling floats*, DOI: `10.1029/2011JC007134`。
- Lin et al. (2019), *Thermohaline Structures and Heat/Freshwater Transports of Mesoscale Eddies in the Bay of Bengal Observed by Argo and Satellite Data*, DOI: `10.3390/rs11242989`。
- Mason et al. (2017), *Subregional characterization of mesoscale eddies across the Brazil-Malvinas Confluence*, DOI: `10.1002/2016JC012611`。
- Qu et al. (2022), *Spatial Structure of Vertical Motions and Associated Heat Flux Induced by Mesoscale Eddies in the Upper Kuroshio-Oyashio Extension*, DOI: `10.1029/2022JC018781`。
- Freeland (2013), *Vertical velocity estimates in the North Pacific using Argo floats*, DOI: `10.1016/j.dsr2.2012.07.019`。
- Colin de Verdiere and Ollitrault (2016), JPO Argo displacement / circulation method, DOI: `10.1175/JPO-D-15-0046.1`。
- Colin de Verdiere et al. (2019), JGR Oceans Argo-derived geostrophic/LVB vertical velocity method, DOI: `10.1029/2018JC014565`。

## 对我们当前异常图像的判断

当前结果不像偶极子，主要不是 Argo 年份覆盖太短，而是正式 `z_rho` 重建 W 链条还没有和
Argo 垂直速度文献的观测量口径对齐。已确认的风险包括：

1. 匹配记录错误：早期代码在同一条 Argo 有多个候选涡旋时，把候选涡旋的 `dx/dy`
   向量写进了记录，而不是只写最佳候选 `dx(best_pos)/dy(best_pos)`。这会污染
   `x/R, y/R`，让 W 图像像稀疏点图或数量图。
2. `rho0` 口径错误风险：若每条 profile 用自身 parking-depth density 当目标密度，
   反插值出的 `z_rho` 会退化到自身 parking depth 附近，`dz_rho/dx` 会被压得接近
   0，不可能形成稳定偶极子。
3. 速度来源风险：当前正式脚本用相邻 profile 位置差估计 `u_Argo/v_Argo`，这不是严格
   parking-depth displacement velocity，会影响 `u_bg`、`c_x_rel` 和 `term2`。
4. `z_rho` 应优先转为等密面位移异常后再求梯度；当前对绝对 `z_rho` 求梯度会混入背景
   水团/纬向密度坡度。
5. 移动坐标符号和速度参考系没有被 `I_Wpk` 或文献公式锁定，`+c dzdx`、`-c dzdx`、
   `u_abs`、`u_rel` 都需要敏感性测试。
6. 网格化方式不稳：单纯 bin median 会只显示离散采样格点；无约束 scattered 插值
   又可能在样本很稀疏处补出假连续结构。文献里更常见的是在统一网格上做有半径和
   权重约束的 objective mapping / composite。

## Cressman 插值细节

Wang et al. (2020) 在 Argo 轨迹速度映射中使用 Cressman interpolation：

```text
z_hat(x0,y0) = sum(lambda_i * z_i)
lambda_i = omega_i / sum(omega_i)
omega_i = (R^2 - r_i^2) / (R^2 + r_i^2)
```

只使用插值半径 `R` 内的样本。该文在物理坐标中使用 `R = 30 km`，约为
`1/4 degree`，并指出真实 Argo 轨迹速度在细网格上非常稀疏，因此需要谨慎对待
样本支撑。

我们在涡旋归一化坐标 `x/R_eddy, y/R_eddy` 上采用同一思想，而不是照搬 30 km：
默认 `--cressman-radius-r 0.5`，即每个目标格点使用半径 `0.5R_eddy` 内样本；
默认 `--cressman-min-obs 3`，少于 3 条样本的格点不显示 W。这个半径是第一版
观测合成的保守参数，后续需要做敏感性测试：`0.3R, 0.5R, 0.75R`。

## 推荐修改方向

- 不再把 `rho0-mode = band_median` 当作最终推荐口径；它只能作为临时稳定共同密度面的
  smoke-test 方案。下一步应建立 `z_rho_anomaly = z_rho - z_rho_bg`，再对异常场求梯度。
- 优先把 `Argo1000m_UVW_TSDen_199601_202306.mat` 的 `I_Upk/I_Vpk/I_Wpk` 与 TEOS-10
  密度剖面匹配：`I_Wpk` 用作观测 W sanity check，`I_Upk/I_Vpk` 用作真实 parking drift
  的第一候选。
- 重写 `z_rho` 反插值质量控制：只使用明确 bracket crossing，记录 crossing 个数、bracket
  厚度和局地层结，剔除多重异常 crossing 与弱层结样本。
- 对 `term1` 保留符号敏感性测试，用 `I_Wpk` 偶极方向校验 `+c dzdx` 还是 `-c dzdx`。
- 对 `term2` 保留速度参考系敏感性测试：`u_abs`、`u_abs-c`、`u_anom` 分开输出，不默认
  直接合成为最终 W。
- 默认 `grid-mapping = cressman`，并输出 `sample_count` 和 `mapped_support`。
  `sample_count` 是原始 bin 中样本数，`mapped_support` 是 Cressman 半径内参与映射的
  样本数；正式 W 图只显示 `mapped_support >= cressman_min_obs` 的格点。
- 多候选涡旋归属保持“最近 r/R”默认：每条 Argo profile 会搜索同纬度带、同时间窗
  内所有 META 涡旋；若同时落入多个 4R 半径，只归属给 `r/R` 最小的涡旋，避免重复计数。
- 暂停全球全样本正式跑，先用 1-2 个纬度带做参数敏感性图，确认出现合理偶极子和量级后
  再扩大到全球。

## 与 Zhang/Hou 偶极子预期的关系

Zhang et al. (2026) 和 Hou et al. (2022) 支持“涡旋诱导 W 常呈偶极子”的科学预期，
但它们不是直接用 sparse Argo parking drift 重建 W 的同一算法。它们给我们的是
形态检查和动力解释：若我们用正确的共同等密面、正确的涡心归一化、足够样本支撑的
objective mapping，合成图应当逐渐接近东西向偶极子；若仍然只有零散点，则优先检查
样本支撑、纬度带样本数、半径归一化和 `rho0/z_rho` 口径，而不是简单归因于年份不足。
