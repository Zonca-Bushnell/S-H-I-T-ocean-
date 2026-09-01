# EP 全生命周期验证阶段报告

验证结果目录：`G:\EDDY_detection\S-H-I-T-ocean-\EP-FLUX\full_lifecycle_validation`

## 核心结论

在当前 EP 诊断框架下，倾斜修正不是小量。主口径 `coherent + radial_seed + TURN + thermal_wind` 中，`|F_z_tilt_correction| / |F_z_ordinary|` 的全生命周期中位数约为 0.514，接近 ordinary vertical EP flux 的一半量级。这说明若使用倾斜涡旋坐标，热/浮力相关的垂向通量项会被明显改写。

EP-PV closure 在 smoke 到全生命周期扩展后保持稳定。主口径核心区 `divF_tilted` 与 PV proxy 的相关中位数约为 0.994，说明该框架能稳定捕捉代表涡内部 EP 散度与 PV 输送代理之间的对应关系。

曲管几何项目前不能强解释。所有组合的 `metric_valid_fraction_median = 0`，`epsilon_curvature = kappa*r` 的组合中位数大约位于 10 到 35 之间，远大于小曲率近似要求。因此 Jacobian/Christoffel 项只能作为几何风险审计，不能被写成已经闭合的物理 forcing。

严格 bootstrap / jackknife 还没有完成。当前结果明确拒绝从最终均值场伪造置信区间；需要后续建立 per-track velocity accumulator，按 track 重合成后再计算 EP。

## 组合统计

| shape | axis | orientation | buoyancy | tilt ratio | EP-PV corr | axis tilt km | metric valid | epsilon curvature | tracks |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| coherent | composite_hua_refined | turned | streamfunction_dz | 0.531 | 0.994 | 12.135 | 0.000 | 10.889 | 6-6 |
| coherent | composite_hua_refined | turned | thermal_wind | 0.484 | 0.994 | 12.135 | 0.000 | 10.889 | 6-6 |
| coherent | composite_hua_refined | unturned | streamfunction_dz | 0.264 | 0.999 | 4.543 | 0.000 | 14.233 | 6-6 |
| coherent | composite_hua_refined | unturned | thermal_wind | 0.222 | 0.999 | 4.543 | 0.000 | 14.233 | 6-6 |
| coherent | radial_seed | turned | streamfunction_dz | 0.532 | 0.994 | 13.694 | 0.000 | 10.884 | 6-6 |
| coherent | radial_seed | turned | thermal_wind | 0.514 | 0.994 | 13.694 | 0.000 | 10.884 | 6-6 |
| coherent | radial_seed | unturned | streamfunction_dz | 0.636 | 0.999 | 6.224 | 0.000 | 15.819 | 6-6 |
| coherent | radial_seed | unturned | thermal_wind | 0.617 | 0.999 | 6.224 | 0.000 | 15.819 | 6-6 |
| upright_like | composite_hua_refined | turned | streamfunction_dz | 0.569 | 0.998 | 6.408 | 0.000 | 17.801 | 5-7 |
| upright_like | composite_hua_refined | turned | thermal_wind | 0.503 | 0.998 | 6.408 | 0.000 | 17.801 | 5-7 |
| upright_like | composite_hua_refined | unturned | streamfunction_dz | 0.138 | 1.000 | 3.300 | 0.000 | 15.172 | 5-7 |
| upright_like | composite_hua_refined | unturned | thermal_wind | 0.082 | 1.000 | 3.300 | 0.000 | 15.172 | 5-7 |
| upright_like | radial_seed | turned | streamfunction_dz | 0.474 | 0.998 | 7.278 | 0.000 | 26.674 | 5-7 |
| upright_like | radial_seed | turned | thermal_wind | 0.455 | 0.998 | 7.278 | 0.000 | 26.674 | 5-7 |
| upright_like | radial_seed | unturned | streamfunction_dz | 0.526 | 1.000 | 2.716 | 0.000 | 35.267 | 5-7 |
| upright_like | radial_seed | unturned | thermal_wind | 0.497 | 1.000 | 2.716 | 0.000 | 35.267 | 5-7 |

## 指导意义

1. 后续热/浮力垂向输送不能继续只用普通垂向坐标，需要保留倾斜坐标修正项。
2. TURN 是主解释口径，UNTURN 只用于检查转向合成是否改变角结构和信号强度。
3. `radial_seed_axis` 与 `composite_hua_refined_axis` 都应保留为 axis source，默认仍使用 radial seed，显式对照 composite-Hua。
4. 曲管 EP 方向值得保留，但下一步应先完成 metric/Jacobian/Christoffel 的更严格尺度控制，而不是直接把一阶近似当成结论。
5. 若要形成统计结论，必须补齐 track-level bootstrap/jackknife。当前全生命周期结果是均值代表涡层面的阶段性验证。
