# EP Flux Smoke Summary

## 口径
- 代表涡：`coherent-only` / `turned` / `tau=0.50`
- Axis source：`radial_seed`
- Buoyancy source：`thermal_wind`
- Curved-tube mode：`jacobian_christoffel`
- Classic EP、tilted EP 与 curved-tube 几何审计项同时输出。
- 本报告是 smoke 验证，不是多年全量结论。

## 关键数值检查
```text
    polarity   rows  core_rows  finite_divF_tilted_fraction  finite_pv_flux_fraction  median_axis_tilt_km  median_abs_tilt_correction_over_ordinary  divF_pv_flux_corr_core  median_abs_curved_minus_tilted_over_tilted  median_abs_curved_total_minus_tilted_over_tilted  median_abs_jacobian_correction_over_tilted  median_abs_christoffel_over_tilted  median_abs_scale_upper_bound_over_tilted  median_epsilon_tilt  median_epsilon_curvature  p90_epsilon_curvature  metric_valid_fraction_core  jacobian_min_core  jacobian_max_core
anticyclonic 2160.0     1296.0                          1.0                      1.0            15.014883                                  0.589473                0.999008                                         0.0                                          0.754708                                1.541681e-15                            0.754708                                  0.362219            42.552519                 18.571482             110.995080                    0.077160        -527.618787         529.618787
    cyclonic 2160.0     1296.0                          1.0                      1.0            10.583885                                  0.430713                0.998154                                         0.0                                          0.177701                                1.448211e-15                            0.177701                                  0.092873            42.691721                 10.927002              69.725251                    0.097222        -555.865130         557.865130
```

## 解释
- `F_z_tilt_correction` 衡量中心轴倾斜导致的垂向导数修正。
- `thermal_wind` 口径用合成地转速度垂向切变反推浮力异常梯度，再恢复截面内的 \(b'\)。
- `divF_jacobian` 是带一阶 Jacobian 的通量散度。
- `divF_christoffel_qg_approx` 是 Bishop-frame 下的一阶 QG 几何联络近似。
- `divF_scale_upper_bound` 是 \(\kappa F_n\) 量级上界，只用于提示曲率可能重要。
- `streamfunction_dz` 口径保留 \(b'=f_0\partial_z\psi\) 的 QG 对照。
- `pv_flux_proxy` 来自代表流函数的 QG-like PV 代理，用于闭合关系的回归检查。

## 注意
- 第一版 curved-tube 中的张量几何和连接项仍是 QG 近似框架，完整 \(T^{ia}\) 需要后续理论实现。
- 若 sensitivity 远大于 resolved divergence，应解释为“曲率项需要完整二维截面张量闭合”，而不是直接判定 curved-tube forcing 极大。
