# Curved-Tube Scale Audit Summary

## 审计口径
- Curved-tube mode：`jacobian_christoffel`。
- 大曲率阈值：`kappa*r <= 1` 才视为一阶 metric 有效。
- 主物理解释仍以 `divF_tilted` 为基准；本文件只评估 Jacobian 与 Christoffel 项的尺度。

## Polarity 汇总
```text
    polarity   rows  core_rows  finite_divF_tilted_fraction  finite_pv_flux_fraction  median_axis_tilt_km  median_abs_tilt_correction_over_ordinary  divF_pv_flux_corr_core  median_abs_curved_minus_tilted_over_tilted  median_abs_curved_total_minus_tilted_over_tilted  median_abs_jacobian_correction_over_tilted  median_abs_christoffel_over_tilted  median_abs_scale_upper_bound_over_tilted  median_epsilon_tilt  median_epsilon_curvature  p90_epsilon_curvature  metric_valid_fraction_core  jacobian_min_core  jacobian_max_core
anticyclonic 2160.0     1296.0                          1.0                      1.0            15.014883                                  0.589473                0.999008                                         0.0                                          0.754708                                1.541681e-15                            0.754708                                  0.362219            42.552519                 18.571482             110.995080                    0.077160        -527.618787         529.618787
    cyclonic 2160.0     1296.0                          1.0                      1.0            10.583885                                  0.430713                0.998154                                         0.0                                          0.177701                                1.448211e-15                            0.177701                                  0.092873            42.691721                 10.927002              69.725251                    0.097222        -555.865130         557.865130
```

## 几何尺度汇总
```text
    polarity  metric_valid_fraction  epsilon_curvature_median  epsilon_curvature_p90  jacobian_min  jacobian_max
anticyclonic               0.000000                 27.036274              47.764083   -527.618787    529.618787
    cyclonic               0.041667                 14.884612              26.296148   -555.865130    557.865130
```

## 判读规则
- `epsilon_curvature = kappa*r` 若接近或大于 1，说明一阶曲管展开不能作为强结论。
- `jacobian_min <= 0` 表示一阶坐标映射局部失效，该层/半径只保留为风险提示。
- `divF_christoffel_qg_approx` 只是一阶联络项近似；若其量级很大，下一步应回到完整截面张量。
