# EP Full Lifecycle Validation Summary

## 口径
- shape: `coherent`
- axis source: `radial_seed`
- orientation: `turned`
- buoyancy source: `thermal_wind`
- curved-tube mode: `scale_audit`

## 结果表
```text
   shape axis_source orientation buoyancy_source     polarity  tau  rows  core_rows  n_objects  n_tracks  finite_divF_tilted_fraction  finite_pv_flux_fraction  median_axis_tilt_km  median_abs_tilt_correction_over_ordinary  divF_pv_flux_corr_core  divF_pv_flux_rmse_core  metric_valid_fraction_median  epsilon_curvature_median  jacobian_min_median  jacobian_max_median
coherent radial_seed      turned    thermal_wind anticyclonic 0.45  2160       1296        112         6                          1.0                      1.0            16.292393                                  0.606319                0.999507                0.014995                           0.0                 16.491781           -15.485720            17.485720
coherent radial_seed      turned    thermal_wind anticyclonic 0.50  2160       1296        111         6                          1.0                      1.0            15.014883                                  0.589473                0.999008                0.012171                           0.0                 18.571482           -17.567741            19.567741
coherent radial_seed      turned    thermal_wind anticyclonic 0.55  2160       1296        112         6                          1.0                      1.0            14.336013                                  0.540495                0.995301                0.009533                           0.0                 19.001957           -17.994644            19.994644
coherent radial_seed      turned    thermal_wind     cyclonic 0.45  2160       1296        120         6                          1.0                      1.0            10.858195                                  0.516095                0.999618                0.044864                           0.0                  9.074963            -8.072751            10.072751
coherent radial_seed      turned    thermal_wind     cyclonic 0.50  2160       1296        120         6                          1.0                      1.0            10.583885                                  0.430713                0.998154                0.034242                           0.0                 10.927002            -9.925439            11.925439
coherent radial_seed      turned    thermal_wind     cyclonic 0.55  2160       1296        119         6                          1.0                      1.0            10.664752                                  0.445959                0.989825                0.022072                           0.0                 10.298774            -9.295341            11.295341
```

## Bootstrap / Jackknife
- 未计算严格 bootstrap/jackknife：missing track-level velocity accumulator; strict track bootstrap cannot be computed from the final representative mean field alone
- 不能把最终代表涡均值场重新抽样当作严格置信区间。

## 判读
- 主结论仍优先看 `turned + radial_seed + thermal_wind`。
- `F_z_tilt_correction / F_z_ordinary` 若在多个 tau 保持同量级，说明倾斜修正不是 tau=0.50 的偶然现象。
- `divF_tilted` 与 `pv_flux_proxy` 的相关和误差用于检查 EP-PV closure。
- `epsilon_curvature` 大或 `metric_valid_fraction` 低时，只能说明曲管项需要更完整理论闭合。
