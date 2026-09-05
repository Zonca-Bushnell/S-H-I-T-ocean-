# Material-Volume EP Validation Summary

## 口径
- shape: `coherent`
- axis source: `radial_seed`
- orientation: `turned`
- buoyancy source: `thermal_wind`
- object: representative coherent material volume in Cartesian coordinates

## 验证版定义
- core radius: `r/R <= 1.5`
- speed core quantile: `0.45`
- abs(PV proxy) core quantile: `0.7`
- mask: low-speed core OR high-|PV proxy| core, keeping the component connected to the axis.

## 结果摘要
```text
    polarity  n_tau  n_depth  finite_G_fraction  mask_fraction_median  boundary_leakage_median_ms  pv_centroid_offset_median_km  weak_centroid_offset_median_km  G_magnitude_median  pv_flux_magnitude_median
anticyclonic     21       54                1.0              0.330556                    0.017996                     33.382364                       26.350232        1.746227e-08              7.608669e-07
    cyclonic     21       54                1.0              0.333160                    0.021215                     30.668326                       23.030888        1.673307e-08              6.737221e-07
```

## 判读
- 这是大曲率 Material-Volume EP 的代表涡验证版，不是最终 object-level material-boundary 闭合。
- `R_ij/B_i/P_i` 在 Cartesian 坐标中定义，避免依赖已经失效的小曲率曲管 Jacobian。
- `G_x_proxy/G_y_proxy` 目前是动量通量水平散度代理，尚未包含完整 QG/PV 反演算子 `T_ij[B_j]`。
- `boundary_leakage_proxy` 越大，越说明当前 mask 不是严格材料边界，不能把体内 forcing 单独解释为全部动力。
