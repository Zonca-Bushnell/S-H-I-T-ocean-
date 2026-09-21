# OFES 直接等密面合成

该入口 `Detection_for_OFES.tools.run_ofes_isopycnal_composite` 是 OFES 的 geometry-first 合成路径。

1. 对每个识别涡旋，按涡旋半径投影到 `x/R,y/R` 网格，并从 OFES 原场采样 `prho` 与原生 `w`。
2. 每条对象场先做 Cressman 映射，再在同一半球和极性组内平均；不同滤波核绝不混合。
3. 从合成密度场涡心柱取得每个名义深度的 `rho0`。在每一个合成密度柱中直接搜索包夹该 `rho0` 的相邻深度层，反插 `D_rho`。
4. 对多重 crossing，选最接近名义深度的有效 bracket。`D_rho-D0`、`dD_rho/dx` 与 `dD_rho/dy` 只从这个直接反插的等密面场得到。

## 几何 QC 与展示范围

科学展示默认只采用 `0–2000 m` 的直接反插结果，深海仍保留在主 NPZ 中供技术审计，但不纳入涡旋结构结论。每个候选反插格点必须同时满足：

- 相邻密度层对目标密度存在有效 bracket；
- 两个 bracket 层的 Cressman 对象支撑均不少于 `8`；
- 所选 bracket 的局地层结 `|delta rho / delta D| >= 1e-4 kg m-4`。

输出中的 `d_rho_qc_m` 与 `d_rho_anom_qc_m` 是该严格掩膜后的几何量；`bracket_crossing_count`、`bracket_stratification_kg_m4` 和 `isopycnal_qc_by_depth.csv` 给出每格/每层的反插可追溯性。截面图采用填色加等值线，便于判断上拱或下凹结构。

此主机的 Python Matplotlib 等值线后端会异常退出，因此正式 QC 截面图由 MATLAB `plot_isopycnal_qc_section.m` 渲染；它只读取 Python 已保存的 QC 等密面场，不参与密度合成、反插或 QC 判定。

## 固定深度密度异常

除等密面位移外，输出同时提供固定深度密度异常：

```text
rho_prime(x,y,D) = rho_comp(x,y,D) - median[rho_comp(2R <= r/R <= 4R,D)]
```

`rho_anom_ring_qc_kg_m3` 使用相同的 `0–2000 m` 和 Cressman 对象支撑限制，图件为 `density_anomaly_section_x_qc.png`。它与等密面位移不是同一物理量：在稳定层结中，气旋等密面上拱通常对应固定深度的正密度异常，反气旋下凹通常对应负密度异常。

本入口禁止使用 `eta_rho=-rho'/(d rho_bg/dD)`、`rho_prime`、`drho_dz` 或任何以密度导数反推等密面位移的路径。原生 `W_native` 仅作为 OFES 模型输出的合成对照；单日 Jan 1 没有可靠传播速度，所以不输出 term1、term2 或 rebuild W。

坐标约定：`x/R` 为东西向，`y/R` 为南北向，深度 `D` 正向下，原生 OFES `W` 保留其向上为正的物理符号。
