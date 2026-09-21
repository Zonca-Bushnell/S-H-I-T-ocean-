# OFES 直接等密面合成

该入口 `Detection_for_OFES.tools.run_ofes_isopycnal_composite` 是 OFES 的 geometry-first 合成路径。

1. 对每个识别涡旋，按涡旋半径投影到 `x/R,y/R` 网格，并从 OFES 原场采样 `prho` 与原生 `w`。
2. 每条对象场先做 Cressman 映射，再在同一半球和极性组内平均；不同滤波核绝不混合。
3. 从合成密度场涡心柱取得每个名义深度的 `rho0`。在每一个合成密度柱中直接搜索包夹该 `rho0` 的相邻深度层，反插 `D_rho`。
4. 对多重 crossing，选最接近名义深度的有效 bracket。`D_rho-D0`、`dD_rho/dx` 与 `dD_rho/dy` 只从这个直接反插的等密面场得到。

本入口禁止使用 `eta_rho=-rho'/(d rho_bg/dD)`、`rho_prime`、`drho_dz` 或任何以密度导数反推等密面位移的路径。原生 `W_native` 仅作为 OFES 模型输出的合成对照；单日 Jan 1 没有可靠传播速度，所以不输出 term1、term2 或 rebuild W。

坐标约定：`x/R` 为东西向，`y/R` 为南北向，深度 `D` 正向下，原生 OFES `W` 保留其向上为正的物理符号。
