# 坐标与符号统一约定

## 唯一正式口径

正式主流程内部只使用 `D` 坐标：

```text
D > 0：正深度向下
D_rho：等密面深度
D'_rho = D_rho - D_rho_bg：等密面深度异常
W_up > 0：向上速度为正
W_up = -D_t
```

历史输出字段名中仍可能保留 `z_rho_m`、`z_rho_bg_m`、`z_rho_anom_m`，这是为了兼容已有结果读取脚本。它们在物理含义上全部按 `D_rho`、`D_rho_bg`、`D'_rho` 解释，单位为 m，正方向向下。

## 正式 W 公式

若 `D'_rho(x,y)` 是正深度向下的等密面异常，涡旋相对背景的纬向传播速度为：

```text
c_x_rel = mean(c_x_raw) - mean(u_bg)
```

水平速度相对背景为：

```text
u_rel = u - u_bg
```

则正式 upward-positive W 为：

```text
term1_up = + c_x_rel * dD'_rho/dx
term2_up = - (u_rel * dD'_rho/dx + v * dD'_rho/dy)
W_up     = term1_up + term2_up
```

深度向下为正的对照量仅用于审计：

```text
W_depth_positive = -W_up
```

## 禁止混用

正式主流程不再直接使用 `z=-D`。如果需要复现前辈程序，必须在 diagnostics/legacy 入口中显式转换：

```text
z_up = -D
dz_up/dx = -dD/dx
```

这类转换只服务于对照图，不允许进入正式生产结果。

## 坐标轴

矩阵列方向是 `x/R`，即局地东西向；矩阵行方向是 `y/R`，即局地南北向。所有水平梯度必须经过 `gradient_xy` 或调用 `w_terms_from_depth_geometry` 的上游统一函数，避免 MATLAB `gradient` 的行列顺序再次引入 x/y 翻转。
