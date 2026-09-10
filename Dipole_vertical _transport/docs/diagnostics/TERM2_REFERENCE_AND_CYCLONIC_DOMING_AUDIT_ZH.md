# Term2 参考系与 Cyclonic 等密面上隆审计

## 结论

本轮审计确认，旧正式公式中的 `term2 = -[(u-c_x_raw) dz'_rho/dx + v dz'_rho/dy]` 存在参考系重复计入问题。因为正式 `term1` 已经使用 `c_x_rel = mean(c_x_raw)-mean(u_bg)`，若 `term2` 再减一次 `c_x_raw`，总式会变成：

```text
W = (c_x_raw-u_bg) dz'_rho/dx - (u-c_x_raw) dz'_rho/dx - v dz'_rho/dy
  = (2 c_x_raw-u_bg-u) dz'_rho/dx - v dz'_rho/dy
```

因此传播速度被重复计入一次。已将正式与诊断中的当前口径修正为：

```text
term1 = (mean(c_x_raw)-mean(u_bg)) dz'_rho/dx
term2 = -[(u-u_bg) dz'_rho/dx + v dz'_rho/dy]
W     = term1 + term2
```

## 20N 2D recommended 全样本复算

用既有 20N crossing / repeat-all / recommended 全样本网格，在不重新匹配的情况下复算修正后 `term2`，结果如下，单位为 `10^-6 m/s`。

| polarity | field | west | east | east-west | q95 |
|---|---:|---:|---:|---:|---:|
| cyclonic | term1 | 5.066 | -5.697 | -10.762 | 4.305 |
| cyclonic | term2 修正后 | -0.018 | -0.200 | -0.182 | 0.246 |
| cyclonic | W 修正后 | 5.061 | -5.905 | -10.966 | 4.399 |
| cyclonic | W 旧公式 | 10.143 | -11.602 | -21.745 | 8.592 |
| cyclonic | I_Wpk | 7.655 | -7.158 | -14.814 | 5.939 |
| anticyclonic | term1 | -4.905 | 4.663 | 9.568 | 3.742 |
| anticyclonic | term2 修正后 | -0.156 | 0.015 | 0.171 | 0.222 |
| anticyclonic | W 修正后 | -5.115 | 4.538 | 9.653 | 3.771 |
| anticyclonic | W 旧公式 | -9.986 | 9.221 | 19.206 | 7.503 |
| anticyclonic | I_Wpk | -6.471 | 6.679 | 13.150 | 5.234 |

修正后 `term2` 不再近似复制 `term1`，W 量级约减半；但 cyclonic 的东西向偶极方向仍由 `term1` 主导。

## Cyclonic 上隆是否反常

常规北半球气旋涡一般对应 SSH 低、冷核、等密面上隆；反气旋涡一般对应 SSH 高、暖核、等密面下隆。因此 cyclonic 在 1000 m 的 BOA anomaly 等密面上隆本身不反常。

20N 2D recommended 的等密面深度诊断为：

| polarity | field | west | center | east |
|---|---:|---:|---:|---:|
| cyclonic | z_rho | 987.684 m | 979.632 m | 989.200 m |
| cyclonic | z_bg | 1000.677 m | 1001.233 m | 1000.457 m |
| cyclonic | z'_rho | -12.885 m | -21.194 m | -11.298 m |
| anticyclonic | z_rho | 996.619 m | 1007.146 m | 999.843 m |
| anticyclonic | z_bg | 997.947 m | 998.417 m | 999.074 m |
| anticyclonic | z'_rho | -2.200 m | 8.341 m | 0.125 m |

其中深度为正向下，负的 `z'_rho` 表示等密面相对 BOA 背景上隆。

## 与深层反转相关的主要差异

目前看，深层是否出现反转最相关的不是 x/y 坐标，而是 `z_rho` 几何定义：

1. 当前正式口径使用 `BOA monthly climatology -> z'_rho anomaly -> Cressman -> gradient`，会扣掉大尺度和背景水团等密面坡度。
2. 前辈程序使用 `composite density absolute field -> isopycnal slope`，并在 term2 中使用背景/合成密度场的斜率，深层大尺度等密面几何仍会进入 W。
3. 三分诊断显示，absolute composite density 与 ISAS-derived background term2 会显著增强深层 term2，并更容易给出深层结构变化；BOA anomaly 则主要保留涡旋异常几何。

因此，如果目标是复现前辈式深层反转，下一步应继续锁定 `BOA anomaly z'_rho` 是否过度移除了深层背景几何，而不是先怀疑 cyclonic 上隆本身或 x/y 坐标旋转。
