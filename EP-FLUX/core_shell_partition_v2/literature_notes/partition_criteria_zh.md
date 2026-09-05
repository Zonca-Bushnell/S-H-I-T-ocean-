# Core-Shell V2 分区判据文献笔记

## 1. Inner material core

Haller & Beron-Vera (2013) 的 geodesic eddy 理论把材料涡边界定义为有限时间内低拉伸、低泄漏的闭合材料曲线。Haller et al. (2016) 的 LAVD 理论进一步给出旋转相干涡核的客观定义。两者共同说明：材料核应优先服务于 trapping / material coherence，而不是被迫包含所有 PV anomaly。

V2 判据：Hua/LAVD 近同位、低速度核心连通、低 boundary leakage、高 particle/weak-core retention。默认内核半径收紧到 `r/R <= 1.2`。

## 2. PV-active shell

Abernathey & Haller (2018) 指出，Eulerian 涡旋与 Lagrangian coherent vortex 并不等价，许多输送来自 filamentation 和外侧交换。Zhang, Wolfe & Abernathey (2020) 的 PV 输送研究也支持 coherent core 与 periphery/shell 分账。

V2 判据：inner core 外侧、高 `|q'|`、高 `|grad q'|`、强速度剪切或月牙强速带。默认外半径扩展到 `r/R <= 2.5`，`|q'|` 默认取区域内前 20%。

## 3. Exchange layer

如果提高 PV retention 会显著增加 leakage 或 boundary exchange，就不应继续要求 `PV core subset LAVD core`。更合理的解释是：运动学材料核与动力 PV shell 分离，中间存在 heat/PV/momentum exchange layer。

V2 判据：inner core 边界外侧 1-2 个网格，或 inner core 与 PV shell 的接触带。这里不作为严格 trapping 区，而是边界交换预算区。
