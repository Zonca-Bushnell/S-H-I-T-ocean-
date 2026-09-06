# Core-Shell V2 / PV-Active Shell EP 验证结果报告

对应 PDF：`core_shell_partition_v2_report_zh.pdf`

## 已完成

服务器 `core_shell_partition_v2` 全生命周期结果已经完成，包含 coherent 与 upright_like，默认口径为：

`radial_seed axis + TURN + thermal_wind`

结果已下载到：

`G:\TEMP\kuroshiou_ep_core_shell_partition_v2`

正式报告整理在：

`G:\EDDY_detection\S-H-I-T-ocean-\EP-FLUX\core_shell_partition_v2`

## 核心结论

当前证据支持双区结构，而不是单一材料涡体积：

\[
\mathcal{T}_{total}
=
\mathcal{T}_{core}^{trap}
+
\mathcal{T}_{shell}^{stir}
+
\mathcal{T}_{exchange}
\]

- `inner_core`：Hua/LAVD 近同位的弱速旋转核，更接近 trapping/material coherence。
- `pv_shell`：PV anomaly、强剪切、月牙状强速带和 heat/PV stirring 更活跃。
- `exchange_layer`：解释 inner core 与 shell 之间的 heat/PV/momentum 边界交换。

## 数值摘要

| shape | region | heat covariance | PV covariance | EP tilt correction | tilt/ordinary |
|---|---:|---:|---:|---:|---:|
| coherent | inner core | 0.526 | 0.441 | 0.501 | 0.512 |
| coherent | PV shell | 0.475 | 0.559 | 0.498 | 0.382 |
| upright_like | inner core | 0.447 | 0.385 | 0.546 | 0.486 |
| upright_like | PV shell | 0.553 | 0.615 | 0.453 | 0.344 |

## 判定

PV covariance 在两类 shape 中都更偏 shell；heat covariance 在 coherent 中接近 core/shell 分担，在 upright_like 中更偏 shell。EP 倾斜修正不是只发生在 inner core，也不是可忽略小项。

因此，后续理论验证应把 `inner material core`、`PV-active stirring shell` 和 `exchange layer` 分开列账，特别是 heat/PV/momentum boundary exchange 不能混进内部 EP forcing。

完整 geodesic/LAVD object-level full 尚未完成，不能声称严格材料体 EP 闭合已经成立。
