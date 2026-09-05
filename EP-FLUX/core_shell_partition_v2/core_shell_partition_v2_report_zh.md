# Core-Shell V2 / PV-Active Shell 分区判据摘要

本文档对应 `core_shell_partition_v2_report_zh.pdf`。

## 核心结论

V2 不再要求单一材料体同时解释 trapping、PV anomaly、heat/PV stirring 和 EP forcing。推荐分区为：

\[
\mathcal{T}_{total}
=
\mathcal{T}_{core}^{trap}
+
\mathcal{T}_{shell}^{stir}
+
\mathcal{T}_{exchange}
\]

## 三个区域

- `inner_material_core`：Hua/LAVD 近同位、低 leakage、高 retention、弱速核心连通，默认 `r/R <= 1.2`。
- `pv_active_shell`：inner core 外侧，高 `|q'|`、高 `|grad q'|`、强剪切或月牙强速带，默认到 `r/R <= 2.5`。
- `exchange_layer`：inner core 与 shell 接触带，用于 heat/PV/momentum boundary exchange。

## 和 v1 的区别

- v1 的 shell 仍偏内，默认 outer radius 是 `1.5R`。
- v2 将 inner core 收紧到 `1.2R`，shell 放宽到 `2.5R`。
- v2 把 exchange layer 作为独立物理解释层，而不是只看 combined volume 是否闭合。

## 判读原则

- inner core 低 leakage 只说明 trapping/material coherence。
- PV shell 协方差强说明 stirring 主要发生在 shell。
- exchange 不小则说明闭合残差来自 core-shell 交换，不能简单说 EP 公式错误。

## 代码入口

```bash
python -m src.EP.cli run-core-shell-v2-validation \
  --shapes coherent,upright_like \
  --orientations turned \
  --axis-sources radial_seed \
  --buoyancy-sources thermal_wind \
  --tau-values 0.50
```

默认 V2 参数已经设置为：

- `--core-radius-over-R 1.2`
- `--shell-outer-radius-over-R 2.5`
- `--pv-shell-quantile 0.80`
- `--inner-boundary-mode levelset_v2`
- `--boundary-budget full_3d`
