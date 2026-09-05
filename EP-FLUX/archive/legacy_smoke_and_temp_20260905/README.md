# EP-FLUX legacy smoke and temporary outputs

这些目录是 EP-flux 工程早期 smoke、临时验证或第一版材料体诊断输出。它们保留为可追溯记录，但不再作为正式结论入口。

正式 EP 诊断入口统一为：

```bash
python -m src.EP.cli ...
```

当前归档内容：

- `full_lifecycle_validation_smoke/`：全生命周期入口早期 smoke 输出。
- `material_volume_validation/`：代表涡材料体第一版验证输出。
- `smoke_outputs_curved_metric_audit/`：曲管 metric/Jacobian/Christoffel 审计 smoke。
- `smoke_outputs_refined/`：EP smoke refined 输出。
- `smoke_outputs_streamfunction_dz/`：`streamfunction_dz` 浮力口径 smoke。
- `smoke_outputs_thermal_wind/`：`thermal_wind` 浮力口径 smoke。

仍在工作中的正式或半正式结果不归档，包括：

- `core_shell_partition_v2/`
- `core_shell_theory_report/`
- `engineering/`
