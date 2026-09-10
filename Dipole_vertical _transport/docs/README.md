# Documentation Map

本目录收纳 `Dipole_vertical _transport` 的方法、诊断和文献记录。

- `method/`：正式生产口径。当前固定为 crossing + BOA anomaly geometry +
  Cressman + thermal-wind W。
- `diagnostics/`：用于解释和对照的诊断入口，包括参数敏感性、深层反转因子、
  zgeometry 对照和性能审计。这里的模式不改变正式生产定义。
- `literature/`：已下载并核验过的文献方法记录。
- `audit/`：Argo 数据源审计记录。

正式结果不生成 combined；W 向上为正，深度变量和图像深度坐标按正深度向下显示。
水平梯度统一通过 `matlab/physics/gradient_xy.m` 计算：矩阵列方向为 `x/R`
（东西向），矩阵行方向为 `y/R`（南北向），返回值固定为
`[dfdx, dfdy]`，避免 MATLAB `gradient` 调用顺序造成东西/南北方向误判。
