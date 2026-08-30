# 原始涡旋间断点审查图设计

本文档定义 `src.post.original_eddy_panels` 的正式后处理职责。该图属于原始 object-day 诊断，不属于代表涡旋合成，也不应放入 `Legacy` 作为主入口。

## 9-Panel Family

当前图已经扩展到超过 9 个面板，但仍称为 9-panel family：

- 1/2：`delta x(z)` 与 `delta y(z)`，显示各层速度中心相对表层中心的偏移，横轴同尺度、以 0 为中心。
- 3/4/5/6：第一、第二间断点上下层的水平速度场与地转压强代理场。
- 8/9/10/11：间断点附近的垂向剖面诊断，可选择 Omega-w 或剖面法向水平速度。
- 7：该 track 的表层中心生命周期轨迹。

## Section Geometry

设相邻层中心跳变为

\[
\Delta \mathbf r=(\Delta x,\Delta y),\quad
\mathbf e_\parallel=\Delta \mathbf r/|\Delta \mathbf r|,\quad
\mathbf e_\perp=(-\Delta y,\Delta x)/|\Delta \mathbf r|.
\]

- `jump-parallel`：剖面沿 \(\mathbf e_\parallel\)，即沿中心跳变路径。右侧 `normal_horizontal_velocity` 显示 \(u_\perp=\mathbf u_h\cdot\mathbf e_\perp\)。
- `jump-normal`：剖面沿 \(\mathbf e_\perp\)，穿过上下层中心中点。右侧显示 \(u_\parallel=\mathbf u_h\cdot\mathbf e_\parallel\)，用于检查是否有横切跳变的速度零线或剪切边界。

剖面中的零线是诊断线，不是 Hua/VG 中心定义。Hua/VG 中心仍来自速度弱核、切向性、反转性、边界速度向量单调旋转等判据。

## Entry Points

- 主入口：`python -m src.post.cli plot-original-eddy-panels`
- 直接入口：`python -m src.post.original_eddy_panels`
- 3D 几何示意：`python -m src.post.cli plot-jump-section-geometry`

旧的 `Legacy` 脚本只可作为兼容 wrapper，不再作为真实实现位置。

## Representative Eddy Variant

`src.post.representative_eddy_panels` 使用已经合成好的 ME_LIUTEX 代表涡速度场生成对应的 latest panel family。它不读取单个 object-day，也不绘制生命周期轨迹；底部 7 号面板改为 `tau-depth` 合成支持度。代表涡没有原始对象的离散间断点，因此 3/4/5/6 与 8/9/10/11 默认使用选定 tau 下代表轴线相邻深度位移最大的 J1/J2 两段：分别展示 upper/from 与 lower/to 的速度场、地转压强代理场，以及 `jump-parallel` 或 `jump-normal` 剖面的法向水平速度。
