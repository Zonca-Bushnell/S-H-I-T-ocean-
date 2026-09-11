# ACC 涡旋识别中 `boundary_monotonic_rotation`、`tangent_alignment` 与 `symmetry` 的文献依据和修改建议

日期：2026-09-11  
适用代码：`Origin_eddy_detection` / Hua-style hybrid eddy detection  
适用数据：Global Ocean Ensemble Physics Reanalysis ACC, 2018-2019

## 1. 当前问题

ACC 当前检测结果显示，表层 surface seed 共有 1,122,571 个，但表层 Hua 通过只有 57,882 个，表层保留率约 5.16%。表层失败原因主要集中在：

| 失败原因 | 数量 | 占全部 surface seed | 占 surface 失败 |
|---|---:|---:|---:|
| `boundary_monotonic_rotation` | 580,571 | 51.72% | 54.53% |
| `tangent_alignment` | 273,847 | 24.40% | 25.72% |
| `symmetry` | 104,309 | 9.29% | 9.80% |

这说明当前数量少的主因不是候选点不足，而是后续圆周几何检验很严格。特别是 ACC 区域存在强背景流、强锋面、拉伸涡和非圆边界时，固定圆周检验容易把真实但变形明显的中尺度涡旋筛掉。

## 2. 当前三个筛选项的机制

### 2.1 `boundary_monotonic_rotation`

当前实现沿候选中心的圆周采样速度向量：

```text
theta_n = atan2(v_n, u_n)
dtheta_n = angle_diff(theta_n, theta_{n+1})
```

然后统计一圈中 `dtheta` 的正负号：

```text
direction_exceptions = min(count(dtheta > 0), count(dtheta < 0))
```

当前 ACC 参数为：

```text
require_boundary_monotonic_rotation = True
boundary_monotonic_exception_limit = 0
```

因此只要圆周上出现一个局部反向转折，就会失败。这个约束的物理意图是保证速度方向沿边界连续绕转，排除喷流、锋面、开口流线和波动。但是它隐含了强圆形边界假设：被测试的圆周必须刚好落在真实涡旋边界附近，并且真实边界没有明显内凹、外凸、椭圆化或局部变形。

### 2.2 `tangent_alignment`

当前实现计算速度与圆周切向方向的夹角。圆周点的理论切向方向为：

```text
t = (-sin(theta), cos(theta))
tangent_cos = |u t_x + v t_y| / sqrt(u^2 + v^2)
```

当前 ACC 参数为：

```text
tangent_tolerance_deg = 24
min_tangent_fraction = 0.70
```

也就是说，圆周有效点中至少 70% 的速度方向必须落在理想圆切向的 +/-24 度范围内。这个参数适合较干净、近圆形、单核涡旋，但对 ACC 中的拉伸涡、椭圆涡、锋面调制涡和地形约束涡偏硬。

### 2.3 `symmetry`

当前实现把圆周上相隔半圈的对径点配对，比较二者速度方向是否接近相反：

```text
diff = |angle_diff(theta_n, theta_{n+half})|
symmetry_ok: |diff - pi| <= symmetry_tolerance
```

这个约束的意图是避免把单侧喷流、锋面剪切或非闭合旋转结构误判为涡旋。但是在真实海洋中，拉伸、倾斜、多核、强背景剪切都会破坏严格的对径反向关系。因此 `symmetry` 更适合作为质量评分或形态诊断，而不宜单独解释为“不是涡旋”。

## 3. 文献中如何处理非圆边界

### 3.1 Velocity-geometry 方法不强制固定圆边界

Nencioli et al. (2010) 的 vector geometry eddy detection 方法使用速度向量的局地几何关系识别涡心。它强调的是中心附近速度分量符号变化、速度局地最小和旋转方向一致性，而不是在固定半径圆周上要求速度方向严格单调绕转。该方法的核心启示是：速度几何可以用于找中心，但不必把边界固定为理想圆。

对当前 ACC 问题的意义：可以保留速度弱中心和旋转一致性思想，但 `boundary_monotonic_rotation` 不应要求固定圆周零例外。

### 3.2 SSH/SLA 闭合等值线方法允许边界变形

Chelton et al. (2011) 以及 Mason et al. (2014) 的 SSH/SLA 方法使用闭合海面高度异常等值线定义涡旋边界，并配合振幅、半径、单极值和形状误差等标准进行质量控制。边界不是预设圆，而是从闭合轮廓中得到，因此天然允许椭圆、拉伸和局部不规则。

Mason et al. (2014) 的实现方向后来影响了 `py-eddy-tracker` 一类工具。这类方法更接近“闭合轮廓 + 几何/物理质量控制”，而不是“固定圆周上速度全程满足切向和单调”。

对当前 ACC 问题的意义：应考虑把固定圆周检验替换为闭合轮廓或椭圆轮廓检验；至少应把固定圆周作为候选诊断，而不是唯一边界。

### 3.3 AMEDA/LNAM 方法用角动量和闭合轮廓处理变形涡旋

AMEDA 类方法使用局地归一化角动量等指标，并结合闭合轮廓来检测和追踪涡旋。它的核心优点是对变形边界和非圆形涡旋更友好，因为它不要求所有边界点落在某个圆上。

对当前 ACC 问题的意义：可以引入局地角动量、涡度或 Okubo-Weiss 一类辅助评分，与 Hua 速度几何检验形成 ensemble score。

## 4. 文献是否给出了 `tangent_alignment` 的统一百分比？

没有找到一个可直接作为通用标准的“百分之多少的速度方向必须落在百分之多少的切向范围内”。

这点很重要。我们当前的：

```text
70% points within +/-24 deg
```

更像是工程化 Hua-style 检验参数，而不是文献中普遍固定使用的国际标准。文献主流做法通常通过以下方式间接约束切向性：

1. 闭合流线或闭合 SLA/SSH 等值线。
2. 单个极值约束，避免一条轮廓包含多个涡核。
3. 形状误差或有效半径约束，避免过度拉长或破碎边界。
4. 最大平均旋转速度半径，用速度剖面定义动态边界。
5. 角动量、涡度或旋转一致性评分。

因此，对于 ACC，我们不应把 `70% / +/-24 deg` 当成文献硬标准。更合理的做法是把它作为一个可调质量评分，并通过敏感性实验确定适合 ACC 的范围。

## 5. 建议修改方案

### 5.1 短期参数扫描

先不重写算法，做一组可解释的参数扫描：

```text
boundary_monotonic_exception_limit: 0, 1, 2, 3
tangent_tolerance_deg: 24, 30, 36, 45
min_tangent_fraction: 0.70, 0.60, 0.50
symmetry_tolerance_deg: 120, 140, 160
```

重点比较：

1. 表层 Hua 通过率是否从 5.16% 回升到合理范围。
2. 通过对象是否明显增加但不过度污染。
3. 30 天以上 tracks 是否增加。
4. panel-family 中新增对象是否仍具有可解释的旋转结构。

### 5.2 把 `boundary_monotonic_rotation` 改成 soft score

建议不再用零例外硬阈值，而是记录：

```text
monotonic_score = 1 - direction_exceptions / n_valid_circle_points
```

然后设置分级：

```text
excellent: monotonic_score >= 0.95
good:      monotonic_score >= 0.85
weak:      monotonic_score >= 0.70
fail:      monotonic_score < 0.70
```

这样固定圆周扫到一个内凹/外凸边界点时，不会直接杀掉整个候选。

### 5.3 把 `tangent_alignment` 改成形变感知指标

建议先保留原指标，但增加两个替代分数：

```text
tangent_score_circle = fraction(|angle_to_circle_tangent| <= tolerance)
tangent_score_ellipse = fraction(|angle_to_ellipse_tangent| <= tolerance)
```

其中椭圆边界可以从 SSH/SLA 局地闭合轮廓或速度模最大环拟合得到。若椭圆切向分数明显高于圆切向分数，说明失败来自边界形变，而不是没有涡旋。

### 5.4 引入闭合轮廓或多半径最优边界

中期建议加入：

1. 多半径搜索：不在第一个失败半径处立即停止，而是允许 `start_radius_cells` 到 `max_radius_cells` 内选择综合评分最高的半径。
2. 闭合 SSH/SLA 轮廓：优先使用闭合轮廓作为边界，圆周只作为 fallback。
3. 轮廓形状误差：记录但不直接硬杀，特别是在 ACC 区域。
4. 最大平均旋转速度半径：用速度结构而不是固定半径定义边界。

## 6. 对当前 ACC 结果的解释

`boundary_monotonic_rotation` 大量失败，不等价于这些候选都不是涡旋。它更可能说明固定圆周经常穿过非圆边界、强剪切、锋面或局部变形区。

`tangent_alignment` 大量失败，说明理想圆切向假设在 ACC 中偏强。对于拉伸或椭圆涡，速度可能沿真实闭合边界切向，但不沿固定圆周切向。

`symmetry` 大量失败，说明对径速度关系不够理想。它可以帮助识别非对称性，但在 ACC 中应作为形态诊断，而不是简单剔除条件。

## 7. 推荐下一步

建议先做两层实验：

1. **参数敏感性实验**：只放宽 `boundary_monotonic_exception_limit` 和 `tangent_alignment`，不改算法结构，快速评估数量和污染风险。
2. **非圆边界实验**：实现闭合 SSH/SLA 轮廓或椭圆边界评分，把圆周单调、切向和对称性从硬阈值改为质量分数。

从文献方向看，ACC 这种强剪切、强变形区域不适合继续使用零例外的固定圆周边界单调检验。更合理的路线是：中心仍可由速度几何确定，边界则应由闭合轮廓、椭圆拟合或角动量一致性确定。

## 参考文献与资料

1. Nencioli, F., Dong, C., Dickey, T., Washburn, L., & McWilliams, J. C. (2010). A Vector Geometry-Based Eddy Detection Algorithm and Its Application to a High-Resolution Numerical Model Product and High-Frequency Radar Surface Velocities in the Southern California Bight. *Journal of Atmospheric and Oceanic Technology*. DOI: 10.1175/2009JTECHO725.1
2. Chelton, D. B., Schlax, M. G., & Samelson, R. M. (2011). Global observations of nonlinear mesoscale eddies. *Progress in Oceanography*. DOI: 10.1016/j.pocean.2011.01.002
3. Mason, E., Pascual, A., & McWilliams, J. C. (2014). A New Sea Surface Height-Based Code for Oceanic Mesoscale Eddy Tracking. *Journal of Atmospheric and Oceanic Technology*. DOI: 10.1175/JTECH-D-14-00019.1
4. Le Vu, B., Stegner, A., & Arsouze, T. (2018). Angular Momentum Eddy Detection and Tracking Algorithm. *Ocean Modelling*.
5. py-eddy-tracker documentation: https://py-eddy-tracker.readthedocs.io/en/stable/

