# Argo-Eddy W 合成方法文献核查

## 已下载到指定目录的论文

PDF 目录：`D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\PDF\Dipole_vertical _transport`

- `JGR Oceans - 2026 - Zhang - Dipole Structure of Vertical Velocity Induced by Mesoscale Eddies.pdf`
- `Hou_etal_2022_Frontiers_Eddy_beta_spiral.pdf`

AGU/Wiley 的 Chaigneau et al. (2011)、Mason et al. (2017)、Qu et al. (2022)
官方 PDF 直链被 Cloudflare 拦截，当前未下载成功，因此不把它们作为本地
PDF 依据。后续若手动下载成功，再纳入本目录和本记录。

## 文献方法要点

Zhang et al. (2026) 是数值模式研究，不是 Argo 观测重建方法。它的结论是
中尺度涡诱导的 W 以偶极子为主，并且 composite 时把每个涡旋快照归一化到
以涡心为原点、以涡半径 R 为尺度的统一坐标，再插值到均匀笛卡尔网格后
平均；文中没有把散点直接画成 W 图。

Hou et al. (2022) 用 QG omega equation / eddy beta-spiral 解释孤立涡的
东西向 W 偶极子。该文强调 W 偶极子来自动力平衡诊断，量级可到
`10^-5-10^-4 m/s`，但它也不是用 Argo parking-depth density 直接重建
`z_rho` 的观测方法。

## 我们当前结果不一样的原因

1. 早期代码把候选涡旋的 `dx/dy` 向量写进了单条匹配记录，而不是只保存最佳
   匹配涡旋的 `dx(best_pos)/dy(best_pos)`。这会污染 `x/R, y/R`，使 W 图像
   看起来像稀疏数量图。
2. 逐 profile 使用自身 parking-depth density 作为 `rho0` 时，`z_rho` 几乎
   退化为该 profile 自身 parking depth，`dz_rho/dx` 和 W 会接近 0。
3. 共同 `rho0` 可以恢复同一等密面起伏，但必须限制 `z_rho` 在 parking layer
   附近；否则共同密度面可能在某些 profile 中跳到浅层或深层交点，造成不合理
   大梯度。
4. 文献 composite 通常会把涡心归一化后的样本插值/平均到连续网格；单纯
   bin median 只能作为覆盖诊断，不应作为正式 W 图。

## 当前代码口径

- 默认全球范围：`0E-360E, 60S-60N`。
- 默认 5 度纬度带，并使用半球显式目录标签。
- 默认 `rho0-mode = band_median`，即每个纬度带/极性用共同 parking-layer
  density 中位数。
- 默认 `z_rho` 有效窗口：`900-1100 m`。
- 默认 `grid-mapping = scattered`，用连续插值 composite 生成 `z_rho/u/v`
  网格后再计算 W。
- 保留 `sample_count` 和 `mapped_support`，分别用于覆盖诊断和插值支撑诊断。
