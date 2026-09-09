# Material-volume EP-flux 推广可靠性验证

日期：2026-09-09  
对象：`material_volume_ep_flux_full_extension_zh.pdf`

## 结论摘要

当前理论推广的大方向是可靠的：它没有把全材料体平均量当成局地空间场取散度，也没有把
QG/PV 平衡反演默认写成一个未经验证的局地通量散度。这两个修正是必要的，尤其适合我们现在
面对的大曲率 coherent 涡旋。

但目前不能直接表述为“水平动量通量和垂直热输运存在简单 EP-flux 局地闭合关系”。经典
TEM/QG EP flux 中的热通量项主要是经向浮力/热通量 `v'b'` 或 `v'T'`，它进入 EP flux 的
垂向分量；而我们关心的 `w'T'` 是垂直热输运，更直接属于热量预算或 APE 转换。二者可以在
材料体预算中共同诊断同一涡旋强迫过程，但不能在没有 PV/QG 反演或额外闭合检验时直接等同。

因此建议将理论表述固定为：

```text
水平动量通量 -> Reynolds stress 局地散度 A_i^R
热/浮力通量 -> 热量/APE 预算，并可通过 QG/PV 非局地平衡响应 L_i^B 影响动量/速度场
总强迫诊断 -> G_i = A_i^R + L_i + R_i
```

## 理论审阅

### 1. 局地场和体平均必须分开

文档中定义的

```text
R_ij^ell(x,t) = <u_i^dagger u_j^dagger>_ell
B_i^ell(x,t) = <u_i^dagger b^dagger>_ell
P_i^ell(x,t) = <u_i^dagger q^dagger>_ell
```

是局地粗粒化场，可以进入 `partial_j(...)`。这和全材料体平均

```text
<phi>_V = |V|^-1 int_V phi dV
```

不同。后者只有整体统计意义，不能在材料体内部再取空间散度。这个区分是可靠的。

### 2. Reynolds stress 散度项是局地闭合项

水平动量通量进入

```text
A_i^R = -rho0^-1 partial_j(rho0 R_ij^ell)
```

这在形式和量纲上是合理的。若 `rho0` 常数，`A_i^R` 的量纲为速度平方除以长度，
即加速度。体积分后可以对该散度项使用 Gauss 定理。但如果 `rho0(z)` 不常数，必须保留
`R_ij^ell partial_j ln(rho0)` 这类项，文档中已经提到，这是正确的。

### 3. 热/浮力通量不能直接写成局地动量通量散度

经典 TEM/QG 形式中常见关系为：

```text
F_y = -rho0 u'v'
F_z = rho0 f0/N^2 v'b'
```

所以传统 EP flux 的垂向分量与 `v'b'` 成正比，而不是 `w'b'`。如果后续论文要强调
“垂直热输运”，应写成 `w'T'` 或 `w'b'` 对热量/APE 的贡献；若要把它和水平动量通量联系起来，
必须通过材料体热成风/PV 平衡响应：

```text
L_i^B = U_i M^-1 S_B[B^ell]
```

这只是非局地响应定义，不是已经证明的局地闭合公式。只有当数值上能证明存在小残差局地化：

```text
L_i^B = rho0^-1 partial_j(rho0 H_ij^B) + epsilon_i^B,
|epsilon_i^B| << |L_i^B|
```

才可以回到类似传统 EP flux 的局地通量解释。

### 4. 需要补充的理论条件

为了让理论从“可靠框架”变成“可计算方法”，文档还需要固定这些定义：

- `M` 的具体形式：QG PV inversion、omega equation、SQG-like inversion，还是经验 Green 函数。
- 边界条件 `B`：水平边界、上下边界、材料体边界是否允许穿透。
- 背景态：`N^2(z)`、`rho0(z)`、`f0/beta` 是局地纬度、纬度带平均还是气候态。
- 粗粒化尺度 `ell`：相对涡旋半径 `R` 的多少，是否随纬度/半径变化。
- 残差项 `R_i`：年龄转、混合、非保守 PV、边界识别误差、尺度截断误差如何估计。

## 数值验证设计

### 目标

验证的目标不是证明 `w'T' = u'v'`，而是判断：

```text
A_x^R, B_y=v'b', B_z=w'b' 或 w'T'
```

是否能在同一批涡旋样本中形成稳定、可解释、残差可控的材料体预算关系。

### 第一阶段：20N proof-of-concept

固定已有的 `20N crossing / 1R / match-mode all / recommended` 口径，cyclonic 和 anticyclonic
分开计算，不生成 combined。

需要输出的诊断量：

```text
R_xy = <u'v'>
R_xz = <u'w'>
B_y  = <v'b'> 或 <v'T'>
B_z  = <w'b'> 或 <w'T'>
A_x^R = -rho0^-1 [partial_x(rho0 R_xx) + partial_y(rho0 R_xy) + partial_z(rho0 R_xz)]
```

如果目前没有可靠的三维水平速度异常 `u',v'`，第一阶段只做可用项：

```text
R_xz = <u'W_rebuild>
B_z = <W_rebuild T'>
```

并明确标注 `R_xy` 与完整 `A_x^R` 尚未闭合。

### 第二阶段：区分 `v'b'` 与 `w'b'`

至少做两套热/浮力通量诊断：

```text
EP-like heat-flux branch: B_y = <v'b'>
vertical-heat-transport branch: B_z = <w'b'> 或 <w'T'>
```

判据：

- 若 `B_y` 与 EP/TEM 预期方向一致，而 `B_z` 主要解释热量/APE 变化，则理论应写成“两条耦合预算”，不能写成简单替换。
- 若 `B_z` 经由明确的 `S_B -> M^-1 -> L_x^B` 后能解释动量强迫，才可以说垂直热输运通过平衡响应连接到水平动量通量。

### 第三阶段：残差预算

定义候选目标强迫 `G_x`。如果没有直接动量倾向，可先用涡旋速度/质心或合成动量变化的诊断代理。

计算：

```text
R_x = G_x - A_x^R - L_x^B - L_x^P - L_x^c
```

接受标准：

- `A_x^R + L_x^B` 比单独 `A_x^R` 显著提高解释率。
- `|R_x|` 的 95% 分位不大于主项量级。
- bootstrap/jackknife 后相关和回归系数符号稳定。
- cyclonic/anticyclonic 分开成立，而不是只在 combined 中成立。

## 20N 输出建议

第一张四联图：

```text
R_xy 或 R_xz
A_x^R
w'T' 或 w'b'
residual / correlation section
```

第二张剖面图：

```text
depth vs x/R:
A_x^R
B_y branch response
B_z branch response
G_x 或 residual
```

第三张表：

```text
polarity
sample count
valid fraction
corr(A_x^R, G_x)
corr(A_x^R + L_x^B, G_x)
regression slope
residual q95
bootstrap confidence interval
```

## 文献依据

- Andrews and McIntyre 1976/1978；Andrews et al. 1987：TEM/EP flux 的经典理论，EP flux 垂向分量与经向热/浮力通量有关。可参考 GFDL 收录的 Andrews 综述材料：<https://www.gfdl.noaa.gov/>。
- Plumb 1985/1986：三维波活动通量和 QG stationary/transient wave activity flux，不等同于任意材料体内的局地热输运。Plumb 1986 的题名为 *Three-Dimensional Propagation of Transient Quasi-Geostrophic Eddies and Its Relationship with the Eddy Forcing of the Time-Mean Flow*。
- NCAR NCL `epflux` 说明：EP flux divergence 常作为 eddy PV flux 诊断，适合作为“诊断工具”而非任意闭合公式的参考：<https://www.ncl.ucar.edu/>。
- Ferrari et al. 2010 及相关 PV inversion 文献：PV/温度反演是依赖边界条件的非局地问题；可参考 *A boundary-value problem for the parameterized mesoscale eddy transport*：<https://empslocal.ex.ac.uk/>。

## 当前可靠性评级

| 部分 | 评级 | 理由 |
|---|---|---|
| 大曲率材料体替代 curved-tube 小曲率框架 | 较可靠 | 当前涡旋曲率风险高，使用 Cartesian 材料体更稳。 |
| 局地 Reynolds stress 散度 `A_i^R` | 可靠 | 形式、量纲和体积分关系成立。 |
| 非局地响应 `L_i=U_i M^-1 S_i` | 概念可靠 | 但需要明确 `M`、源项和边界条件。 |
| `v'b'` 作为 EP-like 热通量项 | 可靠 | 符合经典 TEM/QG EP flux。 |
| `w'T'` 直接作为 EP flux 垂向分量 | 不可靠 | 它是垂直热输运/APE 项，不能直接替代 `v'b'`。 |
| 水平动量通量与垂直热输运的关系 | 待数值验证 | 可以通过材料体预算和 PV/QG 响应建立联系，但不是简单代数等式。 |

## 推荐修改措辞

建议把核心结论写成：

```text
在大曲率 coherent 涡旋中，水平动量通量通过 Reynolds stress 局地散度直接进入材料体动量强迫；
垂直热输运通过热量/APE 改变密度和热成风结构，并可经 QG/PV 非局地平衡反演影响同一材料体强迫。
二者不是简单相等或局地替代关系，而是在 G_i = A_i^R + L_i + R_i 的预算中共同闭合。
```

不建议写成：

```text
垂直热输运就是 EP flux 垂向分量。
水平动量通量和 w'T' 存在直接局地散度平衡。
```

## 下一步执行建议

先不要修改正式 W 管线。下一步应新开独立 worktree，实现一个 `EP-FLUX` 诊断入口，只跑 20N：

1. 从现有 `W_3D` 与密度/温度异常结果读入 `W`、`T'`、`rho'`。
2. 若已有三维速度异常，计算 `R_xy/R_xz`；否则先计算 `R_xz=<u'W>` 和 `B_z=<WT'>`。
3. 输出 20N 两极性四联图和指标表。
4. 根据残差和相关结果决定是否进入 QG/PV 反演实现。
