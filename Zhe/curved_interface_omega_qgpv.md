# A-only 曲面/等密度面 QGPV：从标准层模型到 revised curved-surface QGPV

## 摘要

本文将原来的多分支经验修正体系修订为 **baseline / revised A** 两类模型。本文只讨论标准 QGPV 与 revised curved-surface QGPV；旧的非地转诊断分支不再作为正式理论模型参与本文推导、验证或图表解释。

本文的 revised A model 不是单独叠加一个经验修正项，而是把标准 QGPV 的水平平面算子替换为等密度面上的曲面算子。理论主线为：

```math
\text{standard QGPV}
\rightarrow
S_i:\rho=\rho_i
\rightarrow
g_{ab},\ g^{ab},\ \Delta_g
\rightarrow
q_i^{A}
\rightarrow
\text{tilted eddy response}.
```

核心思想是：等密面 \(S_i\) 是物质面，涡旋倾斜首先表现为不同密度面上的流函数、涡心和等密面几何发生相对偏移。因此，适合描述该过程的模型应直接在 \(S_i\) 的切平面上定义地转流、曲面 Laplacian 和 QGPV，而不是先在固定 \(p\) 坐标中诊断密度面形变。

## 0. 符号和坐标约定

设三维位置为 \(\mathbf x=(x,y,z)\)。密度为 \(\rho(\mathbf x,t)\)，参考密度为 \(\rho_0\)。第 \(i\) 个等密度面定义为

```math
S_i(t)=\{\mathbf x:\rho(\mathbf x,t)=\rho_i\}.
```

在 \(S_i\) 上取局地曲面坐标 \(\xi^1,\xi^2\)，曲面嵌入写作

```math
\mathbf r_i(\xi^1,\xi^2,t)
=
\mathbf r_i^0(\xi^1,\xi^2)
+\eta_i(\xi^1,\xi^2,t)\mathbf n_i^0(\xi^1,\xi^2),
```

其中 \(\eta_i\) 为等密度面的法向位移，\(\mathbf n_i^0\) 为基准等密度面的单位法向量。曲面切向基矢为

```math
\mathbf r_a=\frac{\partial \mathbf r_i}{\partial \xi^a},
\qquad a=1,2.
```

曲面度量张量定义为

```math
g_{ab}=\mathbf r_a\cdot\mathbf r_b,
\qquad
g=\det(g_{ab}),
\qquad
g^{ab}=(g_{ab})^{-1}.
```

当曲面可写成图形面 \(z=Z_i(x,y,t)\) 时，

```math
g_{xx}=1+Z_x^2,\qquad
g_{xy}=Z_xZ_y,\qquad
g_{yy}=1+Z_y^2,
```

并且

```math
g=1+Z_x^2+Z_y^2=1+|\nabla Z|^2.
```

曲面 Laplace-Beltrami 算子定义为

```math
\Delta_g\psi
=
\frac{1}{\sqrt g}
\frac{\partial}{\partial \xi^a}
\left(
\sqrt g\,g^{ab}
\frac{\partial \psi}{\partial \xi^b}
\right).
```

重复指标 \(a,b\) 采用 Einstein 求和约定。流函数 \(\psi_i\) 在 \(S_i\) 上定义，曲面地转速度为

```math
u_i^a
=
\frac{1}{f_0}\epsilon^{ab}\nabla_b\psi_i,
```

其中 \(\epsilon^{ab}\) 是曲面上的反对称张量。在局地平面极限下，

```math
u_i=-\frac{\partial\psi_i}{\partial y},
\qquad
v_i=\frac{\partial\psi_i}{\partial x}.
```

## 1. 标准 QGPV 的问题

标准连续 QGPV 可写为

```math
q
=
f_0+\beta y+\nabla_h^2\psi
+
\frac{\partial}{\partial z}
\left(
\frac{f_0^2}{N^2}
\frac{\partial\psi}{\partial z}
\right).
```

这里 \(\nabla_h^2=\partial_x^2+\partial_y^2\) 是固定水平平面上的 Laplacian，\(N^2\) 为浮力频率平方。该形式适合弱位移、弱倾斜并且密度面接近水平的情况。

但是在倾斜涡旋中，密度面本身发生可观弯曲，不同层涡心也会产生相对位移。若仍把所有水平导数都放在固定 \((x,y)\) 平面中，则等密面几何只通过后处理诊断进入，容易把物质面形变解释成固定坐标下的垂向混合或虚假 stretching。

因此 revised A model 的出发点是：在 \(S_i\) 上直接定义 PV，而不是先在 \(p\) 坐标或固定深度坐标中写完方程再诊断 \(S_i\)。

## 2. 等密度面是物质面

等密度面的核心约束是

```math
\frac{D\rho}{Dt}=0.
```

因此对 \(S_i:\rho=\rho_i\)，有

```math
\frac{D}{Dt}(\rho-\rho_i)=0.
```

若用法向位移 \(\eta_i\) 描述曲面运动，则曲面法向速度为

```math
w_{n,i}
=
\frac{D\mathbf r_i}{Dt}\cdot\mathbf n_i^0
=
\partial_t\eta_i+u_i^a\nabla_a\eta_i.
```

这个式子说明：等密面形变不是外加图形，而是被切向流和法向运动共同推进的物质面几何。倾斜涡旋的“层间涡心偏移”应理解为不同 \(S_i\) 上流函数结构的相对平移和曲面几何差异。

## 3. 表面节点与内部流形的统一 PV 反演

原始 CMEMS 数据包含海面高度 `zos_glor`，本工程在每日输入中记为 `adt`，在生命周期合成中记为 `adt_anom`。因此海表不能作为刚性盖，也不应作为与内部孤立相加的外部补丁。按照 Bretherton 边界 PV 与 surface-QG 的思想，海表边界与内部 PV 是同一个椭圆反演问题的边界源和体源。本文把自由面写成第 0 个流形节点

```math
S_0=S_\eta:\ z=\eta(x,y,t),
```

内部等密度面仍为

```math
S_i(t)=\{\mathbf x:\sigma_0(\mathbf x,t)=\rho_i\},\qquad i=1,\dots,N.
```

这里 \(S_0\) 与 \(S_i\) 共同构成一个可变形流形网络。revised A 的核心创新仍然是：每个 \(S_i\) 都可以发生凹凸、倾斜和相对偏移，而不是刚性平面平移。

海表压力异常为

```math
p_s'=\rho_0g\eta,
\qquad
\psi_\eta=g\eta.
```

因此表面节点的地转速度为

```math
u_{g,\eta}=-\frac{g}{f_0}\frac{\partial\eta}{\partial y},
\qquad
v_{g,\eta}=\frac{g}{f_0}\frac{\partial\eta}{\partial x}.
```

在合成坐标 \(x^\ast=x/R,\ y^\ast=y/R\) 中，

```math
\psi_\eta^\ast=\frac{g\eta}{UR},
\qquad
u_{g,\eta}=-\frac{g}{f_0R}\frac{\partial\eta}{\partial y^\ast},
\qquad
v_{g,\eta}=\frac{g}{f_0R}\frac{\partial\eta}{\partial x^\ast}.
```

自由面对应的外模变形半径仍定义为

```math
R_{\eta,1}=\frac{\sqrt{gH_1}}{f_0},
```

但它只提供表面节点与顶层内部节点之间的耦合强度，而不是一个独立叠加项。

## 4. Morel 2019 与等密面 PV 闭合约束

Morel, Gula and Ponte (2019) 提供了一个对 revised A 特别重要的约束：PV 不应只被看作固定深度网格上的局地点值，而应同时满足体积分与边界通量之间的闭合关系。对任意密度场 \(\sigma_0(\mathbf x,t)\)，Ertel PV 可写成散度形式

```math
PV
=
\nabla\cdot
\left(
\mathbf U_a\times\nabla\sigma_0
\right),
```

其中 \(\mathbf U_a\) 为绝对速度，\(\nabla\sigma_0\) 为总位势密度梯度。这个形式只依赖矢量微积分恒等式，因此天然适合一般曲面坐标、等密度坐标和本文的可变形流形 \(S_i:\sigma_0=\rho_i\)。

对任意由两张相邻等密面围成的控制体

```math
V_i(t)
=
\{
\mathbf x:
\rho_i\le \sigma_0(\mathbf x,t)\le\rho_{i+1}
\},
```

其边界 \(\partial V_i\) 包括上、下等密面、表面节点 \(S_\eta\) 可能产生的 outcropping 边界、侧边界以及可能的底边界。由散度定理可得

```math
\int_{V_i} PV\,dV
=
\oint_{\partial V_i}
\left(
\mathbf U_a\times\nabla\sigma_0
\right)\cdot d\mathbf S.
```

这个等式说明：revised A 的曲面几何项不能只看局地 \(\Delta_g\psi\) 的形状，也必须接受 \(V_i\) 上的 PV 体积分与边界通量闭合约束。若一个曲面修正项在局部看起来很强，但破坏了该闭合关系，则它只能作为 diagnostics，不能直接进入正式 QGPV forcing。

为了把这个约束写成可验证量，定义第 \(i\) 个等密面控制体内的 PV anomaly 为

```math
PVA_i
=
PV_i-PV_{i,\mathrm{ref}},
```

其中 \(PV_{i,\mathrm{ref}}\) 是同一密度层在参考静止层结下的 PV。对应的闭合残差定义为

```math
\epsilon_i^{PV}
=
\frac{
\left|
\int_{V_i} PVA_i\,dV
-
\oint_{\partial V_i}
\left(
\mathbf U_a\times\nabla\sigma_0
\right)\cdot d\mathbf S
\right|
}{
\left|\int_{V_i} PVA_i\,dV\right|+\epsilon
},
```

其中 \(\epsilon\) 是避免分母为零的小量。后续数值验证中，\(\epsilon_i^{PV}\) 应作为 A model 是否可进入正式 forcing 的质量控制指标。

Morel 2019 对孤立涡旋还指出，层内 PVA 体积分与表面密度异常、表面涡度及边界条件之间存在约束。对本文而言，这意味着 \(S_\eta\) 不是外加补丁，而是 \(S_\eta+S_i\) 统一 PV 网络的边界节点；内部 \(S_i\) 的 PV anomaly 与自由面、表层密度和边界通量必须共同满足同一个闭合条件。

为了描述涡旋倾斜过程中的 PV 主体偏移，定义 PV centroid：

```math
\mathbf r_{PV,i}
=
\frac{
\int_{V_i}\mathbf r\,|PVA_i|\,dV
}{
\int_{V_i}|PVA_i|\,dV
}.
```

这里使用 \(|PVA_i|\) 作为第一版权重，是为了避免正负 PVA 在同一层内相互抵消。后续可进一步分别定义正 PVA centroid 与负 PVA centroid。相对表层的位涡距定义为

```math
TD_{PV,i}^{\ast}
=
\frac{
|\mathbf r_{PV,i}-\mathbf r_{PV,0}|
}{
R_i
},
```

相邻层的位涡距定义为

```math
TD_{PV,i,i-1}^{\ast}
=
\frac{
|\mathbf r_{PV,i}-\mathbf r_{PV,i-1}|
}{
\bar R_{i,i-1}
}.
```

因此，本文后续区分三类倾斜中心：

```text
velocity core tilt:
    速度零点或速度结构中心的偏移。

completed-center tilt:
    由 completed centers 给出的连续涡心结构偏移。

PV-centroid tilt:
    由 PVA 主体质量中心给出的位涡距偏移。
```

若 \(TD_{PV,i}^{\ast}\) 与 completed-center tilt 同相，说明涡旋倾斜对应 PV 主体本身的偏移；若二者不同相，则说明速度核心、几何涡心与 PV 异常主体发生脱耦。对于 complex 类型，若 \(TD_{PV,i}^{\ast}\) 随深度跳变或 \(\epsilon_i^{PV}\) 很大，则更适合解释为多核、断裂或边界通量主导，而不是单一连续涡柱的倾斜。

## 5. Revised A：统一曲面-层间耦合 QGPV

在第 \(i\) 个内部等密度面 \(S_i\) 上，revised A 写作

```math
q_i^A
=
f_{n,i}
+
\frac{1}{f_0}\Delta_{S_i}\psi_i
+
\mathcal S_i^A.
```

其中 \(\Delta_{S_i}\) 是可变形密度流形上的 Laplace--Beltrami 算子，\(\mathcal S_i^A\) 是统一的曲面-层间耦合算子。一般形式为

```math
\mathcal S_i^A
=
C_{i-\frac12}
\left(\mathcal P_{i-1\to i}\psi_{i-1}-\psi_i\right)
+
C_{i+\frac12}
\left(\mathcal P_{i+1\to i}\psi_{i+1}-\psi_i\right).
```

这里 \(\mathcal P_{j\to i}\) 把相邻流形 \(S_j\) 上的流函数投影到 \(S_i\)。第一版数值实现采用同一 \(x^\ast,y^\ast\) 网格上的直接投影；后续可升级为真正曲面法向投影。

对于顶层内部面 \(S_1\)，上邻居不是空层，而是表面节点 \(S_0=S_\eta\)：

```math
\mathcal P_{0\to1}\psi_0
=
\mathcal P_{\eta\to1}\psi_\eta.
```

因此顶层上边界耦合写作

```math
\mathcal S_{\eta,1}^A
=
C_{\eta,1}
\left(\mathcal P_{\eta\to1}\psi_\eta-\psi_1\right),
\qquad
C_{\eta,1}\sim\frac{f_n^{(1)2}}{f_0^2R_{\eta,1}^2}.
```

在无量纲合成坐标中，最低阶实现为

```math
\mathcal S_{\eta,1}^{A\ast}
=
\left(\frac{R}{R_{\eta,1}}\right)^2
\left(\psi_\eta^\ast-\psi_1^\ast\right),
\qquad
\psi_\eta^\ast=\frac{g\eta}{UR}.
```

这说明自由面和内部顶层是同一个 PV 反演网络中的相邻节点。旧写法 \(- (R/R_{\eta,1})^2\psi_1^\ast\) 只是把 \(\psi_\eta^\ast\) 默认为零的退化情形，不能用于有 `adt_anom` 的数据。

对应的 baseline 仍为固定深度平面形式

```math
q_i^0
=
f_0+\beta y
+
\frac{1}{f_0}\nabla_h^2\psi_i
+
\mathcal S_i^0.
```

因此 revised A 的修正可拆为

```math
\delta q_i^A
=
\frac{1}{f_0}(\Delta_{S_i}-\nabla_h^2)\psi_i
+
\delta\mathcal S_i^A
+
(f_{n,i}-f_0).
```

第一版数值验证保留两类可检查贡献：内部流形曲面 Laplacian 修正，以及表面节点 \(S_\eta\) 与顶层 \(S_1\) 的统一耦合。内部相邻流形间的显式投影差先保留为诊断占位，后续再扩展。

## 6. 小坡度展开

当 \(S_i\) 可写为 \(z=Z_i(x,y,t)\)，且 \(|\nabla Z_i|\ll 1\) 时，

```math
\Delta_g\psi
=
\nabla_h^2\psi
+
\delta\Delta_g\psi
+
O(|\nabla Z|^4).
```

令

```math
Z_x=\frac{\partial Z}{\partial x},\qquad
Z_y=\frac{\partial Z}{\partial y},
```

则一阶有效的曲面修正可写作

```math
\delta\Delta_g\psi
\approx
-
\nabla_h\cdot
\left[
\nabla Z(\nabla Z\cdot\nabla_h\psi)
\right]
+
\frac12\nabla_h\cdot
\left[
|\nabla Z|^2\nabla_h\psi
\right]
-
\frac12|\nabla Z|^2\nabla_h^2\psi.
```

这个式子是当前 revised A 数值实现的核心。它把等密面曲率、坡度和流函数梯度耦合起来，给出相对涡度/PV 的几何改变量。

## 7. 倾斜涡旋中的物理解释

设每层涡心为

```math
\mathbf r_{c,i}=(x_{c,i},y_{c,i}),
```

相对表层偏移为

```math
\Delta\mathbf r_i
=
\mathbf r_{c,i}-\mathbf r_{c,0}.
```

若等密度面上的对称密度异常为 \(\rho'_{s,i}\)，倾斜导致的平移展开为

```math
\rho'_i(\mathbf x-\Delta\mathbf r_i)
\approx
\rho'_{s,i}(\mathbf x)
-
\Delta\mathbf r_i\cdot\nabla_h\rho'_{s,i}.
```

等密面位移满足

```math
\eta_{\rho,i}
=
-
\frac{\rho'_i}{\partial_z\bar\rho}.
```

因此非对称项的一阶主导量为

```math
\eta_{\rho,\mathrm{odd},i}
\approx
\frac{
\Delta\mathbf r_i\cdot\nabla_h\rho'_{s,i}
}{
\partial_z\bar\rho
}.
```

revised A 的作用是进一步指出：这种等密面非对称不仅改变 \(\eta_\rho\)，还通过 \(g_{ab}\)、\(g^{ab}\) 和 \(\Delta_g\psi\) 改变 \(q_i\)。所以倾斜涡旋的速度响应不应只从固定平面上的 \(\nabla_h^2\psi\) 判断，而应比较

```math
q_i^0
\quad\text{and}\quad
q_i^A.
```

## 8. 基态与扰动偏移

在无陆地、近似全球尺度的理想背景下，等密度面可近似为微扰球面。曲面 Laplacian 的本征函数满足

```math
-\Delta_{g_i}Y_{\ell m}
=
\lambda_{\ell m}^{(i)}Y_{\ell m}.
```

在球面极限，

```math
\lambda_{\ell m}=\frac{\ell(\ell+1)}{a^2}.
```

最低非零模态为 \(\ell=1\)。若取轴对称基态，

```math
Y_{10}(\phi)
=
\sqrt{\frac{3}{4\pi}}\sin\phi,
```

两层正压基态可写为

```math
\psi_1^0=\psi_2^0=AY_{10}(\phi).
```

斜压扰动定义为

```math
\delta\psi_i=\psi_i-\psi_i^0.
```

若扰动被局地权重 \(W(\lambda)\) 限制，并且第 \(i\) 层相对顶层存在小经向或纬向偏移 \(\Delta\lambda_i\)，可写作

```math
\delta\psi_i
=
\sigma\epsilon
W(\lambda-\Delta\lambda_i)
Y_{10}(\phi).
```

小偏移展开为

```math
W(\lambda-\Delta\lambda_i)
\approx
W(\lambda)-\Delta\lambda_i W'(\lambda).
```

当深层斜压调整受基态流平流且存在时滞时，

```math
\frac{\partial \Delta\lambda_i}{\partial t}
\sim
\frac{u_i^0}{R_{d,i}},
```

从而

```math
\Delta\lambda_i
\sim
\frac{u_i^0 t}{R_{d,i}}.
```

这给出层间系统性偏移的理论来源。revised A 将这种偏移进一步映射为曲面 PV 几何项，而不是把它解释为单纯的固定坐标伸缩项。

## 9. 验证口径

正式验证只比较两种模型：

```math
\text{baseline}: q_i^0,
\qquad
\text{revised A}: q_i^A.
```

验证输出应包含

```text
qgpv_baseline_full
qgpv_model_A_full
qgpv_model_A_correction
pv_balance_residual
pv_centroid_x_R
pv_centroid_y_R
TD_PV_star
TD_PV_adjacent_star
```

不应再生成其他经验修正分支作为正式模型变量。

速度验证目标仍为生命周期合成中的去气候态涡旋异常速度：

```math
u' = u-\overline u_{\mathrm{doy}},
\qquad
v' = v-\overline v_{\mathrm{doy}}.
```

Morel 2019 对本文的验证口径提出一个额外要求：A model 的曲面几何项必须同时接受 PV 闭合检查。若某一层或某一相位中

```math
\epsilon_i^{PV}
```

较大，则该层的 \(q_i^A-q_i^0\) 只能作为 diagnostics 解释，不应直接进入正式 QGPV forcing。换言之，几何项是否“看起来像”涡旋结构并不足够；它必须同时满足等密面控制体 \(V_i\) 上的 PV 体积分与边界通量闭合。

除速度 skill 外，验证还应比较三种倾斜诊断：

```text
completed-center tilt:
    TD_i^* from completed centers.

PV-centroid tilt:
    TD_PV_i^* from PVA centroid.

velocity response:
    u'/v' section and topview skill.
```

若 coherent 类型中 \(TD_{PV,i}^{\ast}\) 与 completed-center \(TD_i^\ast\) 同相，并且 revised A 的 PV 闭合残差较小，则说明倾斜是 PV 主体和等密面几何共同偏移的结果。若 \(TD_{PV,i}^{\ast}\) 与 completed centers 脱耦，或者 \(\epsilon_i^{PV}\) 很大，则应优先解释为多核、断裂、边界通量主导或未闭合的几何 proxy，而不是连续涡柱倾斜。

若 revised A 相比 baseline 在 coherent 类型中提升，而在 upright-like 类型中不产生同样提升，则说明曲面/等密度面几何确实更适合描述倾斜涡旋。若 revised A 对速度场不提升，但对密度面或 PV 响应提升，则说明该理论更适合解释等密面/PV 结构，而不是直接闭合完整速度预报。

## 10. QG 柱涡本征原型

为了把 Viúdez 型 Beltrami/Trkalian 精确涡中的“本征模态”思想转化为更适合中尺度涡的语言，本文采用 QG 位涡反演算子而不是 curl 算子。常 \(f_0,N\) 下，

```math
L\psi
=
\nabla_h^2\psi
+
\partial_z
\left(
\frac{f_0^2}{N^2}
\partial_z\psi
\right),
\qquad
\hat H\psi=-L\psi.
```

在柱坐标中，分片柱涡可写为

```math
q'
=
\begin{cases}
-K^2\psi, & r<a,\\
0, & r>a.
\end{cases}
```

内部满足 \(L\psi+K^2\psi=0\)，外部满足 \(L\psi=0\)。单个 \((m,k_z)\) 模态的内部径向结构为 \(J_m(\kappa r)\)，外部衰减结构为 \(K_m(\gamma r)\)，其中

```math
\kappa^2=K^2-\frac{f_0^2}{N^2}k_z^2,
\qquad
\gamma=\frac{f_0}{N}|k_z|.
```

边界 \(r=a\) 上的 \(\psi\) 与 \(\partial_r\psi\) 匹配给出离散/连续混合谱的本征条件。直立涡柱对应 \(m=0\)，直线倾斜的一阶响应对应 \(m=1\) 位移模态，helical 倾斜对应 \(\cos(\theta+k_z z)\) 型相位模态。完整推导和可计算验证量见 `qg_cylindrical_eigenmodel.md`，验证实现见 `src/validation/qg_cylindrical_eigenmodel.py`。

## 参考资料

- Majda, A. J. and Wang, X. (2005). *Nonlinear Dynamics and Statistical Theories for Basic Geophysical Flows*.
- Vallis, G. K. *Atmospheric and Oceanic Fluid Dynamics*.
- Pedlosky, J. *Geophysical Fluid Dynamics*.
- Morel, Y., Gula, J., and Ponte, A. (2019). Potential vorticity diagnostics based on balances between volume integral and boundary conditions. *Ocean Modelling*, 138, 23-35.
- 本文档的 revised A 推导依据用户提供的 `cuverlinear theory.docx` 中“曲面模型”修改意见整理。
