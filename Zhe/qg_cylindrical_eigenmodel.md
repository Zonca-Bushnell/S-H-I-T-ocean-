# QG 柱涡本征模型：PV 算子的类量子分解

本文给出一个用于倾斜中尺度涡解释的第一版解析模型。它不使用
Beltrami/Trkalian 条件

```math
\nabla\times\mathbf u=k\mathbf u,
```

而是把准地转位涡反演算子作为本征算子：

```math
q'=L\psi,\qquad
L=\nabla_h^2+\partial_z\left(S\partial_z\right),
\qquad
S=\frac{f_0^2}{N^2}.
```

在常 \(f_0,N\) 的第一版中，\(S\) 为常数。定义形式 Hamiltonian

```math
\hat H\psi=-L\psi.
```

若

```math
\hat H\psi=\Lambda\psi,
```

则 \(\Lambda\) 是 QG PV 算子的本征值。这个类比只用于算子、本征模态、
连续谱和波包，不把 \(\Lambda\) 解释为真实量子能量。

## 1. 柱坐标中的 QG PV 算子

在柱坐标 \((r,\theta,z)\) 中，

```math
L\psi=
\frac{1}{r}\partial_r(r\partial_r\psi)
+\frac{1}{r^2}\partial_{\theta\theta}\psi
+S\partial_{zz}\psi.
```

取单模态

```math
\psi_m(r,\theta,z)=R(r)e^{im\theta}e^{ik_z z},
```

则

```math
L\psi_m=
\left[
R''+\frac{1}{r}R'
-\frac{m^2}{r^2}R
-Sk_z^2R
\right]e^{im\theta}e^{ik_z z}.
```

因此 \(-i\partial_\theta\) 给出方位模态数 \(m\)，
\(-i\partial_z\) 给出垂向连续波数 \(k_z\)。没有上下边界时，
\(k_z\) 是连续谱；局地涡柱由多个 \(k_z\) 叠成 wave packet。

## 2. 分片柱涡

令涡半径为 \(a\)，内部 PV 与流函数成正比，外部无 PV 异常：

```math
q'=
\begin{cases}
-K^2\psi, & r<a,\\
0, & r>a.
\end{cases}
```

内部满足

```math
L\psi+K^2\psi=0,
```

外部满足

```math
L\psi=0.
```

对单个 \((m,k_z)\) 模态，内部径向方程为

```math
R_i''+\frac{1}{r}R_i'
-\frac{m^2}{r^2}R_i
+\kappa^2R_i=0,
\qquad
\kappa^2=K^2-Sk_z^2.
```

正则解为

```math
R_i(r)=J_m(\kappa r).
```

外部径向方程为

```math
R_e''+\frac{1}{r}R_e'
-\frac{m^2}{r^2}R_e
-\gamma^2R_e=0,
\qquad
\gamma=\sqrt{S}|k_z|.
```

随 \(r\to\infty\) 衰减的解为

```math
R_e(r)=C K_m(\gamma r),
```

其中 \(K_m\) 是第二类 modified Bessel 函数。由 \(\psi\) 连续得到

```math
C=\frac{J_m(\kappa a)}{K_m(\gamma a)}.
```

若进一步要求径向速度/压力梯度意义下的 \(\partial_r\psi\) 连续，则得到
本征条件

```math
\kappa\frac{J_m'(\kappa a)}{J_m(\kappa a)}
=
\gamma\frac{K_m'(\gamma a)}{K_m(\gamma a)}.
```

这就是分片 QG 柱涡的“量子化”条件：给定 \(a,S,K\) 后，允许的
\((m,k_z)\) 由边界匹配残差决定。

## 3. 倾斜模态

轴对称基态为

```math
m=0.
```

它表示直立柱涡。小幅直线倾斜可以写成基态的水平位移展开：

```math
\psi(r,\theta,z)
\approx
\psi_0(r,z)
-X(z)\partial_x\psi_0
-Y(z)\partial_y\psi_0.
```

由于 \(\partial_x\psi_0,\partial_y\psi_0\) 带有 \(\cos\theta,\sin\theta\)
结构，直线倾斜的一阶响应属于 \(m=1\) 扰动。

螺旋倾斜可以直接写成 \(m=1\) 相位模态：

```math
\psi_1(r,\theta,z)=A(r)\cos(\theta+k_z z+\phi_0).
```

它描述随高度旋转的偏心方向。直线倾斜更适合涡心近似沿一个方向随深度
偏移的中尺度涡；螺旋倾斜更接近 helical/相位模态。

## 4. Wave Packet

无上下边界时，垂向波数 \(k_z\) 连续。局地中尺度涡可写成

```math
\psi(r,\theta,z)
=
\int \hat A(k_z)R_m(r;k_z)e^{im\theta}e^{ik_z z}\,dk_z.
```

其中 \(\hat A(k_z)\) 控制垂向局地化、倾斜尺度和相位关系。若
\(\hat A(k_z)\) 集中在某个 \(k_{z0}\)，则涡接近单一 helical 模态；若
\(\hat A(k_z)\) 为宽谱，则可形成围绕某个深度 \(z_0\) 的局地倾斜涡包。

## 5. 可计算验证量

实现位于 `src/validation/qg_cylindrical_eigenmodel.py`。第一版验证以下量：

- 内部残差：
  ```math
  L\psi+K^2\psi.
  ```
- 外部残差：
  ```math
  L\psi.
  ```
- 边界跳跃：
  ```math
  [\psi]_{r=a},\qquad [\partial_r\psi]_{r=a}.
  ```
- 地转速度：
  ```math
  u_g=-\partial_y\psi,\qquad v_g=\partial_x\psi.
  ```

这些量用于判断一个候选柱涡是否是 QG-consistent 的解析原型，而不是用于
证明完整 Euler 或 primitive-equation 解。
