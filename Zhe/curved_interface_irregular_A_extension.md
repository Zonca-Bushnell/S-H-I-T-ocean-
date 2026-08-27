# 不规则等密面 revised A model 扩展

## 1. 从单值图形面到隐式等密面

旧版 revised A 中的 \(z=Z_i(x,y,t)\) 只应理解为小坡度退化极限。实际三维涡旋中的等密面并不必然是单值图形面，更不应近似为球面。更一般地，第 \(i\) 个等密面定义为隐式曲面

```math
S_i(t)=\{\mathbf x:\sigma_0(\mathbf x,t)=\rho_i\},
```

其中 \(\sigma_0\) 是位势密度，\(\rho_i\) 是目标密度值。单位法向量为

```math
\mathbf n_i=\frac{\nabla\sigma_0}{|\nabla\sigma_0|},
```

切平面投影算子为

```math
\mathbf P_i=\mathbf I-\mathbf n_i\mathbf n_i^{\mathsf T}.
```

因此标量 \(\psi_i\) 在不规则等密面上的切向梯度与曲面 Laplacian 为

```math
\nabla_{S_i}\psi_i=\mathbf P_i\nabla\psi_i,
\qquad
\Delta_{S_i}\psi_i=\nabla_{S_i}\cdot\nabla_{S_i}\psi_i.
```

## 2. 不规则等密面 QGPV

标准固定深度 QGPV 写作

```math
q_i^0
=
f_0+\beta y
+
\frac{1}{f_0}\nabla_h^2\psi_i
+
\mathcal S_i(\psi_{i-1},\psi_i,\psi_{i+1}),
```

其中 \(\mathcal S_i\) 是层间斜压耦合项。不规则等密面 revised A 写作

```math
q_i^A
=
f_{n,i}
+
\frac{1}{f_0}\Delta_{S_i}\psi_i
+
\mathcal S_i(\psi_{i-1},\psi_i,\psi_{i+1}),
```

其中

```math
f_{n,i}=2\boldsymbol\Omega\cdot\mathbf n_i
```

是行星涡度在等密面法向上的投影。因此 A 修正为

```math
\delta q_i^A
=
\frac{1}{f_0}(\Delta_{S_i}-\nabla_h^2)\psi_i
+
(f_{n,i}-f_0)
+
\delta\mathcal S_i.
```

当 \(S_i\) 可写为 \(z=Z_i(x,y,t)\)，且 \(|\nabla Z_i|\ll1\) 时，上式退化为小坡度 Laplace--Beltrami 展开。

## 3. 等密面凹凸与涡心偏移

等密面的平均曲率定义为

```math
H_i=\frac12\nabla\cdot\mathbf n_i.
```

在倾斜对齐坐标中，左右凹凸不对称定义为

```math
\Delta H_i
=
\langle H_i\rangle_{x/R>0}
-
\langle H_i\rangle_{x/R<0}.
```

每层涡心为

```math
\mathbf r_{c,i}(t)=(x_{c,i},y_{c,i}),
```

相对表层偏移为

```math
\Delta\mathbf r_i(t)=\mathbf r_{c,i}(t)-\mathbf r_{c,0}(t).
```

归一化偏移与相邻层偏移为

```math
TD_i^\ast=\frac{|\Delta\mathbf r_i|}{R_i},
\qquad
TD_{i,i-1}^\ast=
\frac{|\mathbf r_{c,i}-\mathbf r_{c,i-1}|}{\bar R_{i,i-1}}.
```

若对称密度异常为 \(\sigma'_{s,i}\)，小偏移展开给出

```math
\sigma'_i(\mathbf x-\Delta\mathbf r_i)
\approx
\sigma'_{s,i}(\mathbf x)
-
\Delta\mathbf r_i\cdot\nabla_h\sigma'_{s,i}.
```

因此密度奇分量为

```math
\sigma'_{\mathrm{odd},i}
\approx
-
\Delta\mathbf r_i\cdot\nabla_h\sigma'_{s,i}.
```

等密面位移近似为

```math
\eta_{\rho,i}\approx-\frac{\sigma'_i}{\partial_z\sigma_0},
```

所以

```math
\eta_{\rho,\mathrm{odd},i}
\approx
\frac{\Delta\mathbf r_i\cdot\nabla_h\sigma'_{s,i}}
{\partial_z\sigma_0}.
```

## 4. 固定深度速度剖面响应

固定深度剖面中的速度响应由热成风关系连接到密度异常：

```math
\frac{\partial u_g'}{\partial z}
=
-
\frac{g}{\rho_0 f_0}
\frac{\partial\sigma_0'}{\partial y},
\qquad
\frac{\partial v_g'}{\partial z}
=
\frac{g}{\rho_0 f_0}
\frac{\partial\sigma_0'}{\partial x}.
```

因此 `YZ` 剖面优先观察 \(u'\) 的正负偶极，`XZ` 剖面优先观察 \(v'\) 的正负偶极。若等密面凹凸左右不一致，速度偶极应表现为一侧更强、一侧更厚或垂向延伸不同。

最终可检验链条为

```math
\Delta\mathbf r_i,\ TD_i^\ast,\ TD_{i,i-1}^\ast
\rightarrow
\sigma'_{\mathrm{odd},i}
\rightarrow
\eta_{\rho,\mathrm{odd},i},\ H_i,\ \Delta H_i
\rightarrow
\delta q_i^A
\rightarrow
\partial_z u_g',\ \partial_z v_g'.
```

若 coherent 类型中 \(\Delta H_i\)、\(TD_i^\ast\)、\(\eta_{\rho,\mathrm{odd},i}\) 与速度偶极不对称同步增强，而 upright-like 中不存在同样增强，则说明 revised A 的不规则等密面几何更适合描述倾斜涡旋。若该链条主要改善 \(\eta_\rho\)、\(\sigma_0\) 或 PV，而不显著改善完整速度场，则 revised A 应解释为等密面/PV 几何理论，而不是完整速度预报模型。
