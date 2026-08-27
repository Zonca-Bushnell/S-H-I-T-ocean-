# Kuroshiou 代表涡旋 stirring 输送任务口径

## 当前任务

本任务只研究 strict-contiguous coherent-only 代表涡旋的 stirring 输送。当前不计算 trapping，不计算真实地理北向净输送。

输入代表涡旋：

```text
/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/representative_vortex
```

输出建议：

```text
/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/stirring_transport
```

## 方向口径

代表涡旋已经按 `global_ls_alpha` 对齐，所以只有一个合成坐标系：

$$
x_{\rm rot}: \text{整体偏移方向},
\qquad
y_{\rm rot}: \text{横向/法向方向}.
$$

本任务所有 stirring 通量只使用：

$$
v'_{\rm rot}
$$

也就是 \(y_{\rm rot}\) 方向速度。

不输出、不解释：

$$
v_{\rm north}
$$

地理北向输送留给未来对象级海盆净输送任务。

## 基础场

速度扰动：

$$
u'=u_{30-180d},
\qquad
v'=v_{30-180d}.
$$

温度扰动：

$$
\theta'=\theta_{30-180d}.
$$

当前正在补齐 `thetao_glor` 的 30-180 天带通场。完成后，热 stirring 才能严格计算。

PV 使用现有 QG-like 口径：

$$
q'
=
\nabla_h^2\psi'
+
\partial_z
\left(
\frac{f_0^2}{N^2}
\partial_z\psi'
\right),
$$

其中 \(\psi'\) 从 30-180 天带通速度反演。

## Stirring 公式

热 stirring：

$$
H^{\rm stir}_{\rm rot}
=
\rho_0 C_p
\left\langle
v'_{\rm rot}\theta'
\right\rangle_\phi.
$$

内部搅拌主口径采用环向异常协方差：

$$
H^{\rm stir}_{\rm rot,rel}
=
\rho_0 C_p
\left\langle
\left(v'_{\rm rot}-\langle v'_{\rm rot}\rangle_\phi\right)
\left(\theta'-\langle\theta'\rangle_\phi\right)
\right\rangle_\phi.
$$

PV stirring：

$$
P^{\rm stir}_{\rm rot}
=
\left\langle
v'_{\rm rot}q'
\right\rangle_\phi.
$$

内部搅拌主口径采用：

$$
P^{\rm stir}_{\rm rot,rel}
=
\left\langle
\left(v'_{\rm rot}-\langle v'_{\rm rot}\rangle_\phi\right)
\left(q'-\langle q'\rangle_\phi\right)
\right\rangle_\phi.
$$

## 需要输出的最小表

聚合粒度：

```text
polarity, tau, depth, r_over_R
```

核心列：

```text
heat_stir_rot
heat_stir_rot_rel
pv_stir_rot
pv_stir_rot_rel
count
n_objects
n_tracks
```

核心区统计：

```text
r/R <= 1.5
r/R <= 2.5
```

## 明确不做

- 不计算 trapping。
- 不估计涡心平移速度 \(V_c\)。
- 不输出地理北向 \(v_{\rm north}\) 通量。
- 不生成 `input_daily/`。
- 不把已有 `pv_flux` 解释为真实地理经向输送。
