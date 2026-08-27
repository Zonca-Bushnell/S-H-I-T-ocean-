# 代表涡旋热输送与 PV 输送的 stirring 口径

本文记录当前 Kuroshio Hua b3 strict-contiguous coherent-only 代表涡旋后处理采用的热输送与位涡输送口径。当前阶段只研究 **stirring**，暂不计算 trapping。

核心约定是：代表涡旋已经通过 `global_ls_alpha` 旋转对齐，因此代表涡旋只有一个内部坐标系。所有 stirring 输送都在这个旋转坐标中解释，不再同时输出“地理北向”和“代表轴法向”两套方向。

## 1. 当前研究对象

当前权威代表涡旋结果为：

```text
/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/representative_vortex
```

这个结果满足：

- Hua/Nencioli `b3_start2`；
- strict-contiguous 垂向扩展；
- 30-180 天带通速度；
- life30 shape classification；
- coherent-only；
- `global_ls_alpha` 旋转对齐；
- `cyclonic` 与 `anticyclonic` 分开合成。

本轮输送诊断只回答：

> 在所有 coherent 涡旋按 global alpha 对齐后，代表涡旋在其横向方向上是否存在系统性 stirring 热输送与 PV 输送？

本轮不回答：

> 这些涡旋在真实地理坐标中对黑潮区域造成多少净经向输送？

后者需要对象级轨迹和平移速度，是另一个任务。

## 2. 扰动场定义

速度扰动使用 30-180 天带通信号：

$$
u' = u_{30-180d},
\qquad
v' = v_{30-180d}.
$$

热输送需要温度扰动也使用同一滤波定义：

$$
\theta' = \theta_{30-180d}.
$$

因此必须补齐：

```text
/root/autodl-fs/kuroshiou/Filter/global_phy_YYYY_bandpass_30_180d.nc
```

中的 `thetao_glor`，或在后续实现中允许从独立 sidecar 读取：

```text
/root/autodl-fs/kuroshiou/Filter_thetao_fast/global_phy_YYYY_thetao_bandpass_30_180d.nc
```

背景态可在需要时定义为：

$$
u_{\rm bg}=u_{\rm raw}-u_{30-180d},
\qquad
v_{\rm bg}=v_{\rm raw}-v_{30-180d},
\qquad
\theta_{\rm bg}=\theta_{\rm raw}-\theta_{30-180d}.
$$

但 stirring 通量本身使用带通扰动场。

## 3. global-alpha 旋转坐标

coherent-only 代表涡旋使用 `global_ls_alpha` 对齐。每个涡旋对象先估计整体偏移方向，然后旋转到合成坐标的正 \(x_{\rm rot}\) 方向。

物理含义为：

$$
x_{\rm rot}: \text{涡旋整体偏移方向},
$$

$$
y_{\rm rot}: \text{垂直于偏移方向的横向/法向方向}.
$$

采用当前代码的旋转约定：

$$
x_{\rm rot}=x\cos\alpha-y\sin\alpha,
$$

$$
y_{\rm rot}=x\sin\alpha+y\cos\alpha.
$$

速度按同一旋转：

$$
u_{\rm rot}=u\cos\alpha-v\sin\alpha,
$$

$$
v_{\rm rot}=u\sin\alpha+v\cos\alpha.
$$

其中 \(v_{\rm rot}\) 是当前代表涡旋 stirring 诊断唯一使用的横向速度分量。

从 rotated frame 回到地理坐标的公式仍可作为审计信息：

$$
u_{\rm east}=u_{\rm rot}\cos\alpha+v_{\rm rot}\sin\alpha,
$$

$$
v_{\rm north}=-u_{\rm rot}\sin\alpha+v_{\rm rot}\cos\alpha.
$$

但本轮代表涡旋诊断不使用 \(v_{\rm north}\)。地理北向输送只有在研究海盆尺度净经向通量时才启用。

## 4. 热 stirring

在代表涡旋坐标中，热 stirring 定义为：

$$
\boxed{
H^{\rm stir}_{\rm rot}
=
\rho_0 C_p
\left\langle
v'_{\rm rot}\theta'
\right\rangle_\phi
}
$$

其中 \(\langle\cdot\rangle_\phi\) 表示同一 \((\tau,z,r/R)\) 上沿方位角平均。

为了只保留内部非轴对称搅拌，也可以使用环向异常形式：

$$
v'_{\rm rot,rel}
=
v'_{\rm rot}
-
\left\langle v'_{\rm rot}\right\rangle_\phi,
$$

$$
\theta'_{\rm rel}
=
\theta'
-
\left\langle\theta'\right\rangle_\phi,
$$

$$
\boxed{
H^{\rm stir}_{\rm rot,rel}
=
\rho_0 C_p
\left\langle
v'_{\rm rot,rel}\theta'_{\rm rel}
\right\rangle_\phi
}
$$

后续实现应同时保存直接协方差和环向异常协方差：

- `heat_stir_rot = rho0_cp * mean(v_rot * theta)`
- `heat_stir_rot_rel = rho0_cp * mean((v_rot - mean_phi(v_rot)) * (theta - mean_phi(theta)))`

解释时以 `heat_stir_rot_rel` 作为“内部 stirring”主口径，`heat_stir_rot` 作为总协方差参考。

## 5. PV stirring

PV 采用现有 E-P 诊断的 QG-like 口径，而不是 Ertel PV：

$$
q'
=
\nabla_h^2\psi'
+
\frac{\partial}{\partial z}
\left(
\frac{f_0^2}{N^2}
\frac{\partial\psi'}{\partial z}
\right).
$$

其中 \(\psi'\) 由 30-180 天带通速度的相对涡度反演得到：

$$
\zeta'=\partial_x v'-\partial_y u',
\qquad
\nabla_h^2\psi'=\zeta'.
$$

代表涡旋坐标中的 PV stirring 为：

$$
\boxed{
P^{\rm stir}_{\rm rot}
=
\left\langle
v'_{\rm rot}q'
\right\rangle_\phi
}
$$

环向异常形式为：

$$
q'_{\rm rel}
=
q'
-
\left\langle q'\right\rangle_\phi,
$$

$$
\boxed{
P^{\rm stir}_{\rm rot,rel}
=
\left\langle
v'_{\rm rot,rel}q'_{\rm rel}
\right\rangle_\phi
}
$$

后续实现应输出：

- `pv_stir_rot = mean(v_rot * q_prime)`
- `pv_stir_rot_rel = mean((v_rot - mean_phi(v_rot)) * (q_prime - mean_phi(q_prime)))`

当前已有 `ep_flux_terms/continuous_ep_flux_profiles.parquet` 中的 `pv_flux` 更接近 E-P 诊断中的轴法向 PV flux，可作为 sanity check，但不要再把它解释成地理北向输送。

## 6. 为什么本轮不算 trapping

trapping 描述的是涡旋整体平移携带被俘获异常量：

$$
H^{\rm trap}
\sim
\rho_0 C_p
V_c
\left\langle\theta'\right\rangle,
$$

$$
P^{\rm trap}
\sim
V_c
\left\langle q'\right\rangle.
$$

这里的 \(V_c\) 是涡旋轨迹平移速度。若研究真实地理经向输送，需要用对象级 surface center 轨迹估计 \(V_{c,y}\)。若研究代表涡旋内部坐标，则需要定义 \(V_{c,rot}\)。二者都不是当前已经合成的代表涡旋模板本身直接给出的量。

因此本轮先不计算 trapping，避免把代表涡旋内部旋转速度误当作涡旋整体迁移速度。

## 7. 后续计算任务

当前正在补齐 `thetao_glor` 的 30-180 天带通场。完成后，下一步的 stirring 计算应当：

1. 读取 strict coherent representative 的 selected objects 与 global-alpha 轴线；
2. 按日期读取 `u_{30-180d}, v_{30-180d}, theta_{30-180d}`；
3. 在每个对象的 global-alpha rotated frame 中重采样到统一 \((z,r/R,\phi)\) 网格；
4. 计算 \(v_{\rm rot}\)、\(\theta'\)、\(q'\)；
5. 对每个 object-day 先计算乘积 \(v_{\rm rot}\theta'\) 与 \(v_{\rm rot}q'\)；
6. 再按 \(\tau\)、深度、半径、极性合成。

不要保存巨大对象级四维场。只保存聚合后的：

```text
polarity, tau, depth, r_over_R,
heat_stir_rot, heat_stir_rot_rel,
pv_stir_rot, pv_stir_rot_rel,
count, n_objects, n_tracks
```

## 8. 输出建议

建议输出到：

```text
/root/autodl-fs/kuroshiou/result_strict_contiguous/result_coherent_only/stirring_transport
```

核心文件：

```text
stirring_transport_profiles.parquet
stirring_transport_profiles.csv
stirring_transport_core_r15.csv
stirring_transport_core_r25.csv
stirring_transport_summary_zh.md
figures/
```

其中 `core_r15` 表示 \(r/R\le 1.5\)，`core_r25` 表示 \(r/R\le 2.5\)。

## 9. 一句话口径

当前代表涡旋 stirring 诊断只使用 global-alpha 旋转后的横向速度：

$$
\boxed{
H^{\rm stir}_{\rm rot}
=
\rho_0 C_p
\left\langle v'_{\rm rot}\theta'\right\rangle
}
$$

$$
\boxed{
P^{\rm stir}_{\rm rot}
=
\left\langle v'_{\rm rot}q'\right\rangle
}
$$

地理北向 \(v_{\rm north}\) 不进入本轮代表涡旋 stirring 任务。
