# reference-like 深层反转诊断

本诊断在独立 worktree `S-H-I-T-ocean-zrho-reversal-worktree` 中实现，不改变 main
worktree 的正式生产口径。目标是检验一个具体问题：如果 `triangle z_rho` 不再来自
BOA 背景扣除后的异常等密面起伏，而改为参考程序更接近的“合成总密度场整体等密面斜率”，
`W(x/R, depth)` 是否会更接近参考图中的深层反相结构。

## 可复现入口

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-zrho-reversal-worktree\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --diagnose-reference-like-reversal `
  --target-lat 20 `
  --intersect-radius-r 1 `
  --match-mode all `
  --vertical-mode thermal_wind_depth_stack `
  --depth-levels 10:10:2000 `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_W_3D_20N_reference_like_reversal_worktree"
```

输出只包含 cyclonic 和 anticyclonic，不生成 combined。

## 核心定义

正式 BOA anomaly 口径为：

```text
z'_rho = z_profile(rho_BOA(z0)) - z_BOA(rho_BOA(z0))
W = c_x_rel * d z'_rho / dx - u_rel · grad(z'_rho)
```

reference-like 口径改为先合成总密度场：

```text
rho_abs(x/R, y/R, D)
```

然后用隐式等密面关系计算整体等密面斜率：

```text
rho(x, y, D) = const
dD/dx|rho = -rho_x / rho_D
dD/dy|rho = -rho_y / rho_D
```

其中 `D` 是正深度向下。计算 `W` 时仍采用向上为正：

```text
term1 = c_x_rel * dD/dx|rho
term2 = -[(u_tw - c_x_raw) * dD/dx|rho + v_tw * dD/dy|rho]
W = term1 + term2
```

为避免弱层结处 `rho_D` 太小导致斜率爆炸，本诊断对 `rho_abs` 做水平和垂向平滑，
剔除 `|rho_D| < 5e-5 kg m^-4` 的格点，并按有效斜率 99% 分位封顶。

## 20N 全样本结果

| polarity | matches | unique Argo | valid fraction | q95 W | median corr W/1000m | deep reversal score | first zero depth | slope cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cyclonic | 122621 | 57142 | 0.742 | 34.1 | 0.866 | -0.665 | 55 | 0.000758 |
| anticyclonic | 112942 | 53953 | 0.742 | 44.7 | 0.482 | -0.287 | 55 | 0.00109 |

`q95 W` 单位为 `10^-6 m s^-1`。

## 诊断结论

1. 改用整体等密面斜率后，20N cyclonic 剖面已经出现类似参考图的浅层/中层东西偶极，
   并在约 1200-1500 m 以下形成明显深层反相结构。
2. anticyclonic 也出现深层反相，但深层更块状，说明整体等密面斜率是必要因素，
   但深层规整程度还受样本支撑、弱层结阈值、深层滤波和边界 masking 影响。
3. `term1` 与 `term2` 的剖面都跟随整体 `dD/dx|rho` 出现深层相位改变；
   因此反转不是单纯由热成风速度或 term2 符号制造，而是首先由 `triangle z_rho`
   的几何定义触发。
4. 从耗时看，reference-like 斜率本身不到 1 秒；主要耗时仍是首次 profile/BOA QC
   cache 构建。已把 profile cache key 改为包含唯一 Argo 数和索引摘要，避免 smoke
   与 full run 互相覆盖缓存。

## 下一步可调参数

优先调这些参数，而不是继续改匹配：

- `rho_abs` 水平平滑次数。
- `rho_abs` 垂向平滑次数。
- 弱层结阈值 `|rho_D|`。
- 斜率封顶分位数。
- 深层边缘 masking 或按支撑数加权的 section 取样。

当前判断：如果目标是复现参考图的深层反转，最关键的改动是将正式生产的
`triangle z_rho` 从 BOA anomaly geometry 切换或扩展为合成总密度场的整体等密面斜率诊断。
