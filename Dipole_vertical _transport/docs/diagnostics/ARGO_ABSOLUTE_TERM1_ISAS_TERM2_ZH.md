# Argo absolute term1 + ISAS term2 诊断口径

本诊断用于检验一个更接近前辈程序的混合几何口径：

- `term1` 使用 Argo 合成 absolute density 场，经整体等密面集合反插得到
  `dD_rho/dx`。
- 热成风速度以 1000 m Argo parking drift 为锚点，使用 Argo 合成
  absolute density 的水平密度梯度积分得到。
- `term2` 使用 ISAS/background density 场，经同样的整体等密面集合反插得到
  背景 `dD_rho/dx, dD_rho/dy`。
- `D_rho` 为正深度向下；`W` 为向上为正。
- 只输出 cyclonic 和 anticyclonic，不生成 combined。

该入口是诊断/筛选口径，不替代当前正式默认流程。

## 命令

```powershell
D:\Util\lever\02_miniforge\envs\Dipole_vertical_transport\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\Original_Dipole_vertical _transport\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --run-argo-absolute-term1-isas-term2 `
  --crossing-lats 20,30,40 `
  --intersect-radius-r 1 `
  --match-mode all `
  --vertical-mode thermal_wind_depth_stack `
  --depth-levels 10:10:2000 `
  --workers 8 `
  --output-root "E:\DATA\01_Eddy_correspond\05_Original_Dipole_vertical _transport\ISAS_term2_crossing_20N30N40N"
```

## 与 BOA anomaly 流程的差异

本入口不再逐 profile 计算 BOA 目标密度/QC，也不再生成
`D'_rho = D_rho(profile) - D_rho(BOA)`。它只把 Argo profile density 直接插值到
目标深度层并合成 absolute density，再用前辈式等密面斜率算法计算 `term1`。

ISAS 背景只进入 `term2` 的背景等密面斜率。若找不到极性专属
ISAS 文件，背景场会回退到用户提供的通用 ISAS density MAT；这是因为背景场本身
不是涡旋极性量。

## 文献约束

Zhang et al. (2024) 将三维 density、pressure anomaly 和 geostrophic currents
作为涡旋三维结构的一组共同重建量，说明热成风/地转速度剪切应优先来自涡旋坐标下的
三维合成密度或压力结构，而不是仅由外部背景场替代。Chaigneau et al. (2011) 的
Argo-eddy composite 路线同样是先围绕涡旋中心构造三维温盐/密度结构，再解释涡旋的
垂向结构。由此，本入口将热成风速度来源固定为 Argo composite absolute density；
ISAS/BOA 这类背景场只用于定义 anomaly、环境态或 term2 的背景等密面斜率诊断。

后续遇到难以决断的口径问题时，必须先查文献并把依据写入
`docs/literature/` 或本诊断文档，再决定是否改变默认生产流程。

## 20N/30N/40N 结果摘要

输出位置：

`E:\DATA\01_Eddy_correspond\05_Original_Dipole_vertical _transport\ISAS_term2_crossing_20N30N40N`

本轮六组结果均生成有效 ISAS 背景场。改为 Argo composite absolute density
积分热成风后，`term2` 的量级明显增强；整体 `W` 仍受 `term1` 控制，但
深层 term2 已经不再是可忽略的小项：

| 极性 | 纬线 | match | unique Argo | q95 term1 | q95 term2 | q95 W |
|---|---:|---:|---:|---:|---:|---:|
| cyclonic | 20N | 122621 | 57142 | 123.486 | 43.0505 | 142.142 |
| cyclonic | 30N | 92791 | 43188 | 106.132 | 40.7792 | 121.857 |
| cyclonic | 40N | 31421 | 15844 | 178.564 | 76.6879 | 192.441 |
| anticyclonic | 20N | 112942 | 53953 | 133.915 | 34.5685 | 143.287 |
| anticyclonic | 30N | 100543 | 47886 | 47.1039 | 26.8265 | 59.6055 |
| anticyclonic | 40N | 52707 | 24857 | 130.31 | 50.4461 | 142.037 |

单位均为 `10^-6 m s^-1`。深层 `term2` 的 q95 约为 `44-136 x 10^-6 m s^-1`，
说明热成风速度来源是一级敏感项。当前结论是：速度剪切应从 Argo 合成 absolute
density 积分；ISAS 背景几何对 term2 仍有诊断意义，但不应替代涡旋速度剪切来源。

## 性能修正

为避免 MATLAB 自动打开 process pool 并复制大矩阵，`maybe_start_parallel_pool`
现在强制使用 threads pool；如果已有 process pool，会关闭后重开 threads pool。
`predecessor_isopycnal_slope` 也改为通过同一个受控入口启动并行。

另外，本入口新增 absolute-only 3D builder，不再复用 BOA anomaly 的
profile-depth cache。20N/30N/40N 全深度运行中，200 层 Argo 密度插值通常小于
1 秒，200 层 Cressman 约 4 秒，明显快于旧的 BOA/QC 逐 profile 反插路径。
