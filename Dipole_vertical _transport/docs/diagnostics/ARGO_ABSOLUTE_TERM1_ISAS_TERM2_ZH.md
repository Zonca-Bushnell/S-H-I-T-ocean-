# Argo absolute term1 + ISAS term2 诊断口径

本诊断用于检验一个更接近前辈程序的混合几何口径：

- `term1` 使用 Argo 合成 absolute density 场，经整体等密面集合反插得到
  `dD_rho/dx`。
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

ISAS 背景只进入 `term2` 的背景等密面斜率和背景热成风速度。若找不到极性专属
ISAS 文件，背景场会回退到用户提供的通用 ISAS density MAT；这是因为背景场本身
不是涡旋极性量。

## 20N/30N/40N 结果摘要

输出位置：

`E:\DATA\01_Eddy_correspond\05_Original_Dipole_vertical _transport\ISAS_term2_crossing_20N30N40N`

本轮六组结果均生成有效 ISAS 背景场。主要诊断量显示，`term2` 的量级明显小于
`term1`：

| 极性 | 纬线 | match | unique Argo | q95 term1 | q95 term2 | q95 W |
|---|---:|---:|---:|---:|---:|---:|
| cyclonic | 20N | 122621 | 57142 | 123.486 | 7.08393 | 123.296 |
| cyclonic | 30N | 92791 | 43188 | 106.132 | 10.0232 | 106.057 |
| cyclonic | 40N | 31421 | 15844 | 178.564 | 20.1968 | 178.253 |
| anticyclonic | 20N | 112942 | 53953 | 133.915 | 5.1011 | 133.642 |
| anticyclonic | 30N | 100543 | 47886 | 47.1039 | 9.91714 | 49.4346 |
| anticyclonic | 40N | 52707 | 24857 | 130.31 | 17.2746 | 134.503 |

单位均为 `10^-6 m s^-1`。因此，在该混合口径下，ISAS 背景 `term2`
能够提供深层结构，但整体 `W` 仍主要受 Argo absolute density 几何给出的
`term1` 控制。

## 性能修正

为避免 MATLAB 自动打开 process pool 并复制大矩阵，`maybe_start_parallel_pool`
现在强制使用 threads pool；如果已有 process pool，会关闭后重开 threads pool。
`predecessor_isopycnal_slope` 也改为通过同一个受控入口启动并行。

另外，本入口新增 absolute-only 3D builder，不再复用 BOA anomaly 的
profile-depth cache。20N/30N/40N 全深度运行中，200 层 Argo 密度插值通常小于
1 秒，200 层 Cressman 约 4 秒，明显快于旧的 BOA/QC 逐 profile 反插路径。
