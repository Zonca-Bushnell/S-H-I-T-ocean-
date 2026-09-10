# 三分 z_rho / 背景密度几何诊断

本诊断入口用于 Original worktree 中的 20N 筛选实验，不改变正式生产口径。

命令核心参数：

```powershell
D:\Util\lever\02_miniforge\envs\Dipole_vertical_transport\python.exe `
  meta4_core_argo_vertical_transport.py `
  --diagnose-threeway-z-background `
  --target-lat 20 `
  --intersect-radius-r 1 `
  --match-mode all `
  --depth-levels 10:10:2000
```

三组定义：

- A：正式 BOA anomaly 口径，term1 和 term2 都使用 `BOA climatology -> z'_rho -> Cressman -> gradient_xy`。
- B：整体等密面口径，term1 和 term2 都使用 `composite rho(x,y,z) -> predecessor_isopycnal_slope -> absolute z_rho slope`。
- C：混合口径，term1 使用涡旋合成 absolute `z_rho`，term2 使用背景密度场的等密面斜率；BOA 背景一定输出，ISAS-derived 前辈背景只在本地存在对应极性文件时输出。

判读重点：

- 如果 B 明显强于 A，说明“整体等密面集合”和 BOA anomaly `z'_rho` 的几何定义会系统性改变 W。
- 如果 C 的深层 term2 明显强于 A，说明 BOA anomaly 可能扣除了进入 term2 的深层背景等密面坡度。
- 如果 ISAS-derived C 强于 BOA C，说明前辈 ISAS 背景可能保留了更强的深层大尺度几何。

符号和坐标：

- W 向上为正。
- 深度和等密面深度显示为正深度向下。
- 矩阵列对应东西向 `x/R`，矩阵行对应南北向 `y/R`；水平梯度统一通过 `gradient_xy`。
