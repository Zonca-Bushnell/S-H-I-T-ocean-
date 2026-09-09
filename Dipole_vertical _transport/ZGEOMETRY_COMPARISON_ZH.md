# 20N 等密面几何口径对照

本诊断用于回答一个很具体的问题：如果正式 `rebuild W` 里的 `z_\rho` 不再使用 BOA 背景下的相对异常几何，而改成前辈程序中的“合成密度场整体等密面集合”，深层反转是否会出现。

新增入口参数：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --compare-z-geometry-modes `
  --selection-mode crossing_lat `
  --target-lat 20 `
  --intersect-radius-r 1 `
  --match-mode all `
  --vertical-mode thermal_wind_depth_stack `
  --depth-levels 10:10:2000 `
  --grid-n 61 `
  --cressman-radius-r 1.0 `
  --cressman-min-obs 8 `
  --smooth-passes 4 `
  --compute-device auto `
  --workers 8 `
  --no-grid-nc `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_W_3D_20N_zgeometry_comparison"
```

输出只包含 cyclonic 和 anticyclonic，不生成 combined。每个极性输出：

- `zgeometry_comparison_terms.mat`
- `zgeometry_w_4panel.png`
- `ZGEOMETRY_COMPARISON_ZH.md`

四联图定义：

| 面板 | 口径 |
|---|---|
| A | 当前正式口径：`BOA z'_\rho -> composite z'_\rho -> gradient -> W` |
| B | 只替换几何：`composite rho(x,y,z) -> z_\rho isosurface -> gradient -> W` |
| C | B 的整体等密面几何 + 用 composite density 梯度积分热成风速度 |
| D | 尽量接近前辈：整体等密面几何 + composite-density 热成风 + 绝对速度形式 `term2` |

关键差异是 `∇z_\rho` 的定义。当前口径先对每条 Argo 计算相对于 BOA 多年同月局地背景的 `z'_\rho`，合成的是异常几何；前辈式口径先合成绝对密度场，再在合成场中用中心格点密度 `rho0 = rho_comp(x,y,z0)` 到邻近柱反插整体等密面深度：

```text
dzrho_dx = [zrho(x+dx, y, rho0) - zrho(x-dx, y, rho0)] / (2 dx)
dzrho_dy = [zrho(x, y+dy, rho0) - zrho(x, y-dy, rho0)] / (2 dy)
```

判读规则：

- 如果 B 相对 A 明显出现深层反转，主因就是等密面几何从 BOA 相对异常换成整体等密面集合。
- 如果 B 不反而 C 反，主因更偏向热成风速度口径。
- 如果 C 不反而 D 反，主因更偏向 `term2` 的速度、斜率和符号组合。
- 如果 D 仍不出现前辈式反转，则差异更可能来自输入密度场、样本范围、平滑/插值细节或前辈合成密度场中包含的背景结构。

本轮 20N 正式测试的初步结论是：整体等密面模式显著放大 W 并引入深层块状结构，但没有稳定复现前辈示例中的清晰深层反相偶极，因此“几何定义”是强敏感因子，但不是唯一解释。
