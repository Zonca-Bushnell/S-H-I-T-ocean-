# EP-flux 20N 数值验证入口

本目录新增 `epflux_20n_validation.py` 和 MATLAB 后端 `matlab/run_epflux_20n_validation_backend.m`。它们用于第一阶段验证材料体 EP-flux 推广中，垂直热输运诊断量 `W T'` 与水平动量通量强迫代理量之间是否存在稳定关系。

## 当前口径

- 输入使用已有 `20N crossing / 1R / match-mode all / recommended` 三维热成风 W 产品。
- 只处理 `cyclonic` 和 `anticyclonic`，不生成 combined。
- `W` 采用向上为正。
- 温度异常采用线性 EOS：`T' = -rho'/(rho_ref alpha)`。
- 当前仅计算 `A_x^R(partial) = -partial_y R_xy + partial_D R_xz`，因为现有三维产品尚未保存 `R_xx` 或足够的样本级速度扰动来构造完整 `partial_x R_xx`。

## 重要限制

本入口不会把 `W T'` 直接当成传统 EP flux 的垂向分量。它输出的是局地代理预算：

```text
R_xz = u' W
W T' = W * T'
A_x^R(partial) = -partial_y(u'v') + partial_D(u'W)
residual_proxy = A_x^R(partial) - beta W T'
```

这里的 `residual_proxy` 不是完整理论中的 `G_x - A_x^R - L_x^B`，因为 `G_x` 和非局地 QG/PV 响应 `L_x^B` 尚未定义和求解。

## 运行

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-EPFLUX-20N-worktree\EP-FLUX\epflux_20n_validation.py" `
  --depth-levels 10:10:2000 `
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\EP_FLUX_20N_validation"
```
