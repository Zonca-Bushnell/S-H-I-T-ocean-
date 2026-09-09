# 20N W 深层反转三因素诊断说明

本诊断用于比较当前 `ArgoData_SA_CT_PT_PDen_sigma.mat` 三维 W 管线与前辈 MATLAB 程序的主要差异。诊断入口为：

```powershell
D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --diagnose-reversal-factors `
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
  --output-root "E:\DATA\01_Eddy_correspond\01_Vertical_asymmetric\META4_CoreArgo_W_3D_20N_reversal_factor_diagnosis"
```

## 对照逻辑

诊断固定使用 `20N crossing / 1R / match-mode all / recommended`，不生成 combined，不改变默认正式结果。它先复用正式 3D 管线得到当前结果，再额外构造三类替代口径：

- `composite rho -> isopycnal slope`：按前辈代码的思路，先合成三维密度场 `rho(x,y,z)`，再在左右/南北相邻格点的整条密度剖面中反插中心格点密度，得到 `dz_rho/dx`、`dz_rho/dy`。
- `composite density thermal wind`：用合成密度场的水平梯度积分热成风剪切，并以 1000 m 的 `U1000/V1000` composite 作为锚点。
- `predecessor-like term2`：尽量接近前辈程序的组合方式，使用 `term1 = c0 * dzdx_up` 与 `term2 = U_thw * dzdx_up + V_thw * dzdy_up`。

## 8 组因子

输出的 `reversal_factor_terms.mat` 保存 8 组全因子结果；四联图只展示最高信息量的 4 组：

- `A_current`：当前 BOA `z'_rho` 梯度、当前 `rho_anom` 热成风、当前相对速度 term2。
- `B_comp_isoslope`：只替换为前辈式合成密度等密面斜率。
- `C_comp_isoslope_comp_tw`：前辈式等密面斜率 + 合成密度热成风。
- `D_predecessor_like`：前辈式等密面斜率 + 合成密度热成风 + 前辈式 term2。

其余 4 组用于分离“当前斜率/前辈式 term2”“当前斜率/合成密度热成风”等交叉影响。

## 当前 20N 诊断结论

在 20N 结果中，当前方法的 `median corr(W(z), W(1000m))` 约为 `0.96`，说明 W 垂向相位高度一致；只替换为前辈式等密面斜率后，cyclonic 降到约 `0.81`，anticyclonic 降到约 `0.50`。因此，造成我们与前辈深层反转差异的最大因素是 `triangle z_rho` 的定义：当前方法对 `z'_rho = z_profile - z_BOA_bg` 求梯度，而前辈方法从合成密度场本身反插等密面几何斜率。

热成风速度口径和 term2 组合方式会改变量级与局部深层斑块，但在 20N 诊断中没有单独把当前平滑同相结构转化为规整的深层反相结构。
