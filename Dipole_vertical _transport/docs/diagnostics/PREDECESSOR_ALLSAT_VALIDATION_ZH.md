# META3.2 allsat 前辈算法验证入口

## 目的

该诊断入口用于验证：在原始 META3.2 `twosat` 轨迹文件缺失时，使用本地 META3.2 `allsat` NetCDF 替代轨迹源，前辈式 W 重建算法是否仍能产生深层反转和较规整剖面。

## 运行口径

- 入口参数：`--run-predecessor-allsat-validation`
- META 源：`F:\Eddy\Eddy\META3.2_DT_allsat`
- 抽样方式：按 `track` 对每条涡旋生命史抽取前辈式七阶段快照。
- 默认验证：`20N crossing / 1R / match-mode all`，cyclonic 与 anticyclonic 分开，不生成 combined。
- term1：Argo composite absolute density 的整体等密面斜率。
- 热成风：由 Argo composite absolute density 积分，并以 1000 m parking drift 锚定。
- term2：ISAS background density 的整体等密面斜率。

## 解释边界

该入口是 allsat 替代验证，不是原始 `twosat` 严格复现。若某一极性缺少对应 ISAS 背景 MAT，会在输出 metadata 中以 `isas_strict_polarity_match=false` 标注。

## 标准命令

```powershell
D:\Util\lever\02_miniforge\envs\Dipole_vertical_transport\python.exe `
  "D:\01_Eddy\01_Vertical_asymmetric\Original_Dipole_vertical _transport\Dipole_vertical _transport\meta4_core_argo_vertical_transport.py" `
  --run-predecessor-allsat-validation `
  --target-lat 20 `
  --intersect-radius-r 1 `
  --bbox 0,360,-60,60 `
  --depth-levels 10:10:2000 `
  --match-mode all `
  --output-root "E:\DATA\01_Eddy_correspond\05_Original_Dipole_vertical _transport\predecessor_algorithm_META32_allsat_validation"
```
