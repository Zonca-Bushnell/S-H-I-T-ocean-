# Post 鍚庡鐞嗕富閾?

鏈枃璁板綍 `src.post` 鐨勫畾浣嶅拰杩佺Щ瑙勫垯銆傝瘑鍒€乼racking銆乧atalog銆乻hape classification 鍜屼唬琛ㄦ丁缁撴瀯鍚堟垚浠嶇敱 `src.eddy_pipeline` 璐熻矗锛沗src.post` 鍙帴鏀朵唬琛ㄦ丁涔嬪悗鐨勬寮忓悗澶勭悊銆?

## 褰撳墠姝ｅ紡鍙ｅ緞

榛樿绉戝鍙ｅ緞涓猴細

`hua_b3_start2 + 30-180d bandpass + boundary_monotonic + strict_contiguous + life30 + coherent_only + ME_LIUTEX azimuth_preserved + global_ls_alpha`

榛樿杈撳叆鏄?`/root/autodl-fs/kuroshiou/result_boundary_monotonic/result_coherent_only`銆俆URN 鏄富杈撻€佸彛寰勶紱UNTURN 鍙敤浜庣粨鏋勫鐓с€?

## 妯″潡鍒掑垎

- `src.post.transport`锛歛ggregate-product stirring锛岃緭鍑?`product_mean`銆乣mean_product`銆乣covariance` 鍜屼簩闃剁煩銆備富缁撹浣跨敤涔樼Н鍚庡钩鍧囦笌鍗忔柟宸紝涓嶄娇鐢ㄥ钩鍧囧悗涔樼Н鏇夸唬銆?
- `src.post.structure`锛氫粠 ME_LIUTEX 瑙掑悜浠ｈ〃娑¤緭鍑烘爣鍑嗙粨鏋勫浘锛屾敮鎸?`turned`銆乣unturned`銆乣both`銆?
- `src.post.double_core`锛氫粠浠ｈ〃娑￠€熷害鍦鸿瘖鏂€熷害涓績鍜屾棆杞牳蹇冨垎绂伙紝杈撳嚭 `D_omega/R` 琛ㄦ牸涓庣儹鍥俱€?
- `src.post.cli`锛氭寮忓悗澶勭悊缁熶竴鍏ュ彛銆?

## CLI

```bash
python -m src.post.cli build-transport --shape coherent --orientation turned
python -m src.post.cli plot-structure --shape coherent --orientation both
python -m src.post.cli analyze-double-core --shape coherent --orientation both
python -m src.post.cli run-default --shape coherent --orientation both
```

鍙傛暟鏀惧湪瀛愬懡浠ゅ悗闈€俙--dry-run` 鍙墦鍗拌矾寰勩€佺瀛﹀彛寰勫拰灏嗘墽琛岀殑鍛戒护锛屼笉鍐欏叆缁撴灉銆?

## Legacy 杈圭晫

`src/Legacy/Location/run_representative_stirring_transport.py`銆乣src/Legacy/Location/run_coherent_stirring_transport.py` 鍜?`src/Legacy/experiments/temp/run_aggregate_product_stirring.py` 鐜板湪鏄吋瀹?wrapper锛岀湡瀹炲疄鐜颁綅浜?`src.post.transport`銆?

Li2026/MITgcm/Nencioli/Hua 璁烘枃澶嶅埢銆?-panel 鍗曟丁闂存柇鐐瑰鏌ュ浘銆佷竴娆℃€ф柟娉曞姣斿浘浠嶅睘浜?`legacy/` 鎴?`src/Legacy/experiments/`锛屼笉杩涘叆姝ｅ紡 post 涓婚摼銆?
## 鍚庣画娓呯悊

`src.post.transport` 浠嶆部鐢ㄩ儴鍒?`src.Legacy.First_temp` 鏁板€煎伐鍏蜂互淇濇寔缁撴灉涓嶅彉銆備笅涓€姝ヨ嫢瑕佺户缁伐绋嬪寲锛屽簲鎶?QG/PV銆佹彃鍊煎拰缃戞牸宸ュ叿鎶藉埌绋冲畾宸ュ叿妯″潡锛屽啀鍑忓皯瀵?`First_temp` 鐨勪緷璧栥€?

## 原始涡旋个例图入口

`src.post.original_eddy_panels` 是正式后处理中的原始涡旋间断点审查图入口，负责生成 9-panel/扩展 panel 图，包括：

- 垂向中心偏移 `delta x(z)`、`delta y(z)`；
- 第一、第二间断点上下层水平速度场与地转压强代理；
- Omega-w、`dW/dz` 或 `u_perp` 剖面诊断；
- 生命周期轨迹图。

正式调用方式：

```bash
python -m src.post.cli plot-original-eddy-panels --output-dir <output>
python -m src.post.original_eddy_panels --output-dir <output>
```

`Zhe/legacy/diagnostics/plot_original_eddy_discontinuity_7panel.py` 和
`Zhe/legacy/diagnostics/plot_original_eddy_discontinuity_9panel.py`
现在只保留为兼容 wrapper，真实实现不再放在 `legacy`。

## 1/24° refined center 对 post 的影响

post 不重新判断涡旋中心，只消费 catalog 中的生产中心：

- `layer_centers_completed.longitude/latitude` 是当前生产中心。
- 在新 `result_boundary_monotonic_subgrid_1_24deg` 口径下，这两个字段来自局地 1/24° refined velocity center。
- `longitude_grid/latitude_grid` 保留旧 1/4°格点中心，只用于审计“格点锁定”是否造成看起来竖直或跳变的中心线。
- `plot-original-eddy-panels` 增加 `--show-grid-centers`，可在 `delta x(z)`、`delta y(z)` 面板叠加旧格点中心线；默认关闭。

因此，shape、representative、transport、double-core 和原始涡旋 panel 图在读取新 catalog 时会自动使用 refined 中心，不需要在 post 阶段再次插值。

