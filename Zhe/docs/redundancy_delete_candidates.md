# 鍐椾綑涓庡垹闄ゅ€欓€夋竻鍗?

鏈枃鍙粰鍒犻櫎/褰掓。寤鸿锛屼笉琛ㄧず宸茬粡鍒犻櫎浠讳綍鏂囦欢銆?

鎵弿鑼冨洿锛?

```text
G:\EDDY_detection\S-H-I-T-ocean-\Zhe
/root/Verify
```

鏈嶅姟鍣ㄤ笌鏈湴蹇収涓€鑷达細

```text
non-vendor Python files: 120
entrypoints with __main__: 82
root-level Python files: 13
duplicate file names: 15
```

## 1. Root-level 涓?`src/Legacy/Location` 瀹屽叏鐩稿悓鐨勮剼鏈?

杩欎簺鏂囦欢 hash 瀹屽叏涓€鑷达紝鍙互浣滀负绗竴鎵?`redundant-identical` 鍊欓€夈€傚缓璁厛绉诲姩鍒?`legacy/root_entrypoints/identical_snapshots/`锛岀‘璁?1-2 鍛ㄦ棤寮曠敤鍚庡啀鍒犻櫎銆?

| root-level 鏂囦欢 | 瀵瑰簲姝ｅ紡浣嶇疆 | 鍒ゅ畾 |
|---|---|---|
| `build_cmems_climatology.py` | `src/Legacy/Location/build_cmems_climatology.py` | redundant-identical |
| `classify_3d_eddy_shape.py` | `src/Legacy/Location/classify_3d_eddy_shape.py` | redundant-identical |
| `complete_3d_layer_centers.py` | `src/Legacy/Location/complete_3d_layer_centers.py` | redundant-identical |
| `composite_3d_lifecycle.py` | `src/Legacy/Location/composite_3d_lifecycle.py` | redundant-identical |
| `track_3d_objects.py` | `src/Legacy/Location/track_3d_objects.py` | redundant-identical |

鍒犻櫎鍓嶆鏌ワ細

```powershell
rg "build_cmems_climatology.py|classify_3d_eddy_shape.py|complete_3d_layer_centers.py|composite_3d_lifecycle.py|track_3d_objects.py" G:\EDDY_detection\S-H-I-T-ocean-\Zhe
```

濡傛灉鏂囨。鎴栬剼鏈粛鐩存帴璋冪敤 root-level 璺緞锛屽簲鍏堟敼鎴?`python -m src.Legacy.Location.<module>`銆?

## 2. Root-level 涓?`src/Legacy/Location` 鍚屽悕浣嗗凡鍒嗗弶鐨勮剼鏈?

杩欎簺涓嶈兘鐩存帴鍒犻櫎銆傚畠浠悓鍚嶄絾 hash 涓嶅悓锛岄渶瑕佽繘鍏?`diverged-snapshot review`銆?

| root-level 鏂囦欢 | 瀵瑰簲姝ｅ紡浣嶇疆 | 宸紓瑙勬ā | 鍒ゅ畾 |
|---|---|---:|---|
| `run_kuroshio_full_cpu_pipeline.py` | `src/Legacy/Location/run_kuroshio_full_cpu_pipeline.py` | 绾?120 diff lines | redundant-diverged-review |
| `run_streaming_layer_identification.py` | `src/Legacy/Location/run_streaming_layer_identification.py` | 绾?678 diff lines | redundant-diverged-review |
| `run_streaming_shape_pipeline.py` | `src/Legacy/Location/run_streaming_shape_pipeline.py` | 绾?208 diff lines | redundant-diverged-review |
| `streaming_cmems.py` | `src/Legacy/Location/streaming_cmems.py` | 绾?51 diff lines | redundant-diverged-review |
| `table_io.py` | `src/Legacy/Location/table_io.py` | 绾?33 diff lines | redundant-diverged-review |

寤鸿澶勭悊锛?

1. 閫愪釜 diff銆?
2. 濡傛灉 root-level 鍙槸鏃у弬鏁版垨鏃ц矾寰勶紝褰掓。 root-level銆?
3. 濡傛灉 root-level 鏈変粛闇€瑕佺殑閫昏緫锛屾妸閫昏緫鍚堝苟杩?`src/Legacy/Location`銆?
4. 淇濈暀 root-level wrapper 涓€娈垫椂闂达紝杈撳嚭 deprecation warning銆?

## 3. root-only 鏂囦欢

杩欎簺 root-level 鏂囦欢娌℃湁鍚屽悕 `src/Legacy/Location` 瀵瑰簲鐗╋紝涓嶈兘鏍规嵁鍚屽悕瑙勫垯鍒犻櫎锛?

```text
acc_config.py
crop_acc_raw.py
download_acc_raw.py
```

寤鸿鍒嗙被锛?

- `download_acc_raw.py`銆乣crop_acc_raw.py`锛氬簲杩佸叆 `src/data_downloading/` 鎴?`scripts/legacy_data_download/`銆?
- `acc_config.py`锛氱‘璁ゆ槸鍚︿粛琚?root-level ACC workflow 浣跨敤锛涜嫢鍙湇鍔℃棫娴佺▼锛屽綊妗ｃ€?

## 4. 闈?root 鐨勯噸澶嶅懡鍚?

杩欎簺閲嶅鍚嶄笉蹇呯劧鍐椾綑锛屼絾琛ㄧず璇箟杈圭晫鍙兘涓嶆竻锛?

| 鏂囦欢鍚?| 浣嶇疆 | 寤鸿 |
|---|---|---|
| `run_hua_hybrid_detection_acc.py` | `src/Legacy/Location/` 涓?`src/Legacy/experiments/temp/` | 鏄庣‘涓€涓负 production锛屼竴涓负 replication experiment |
| `unified_math.py` | `src/Legacy/Location/validation/` 涓?`src/Legacy/experiments/theory_validation/` | 鍚堝苟鎴栭噸鍛藉悕锛岄伩鍏嶇悊璁哄伐鍏峰弻婧?|
| `cli.py` | `forecast/` 涓?`validation/` | 姝ｅ父閲嶅鍚嶏紝浣嗗簲杩佸嚭 `Location` |
| `common.py` | `Location/common.py` 涓?`forecast/common.py` | 姝ｅ父閲嶅鍚嶏紝浣?forecast 搴旂嫭绔嬪懡鍚嶇┖闂?|

## 5. `.bak_*` 鏂囦欢

褰撳墠瀛樺湪锛?

```text
src/Legacy/First_temp/direction_fit.py.bak_20260720_170713
src/Legacy/Location/run_representative_vortex.py.bak_20260720_170713
```

寤鸿锛?

- 鑻?Git 鍘嗗彶宸茬粡瑕嗙洊杩欎簺鍐呭锛屽綊妗ｅ埌 `legacy/backups/` 鎴栧垹闄ゃ€?
- 鑻?Git 鍘嗗彶涓嶅畬鏁达紝淇濈暀鍒?`legacy/backups/`锛屼笉瑕佺暀鍦ㄦ寮?package 鐩綍銆?

## 6. 瀹為獙鍖哄垹闄ょ瓥鐣?

`src/Legacy/experiments/temp/` 鐩墠鏈夊涓粛鏈夌瀛︿环鍊肩殑鑴氭湰锛屼笉搴旂洿鎺ュ垹闄わ細

```text
run_azimuthal_representative_vortex.py
run_aggregate_product_stirring.py
run_shape_representative_bundle.py
run_hua_boundary_monotonic_rotation_compare.py
run_rotation_core_modal_tilt_validation_li2026.py
plot_original_eddy_discontinuity_7panel.py
```

鍏朵腑鍓嶄笁涓凡缁忔帴杩戠敓浜т富閾撅紝搴旇€冭檻鎻愬崌锛岃€屼笉鏄垹闄ゃ€?

## 7. Vendor 绠＄悊

`vendor/` 绾?587 MB锛屼笉鏄啑浣欎唬鐮併€傚畠鍖呭惈鍙傝€冩垨绗笁鏂瑰疄鐜帮細

```text
vendor/Hybrid-Eddy-detection
vendor/Hybrid-Eddy-detection-main
vendor/MITgcm
vendor/py-eddy-tracker
```

寤鸿锛?

- 涓嶇洿鎺ョ紪杈?vendor銆?
- 鑻?GitHub 浠撳簱涓嶉€傚悎鎻愪氦 587 MB vendor锛屽彲鏀规垚涓嬭浇鑴氭湰銆佸瓙妯″潡鎴?release artifact銆?
- 鑷爺閫傞厤閫昏緫搴旂暀鍦?`src/`锛屼笉瑕佸啓杩?vendor銆?

## 8. 绗竴鎵瑰畨鍏ㄥ姩浣滃缓璁?

绗竴鎵瑰彧鍋氬綊妗ｏ紝涓嶅仛纭垹闄わ細

1. 鏂板缓 `legacy/root_entrypoints/identical_snapshots/`銆?
2. 绉诲叆 5 涓?hash 瀹屽叏鐩稿悓鐨?root-level 鑴氭湰銆?
3. 鏂板缓 `legacy/root_entrypoints/diverged_snapshots/`銆?
4. 澶嶅埗 5 涓垎鍙?root-level 鑴氭湰杩涘幓锛屽師鏂囦欢鍏堜繚鐣?wrapper銆?
5. 鏂板缓 `legacy/backups/`锛岀Щ鍔?`.bak_*` 鏂囦欢銆?

鍙湁鍦?smoke tests 鍜屾枃妗ｈ矾寰勬洿鏂板悗锛屾墠杩涘叆鐪熸鍒犻櫎闃舵銆?

