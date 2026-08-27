# 冗余与删除候选清单

本文只给删除/归档建议，不表示已经删除任何文件。

扫描范围：

```text
G:\EDDY_detection\S-H-I-T-ocean-\Zhe
/root/Verify
```

服务器与本地快照一致：

```text
non-vendor Python files: 120
entrypoints with __main__: 82
root-level Python files: 13
duplicate file names: 15
```

## 1. Root-level 与 `src/Location` 完全相同的脚本

这些文件 hash 完全一致，可以作为第一批 `redundant-identical` 候选。建议先移动到 `legacy/root_entrypoints/identical_snapshots/`，确认 1-2 周无引用后再删除。

| root-level 文件 | 对应正式位置 | 判定 |
|---|---|---|
| `build_cmems_climatology.py` | `src/Location/build_cmems_climatology.py` | redundant-identical |
| `classify_3d_eddy_shape.py` | `src/Location/classify_3d_eddy_shape.py` | redundant-identical |
| `complete_3d_layer_centers.py` | `src/Location/complete_3d_layer_centers.py` | redundant-identical |
| `composite_3d_lifecycle.py` | `src/Location/composite_3d_lifecycle.py` | redundant-identical |
| `track_3d_objects.py` | `src/Location/track_3d_objects.py` | redundant-identical |

删除前检查：

```powershell
rg "build_cmems_climatology.py|classify_3d_eddy_shape.py|complete_3d_layer_centers.py|composite_3d_lifecycle.py|track_3d_objects.py" G:\EDDY_detection\S-H-I-T-ocean-\Zhe
```

如果文档或脚本仍直接调用 root-level 路径，应先改成 `python -m src.Location.<module>`。

## 2. Root-level 与 `src/Location` 同名但已分叉的脚本

这些不能直接删除。它们同名但 hash 不同，需要进入 `diverged-snapshot review`。

| root-level 文件 | 对应正式位置 | 差异规模 | 判定 |
|---|---|---:|---|
| `run_kuroshio_full_cpu_pipeline.py` | `src/Location/run_kuroshio_full_cpu_pipeline.py` | 约 120 diff lines | redundant-diverged-review |
| `run_streaming_layer_identification.py` | `src/Location/run_streaming_layer_identification.py` | 约 678 diff lines | redundant-diverged-review |
| `run_streaming_shape_pipeline.py` | `src/Location/run_streaming_shape_pipeline.py` | 约 208 diff lines | redundant-diverged-review |
| `streaming_cmems.py` | `src/Location/streaming_cmems.py` | 约 51 diff lines | redundant-diverged-review |
| `table_io.py` | `src/Location/table_io.py` | 约 33 diff lines | redundant-diverged-review |

建议处理：

1. 逐个 diff。
2. 如果 root-level 只是旧参数或旧路径，归档 root-level。
3. 如果 root-level 有仍需要的逻辑，把逻辑合并进 `src/Location`。
4. 保留 root-level wrapper 一段时间，输出 deprecation warning。

## 3. root-only 文件

这些 root-level 文件没有同名 `src/Location` 对应物，不能根据同名规则删除：

```text
acc_config.py
crop_acc_raw.py
download_acc_raw.py
```

建议分类：

- `download_acc_raw.py`、`crop_acc_raw.py`：应迁入 `src/data_downloading/` 或 `scripts/legacy_data_download/`。
- `acc_config.py`：确认是否仍被 root-level ACC workflow 使用；若只服务旧流程，归档。

## 4. 非 root 的重复命名

这些重复名不必然冗余，但表示语义边界可能不清：

| 文件名 | 位置 | 建议 |
|---|---|---|
| `run_hua_hybrid_detection_acc.py` | `src/Location/` 与 `src/experiments/temp/` | 明确一个为 production，一个为 replication experiment |
| `unified_math.py` | `src/Location/validation/` 与 `src/experiments/theory_validation/` | 合并或重命名，避免理论工具双源 |
| `cli.py` | `forecast/` 与 `validation/` | 正常重复名，但应迁出 `Location` |
| `common.py` | `Location/common.py` 与 `forecast/common.py` | 正常重复名，但 forecast 应独立命名空间 |

## 5. `.bak_*` 文件

当前存在：

```text
src/First_temp/direction_fit.py.bak_20260720_170713
src/Location/run_representative_vortex.py.bak_20260720_170713
```

建议：

- 若 Git 历史已经覆盖这些内容，归档到 `legacy/backups/` 或删除。
- 若 Git 历史不完整，保留到 `legacy/backups/`，不要留在正式 package 目录。

## 6. 实验区删除策略

`src/experiments/temp/` 目前有多个仍有科学价值的脚本，不应直接删除：

```text
run_azimuthal_representative_vortex.py
run_aggregate_product_stirring.py
run_shape_representative_bundle.py
run_hua_boundary_monotonic_rotation_compare.py
run_rotation_core_modal_tilt_validation_li2026.py
plot_original_eddy_discontinuity_7panel.py
```

其中前三个已经接近生产主链，应考虑提升，而不是删除。

## 7. Vendor 管理

`vendor/` 约 587 MB，不是冗余代码。它包含参考或第三方实现：

```text
vendor/Hybrid-Eddy-detection
vendor/Hybrid-Eddy-detection-main
vendor/MITgcm
vendor/py-eddy-tracker
```

建议：

- 不直接编辑 vendor。
- 若 GitHub 仓库不适合提交 587 MB vendor，可改成下载脚本、子模块或 release artifact。
- 自研适配逻辑应留在 `src/`，不要写进 vendor。

## 8. 第一批安全动作建议

第一批只做归档，不做硬删除：

1. 新建 `legacy/root_entrypoints/identical_snapshots/`。
2. 移入 5 个 hash 完全相同的 root-level 脚本。
3. 新建 `legacy/root_entrypoints/diverged_snapshots/`。
4. 复制 5 个分叉 root-level 脚本进去，原文件先保留 wrapper。
5. 新建 `legacy/backups/`，移动 `.bak_*` 文件。

只有在 smoke tests 和文档路径更新后，才进入真正删除阶段。

