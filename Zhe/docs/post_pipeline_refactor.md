# Post 后处理主链

本文记录 `src.post` 的定位和迁移规则。识别、tracking、catalog、shape classification 和代表涡结构合成仍由 `src.eddy_pipeline` 负责；`src.post` 只接收代表涡之后的正式后处理。

## 当前正式口径

默认科学口径为：

`hua_b3_start2 + 30-180d bandpass + boundary_monotonic + strict_contiguous + life30 + coherent_only + ME_LIUTEX azimuth_preserved + global_ls_alpha`

默认输入是 `/root/autodl-fs/kuroshiou/result_boundary_monotonic/result_coherent_only`。TURN 是主输送口径；UNTURN 只用于结构对照。

## 模块划分

- `src.post.transport`：aggregate-product stirring，输出 `product_mean`、`mean_product`、`covariance` 和二阶矩。主结论使用乘积后平均与协方差，不使用平均后乘积替代。
- `src.post.structure`：从 ME_LIUTEX 角向代表涡输出标准结构图，支持 `turned`、`unturned`、`both`。
- `src.post.double_core`：从代表涡速度场诊断速度中心和旋转核心分离，输出 `D_omega/R` 表格与热图。
- `src.post.cli`：正式后处理统一入口。

## CLI

```bash
python -m src.post.cli build-transport --shape coherent --orientation turned
python -m src.post.cli plot-structure --shape coherent --orientation both
python -m src.post.cli analyze-double-core --shape coherent --orientation both
python -m src.post.cli run-default --shape coherent --orientation both
```

参数放在子命令后面。`--dry-run` 只打印路径、科学口径和将执行的命令，不写入结果。

## Legacy 边界

`src/Location/run_representative_stirring_transport.py`、`src/Location/run_coherent_stirring_transport.py` 和 `src/experiments/temp/run_aggregate_product_stirring.py` 现在是兼容 wrapper，真实实现位于 `src.post.transport`。

Li2026/MITgcm/Nencioli/Hua 论文复刻、9-panel 单涡间断点审查图、一次性方法对比图仍属于 `legacy/` 或 `src/experiments/`，不进入正式 post 主链。

## 后续清理

`src.post.transport` 仍沿用部分 `src.First_temp` 数值工具以保持结果不变。下一步若要继续工程化，应把 QG/PV、插值和网格工具抽到稳定工具模块，再减少对 `First_temp` 的依赖。

