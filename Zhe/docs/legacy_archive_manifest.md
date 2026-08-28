# Legacy 归档清单

本轮统一历史脚本位置：正式历史区为仓库根下的 `Zhe/legacy/`，不再把 `Zhe/src/Legacy/` 当作可 import 包使用。

## 当前状态

- `Zhe/src/Legacy/`：扫描时为空目录，没有纳入 Git 的文件。
- `Zhe/legacy/diagnostics/`：单次诊断图和原始涡旋图。
- `Zhe/legacy/method_comparison/`：Hua/Nencioli/C++ 方法对比。
- `Zhe/legacy/paper_replication/`：论文复刻、MITgcm、模态验证。

## 迁移规则

- 正式生产识别主链：迁入或保留在 `src.eddy_pipeline`。
- 正式代表涡后处理：迁入或保留在 `src.post`。
- 临时研究入口：保留在 `src.experiments`。
- 历史、论文复刻、一次性图：保留在 `legacy`。

## Wrapper 替代关系

| 旧入口 | 当前真实实现 |
| --- | --- |
| `src/Location/run_representative_stirring_transport.py` | `src.post.transport` |
| `src/Location/run_coherent_stirring_transport.py` | `src.post.transport` |
| `src/experiments/temp/run_aggregate_product_stirring.py` | `src.post.transport` |

