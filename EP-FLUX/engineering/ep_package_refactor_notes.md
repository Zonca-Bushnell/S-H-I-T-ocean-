# EP 包自洽化重构记录

生成时间：2026-09-05

本文记录 `Zhe/src/EP` 的第一轮自洽化改造。目标是让 EP 理论诊断包可以独立维护，不再从 `src.post` 或历史 `src.utils` 中反向借工具。

## 目标边界

`src.EP` 负责：

- 经典 EP、倾斜坐标 EP、曲管尺度审计；
- material-volume / core-shell / object-level EP 诊断；
- 代表涡 axis source 的读取与必要持久化；
- EP 诊断所需的 object-day aggregate-product moments；
- EP 自己的 NetCDF、CSV、JSON、NPZ 读取写出和数值函数。

`src.EP` 不负责：

- Hua 识别、tracking、shape classification；
- 代表涡结构合成；
- panel family 图片审查；
- `src.post` 中面向展示的图像逻辑。

## 新增内部基础层

| 模块 | 职责 | 替代关系 |
| --- | --- | --- |
| `src.EP.io` | Filter day 读取、变量清洗、JSON/CSV/parquet 写出 | 替代 EP 内散落的 NetCDF 读写 |
| `src.EP.numerics` | 网格距离、相对涡度、流函数反演、双线性采样、N2 读取、径向/垂向/方位导数 | 替代 `src.utils.axis_streamfunction` 和 `src.utils.field_sampling` |
| `src.EP.axis_sources` | `radial_seed_axis` 与 `composite_hua_refined_axis` 的持久化和读取 | 替代 `src.post.representative_eddy_panels` |
| `src.EP.transport_moments` | object-day 旋转采样、QG-like `q_prime`、tau 加权和 aggregate-product moments 所需函数 | 替代 `src.post.transport` 的内部函数 |

## 对象模型

新增 `EPCase`，用于表达：

```text
shape + axis_source + orientation + buoyancy_source
```

后续全生命周期验证、core-shell 分区、材料边界验证都应优先传递 `EPCase` 或等价对象，而不是在大函数之间散传裸字符串。

## 兼容性

本次不改 CLI 名称：

- `python -m src.EP.cli build-smoke`
- `python -m src.EP.cli run-lifecycle-validation`
- `python -m src.EP.cli run-core-shell-ep-validation`
- `python -m src.EP.cli run-core-shell-v2-validation`
- `python -m src.EP.cli run-material-volume-validation`

已有命令仍保持原入口，只是内部工具来源切换到 `src.EP`。

## 当前保留

- `src.utils.ep_flux` 暂时保留为历史参考，不删除。
- `src.post.transport` 仍是正式后处理中的 aggregate-product stirring 入口，但 EP 不再 import 它。
- `src.post.representative_eddy_panels` 仍负责代表涡 panel family，但 EP axis source 构建不再依赖它。

## 后续建议

1. 继续把 `material_volume.py`、`material_geodesic.py`、`core_shell.py` 中的大函数拆成 `EPCase`、`RepresentativeField`、`RegionMaskSet`、`FluxBudget` 等对象方法。
2. 为 `src.EP.numerics` 加 synthetic 单元测试，尤其是流函数反演、双线性插值和曲率尺度。
3. 将各验证入口的输出 schema 固定到文档，避免 CSV 字段在实验迭代中漂移。
4. 如果某个工具同时被 `post` 和 `EP` 需要，应先判断它是展示后处理工具还是理论诊断工具，避免再次互相 import。
