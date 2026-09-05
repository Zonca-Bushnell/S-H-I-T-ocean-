# EP 包自洽化与对象边界重构记录

生成时间：2026-09-05

本文记录 `Zhe/src/EP` 的工程边界。目标是让 EP 理论诊断包可以独立维护，不再反向依赖 `src.post` 或历史 `src.utils`。

## 正式边界

`src.EP` 负责：

- classic / tilted / curved-tube EP 诊断；
- material-volume、material-boundary、core-shell、object-level EP 验证；
- 代表涡 axis source 的读取与必要持久化；
- EP 所需的 object-day aggregate-product moments；
- EP 自己的 IO、数值微分、采样、流函数反演和 N2 读取。

`src.EP` 不负责：

- Hua 识别、tracking、shape classification；
- 代表涡结构合成；
- 面向展示的 post panel family。

正式 EP 入口只允许从：

```bash
python -m src.EP.cli ...
```

## 模块清单

| 模块 | 职责 |
| --- | --- |
| `src.EP.io` | 代表涡、axis source、Filter day、CSV/JSON/parquet/NPZ 读写 |
| `src.EP.numerics` | 网格距离、涡度、流函数反演、插值、导数、N2 等数值函数 |
| `src.EP.axis_sources` | `radial_seed_axis` 与 `composite_hua_refined_axis` 的构建和读取 |
| `src.EP.transport_moments` | EP 内部 object-day aggregate-product moments |
| `src.EP.core_shell_runner` | core-shell 验证 runner 与兼容数值主流程 |
| `src.EP.partition` | inner core / PV-active shell / exchange layer 分区接口 |
| `src.EP.region_flux` | 分区 heat/PV/momentum aggregate-product 统计接口 |
| `src.EP.region_ep` | 分区 EP flux 与 tilt-correction 统计接口 |
| `src.EP.boundary_strategy` | 统一材料边界策略接口与模式注册表 |

`src.EP.core_shell` 现在只保留兼容 facade。旧 notebook 或命令仍可 import 它，但新代码应直接使用 `core_shell_runner` 或上表中的分区模块。

## BoundaryStrategy

材料边界策略统一为：

| 策略 | 物理含义 |
| --- | --- |
| `ThresholdBoundary` | 瞬时阈值连通核心 |
| `LevelSetBoundary` | level-set / morphology 优化后的低 leakage 边界 |
| `LagrangianBoundary` | track-wise 平流连续边界 |
| `LAVDBoundary` | 旋转相干 LAVD 闭合边界 |
| `GeodesicBoundary` | Cauchy-Green / geodesic 材料边界 |
| `PVRetentionBoundary` | 优先保留 PV anomaly core 的 hybrid 边界 |

后续新增边界算法时，应先注册到 `boundary_strategy.py`，再接入具体 runner，避免不同模块各自维护模式字符串。

## 输出目录整理

早期 smoke 和临时验证输出已经归档到：

```text
EP-FLUX/archive/legacy_smoke_and_temp_20260905/
```

仍作为当前理论和结果依据的目录保留在顶层，例如：

- `core_shell_partition_v2/`
- `core_shell_theory_report/`
- `engineering/`

## 后续工作

1. 继续把 `core_shell_runner.py` 中的大函数物理迁移到 `partition.py`、`region_flux.py`、`region_ep.py`，当前调用面已经预留。
2. 将 `material_volume.py`、`material_coherence.py`、`material_geodesic.py` 的具体 mask 构造逐步改成 `BoundaryStrategy` adapter。
3. 为 `src.EP.numerics` 和 `src.EP.axis_sources` 增加 synthetic 单元测试。
4. 固定 EP 输出 schema，避免 CSV 字段在验证迭代中漂移。
