# 歧义方法 Registry：识别、代表涡、输送口径

本文登记当前工程中容易混用的科学与工程口径。默认值写在每节第一行。

## 1. 速度异常定义

默认：`30-180d bandpass`

| 口径 | 状态 | 说明 |
|---|---|---|
| raw velocity | 禁止作为默认中心识别 | 只能作为对照图使用 |
| raw - climatology | legacy/diagnostic | 早期中心实验使用过，不再是主口径 |
| 30-180d bandpass | production default | 当前 Hua、速度场、热/PV stirring 使用 |

## 2. Hua/Nencioli 识别版本

默认：`Hua b3_start2 + boundary-monotonic`

| 版本 | 状态 | 说明 |
|---|---|---|
| Nencioli VG 原始 2D | reference/vendor | 用于对比表层速度中心，不含 SSH seed、3D 扩展、tracking |
| Hua hybrid 原始复刻 | diagnostic | 用于方法复刻和论文式图，不是当前默认 |
| Hua b3_start2 | legacy/default ancestor | b3 参数曾作为最佳版本 |
| Hua b3_start2 + boundary-monotonic | production default | 当前最新 Kuroshiou 默认识别口径 |

## 3. 垂向扩展

默认：`strict-contiguous`

| 口径 | 状态 | 说明 |
|---|---|---|
| non-contiguous passed layers | legacy/diagnostic | 允许深层孤立通过层，容易产生跳点 |
| strict-contiguous | production default | 从表层开始，遇到第一层失败即终止 |

## 4. Tracking

默认：`feature/group overlap tracking`

| 口径 | 状态 | 说明 |
|---|---|---|
| nearest-neighbor prototype | legacy | 只能用于早期快速测试 |
| Hua/Rutgers feature/group tracking | production default | 应作为正式轨迹连接依据 |
| C++ 原库 cross-validation | reference | 用于小窗口验证接口差异，不作为生产主链 |

## 5. Shape 筛选

默认：`coherent-only`

| shape | 状态 | 说明 |
|---|---|---|
| coherent | production default | 当前主代表涡与输送分析默认对象 |
| upright_like | planned/comparison | 可单独合成，但不能混入 coherent |
| mixed / complex / transitional | diagnostic | 用于结构差异、错位轴线和失败机制分析 |
| all-shape | legacy/statistical | 可做总体统计，但不作为当前机制主分析 |

## 6. 代表涡合成

默认：`ME_LIUTEX azimuth-preserved TURN`

| 版本 | 状态 | 说明 |
|---|---|---|
| radial/ring-mean representative | legacy/diagnostic | 会抹掉角结构和月牙状速度带 |
| radial seed | active dependency | 不是最终结构图，而是对象、轴线、tau、alpha 的基础输入 |
| ME_LIUTEX TURN | production default | global alpha 对齐后保留 `r × theta` 角向结构 |
| ME_LIUTEX UNTURN | structural control | 不旋转直接合成，用于观察相位相消 |

## 7. 输送诊断

默认：`aggregate-product stirring`

| 口径 | 状态 | 说明 |
|---|---|---|
| mean-product | 对照 | `mean(v) * mean(tracer)`，不能作为主输送结论 |
| product-mean | production default | `mean(v * tracer)`，对应真实协方差合成 |
| covariance | production default diagnostic | `product_mean - mean_product` |
| trapping | out of current default | 需要对象迁移速度与 trapped boundary，不应混入 stirring |

## 8. 方向口径

默认：`global_ls_alpha 后的 y_rot`

| 方向 | 状态 | 说明 |
|---|---|---|
| geographic north | basin transport only | 研究真实海盆净经向输送时使用 |
| y_rot | representative default | 代表涡内部横向 stirring 方向 |
| unturned east/north | structural control | 只用于 UNTURN 对照 |

## 9. 中心定义

默认主轴：`velocity center`；辅助诊断：`rotation core`

| 中心 | 状态 | 说明 |
|---|---|---|
| velocity center | production axis | Hua/VG 速度弱中心 + 几何约束 |
| rotation core | diagnostic model | `|zeta|` 或旋转核心，用于双核心模型 |
| psi center | theory comparison | 速度反演流函数中心，适合模态机制对照 |
| temperature center | external literature | Li/Yang/Xu 等论文可能使用，不是我们的主中心定义 |

## 10. 默认解释规则

- 没有显式写出口径的图或表，不应进入论文主结论。
- 结构图不能替代输送协方差。
- coherent、upright_like、complex 不得混合合成后解释为单一机制。
- TURN 可连接 aggregate-product stirring；UNTURN 只作为结构相消对照。

