# 文献文件作用说明

本目录用于保存 Original worktree 中与前辈口径、三维涡旋结构和热成风速度来源有关的
PDF。后续遇到难以决断的科学口径问题时，先补查文献，把可用 PDF 放入本目录，并在
`Dipole_vertical _transport/docs/literature/` 或相应诊断文档中记录依据。

## 有效 PDF

| 文件 | 验证 | 在当前诊断中的作用 |
|---|---|---|
| `Zhang_etal_2024_OLAR_Three_Dimensional_Structure_Oceanic_Mesoscale_Eddies.pdf` | `pypdf` 可解析，17 页，题名为 `Three-Dimensional Structure of Oceanic Mesoscale Eddies` | 支持把三维 density、pressure anomaly、geostrophic currents 作为同一套涡旋三维结构重建量。由此，热成风速度剪切应优先从 Argo/涡旋合成 absolute density 的水平梯度积分，而不是直接用 ISAS/BOA 背景密度替代。 |

## 已尝试但无效

| 文献 | 状态 | 后续处理 |
|---|---|---|
| Chaigneau et al. (2011), `Vertical structure of mesoscale eddies in the eastern South Pacific Ocean: A composite analysis from altimetry and Argo profiling floats`, DOI `10.1029/2011JC007134` | Wiley PDF 直链下载得到 HTML/拦截页，`pypdf` 无法解析，已删除无效文件 | 仍可作为 DOI 和方法逻辑依据：Argo-eddy composite 先围绕涡旋中心构造三维温盐/密度结构，再解释涡旋垂向结构；若后续手动获得有效 PDF，再放入本目录并更新此表。 |

## 对当前代码口径的约束

1. 热成风速度 `u(z), v(z)` 是涡旋三维密度/压力结构对应的地转速度剪切，应优先用
   Argo composite absolute density 的水平密度梯度积分，并以 1000 m Argo parking
   drift 为锚点。
2. ISAS/BOA 背景场主要用于定义环境态、异常场或背景等密面坡度诊断，不应直接替代
   涡旋合成密度场来给出涡旋速度剪切。
3. 若后续要改变 `term1`、`term2`、热成风积分源、背景扣除方式或等密面反插方式，
   必须先在文档中写清楚对应文献依据、物理意义和与前辈程序的差异。
