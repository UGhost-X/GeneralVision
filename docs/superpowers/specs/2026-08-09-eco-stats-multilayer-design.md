# LIF 生态游戏 — 统计面板 + 多层结构突变 + 选型交配/存活alpha 设计

> 在 v2 回合制（`eco_engine.py` / `eco_server.py` / `eco_game.html`）基础上的一次合并迭代。
> 两组功能：**① 统计与解剖展示**（种群统计历史、存活回合数分布、个体架构解构）与 **② 多层结构突变**（架构继承 + 五类保函数突变），叠加 **③ 选型交配 + 存活奖励 alpha**（承接 `2026-08-09-eco-assortative-alpha-design.md`，本设计为其实施基线并整合进同一引擎）。

## 背景

v2 生态（纯权重进化、一生无学习）现状：架构固定 784→100→10，正确率贴随机线（best≈0.33）。本次三组改动：

1. **统计面板没数据**：前端曲线需 2 回合才出现、刷新即清零（服务端无历史）；且无存活结构信息。
2. **想看到多层进化**：初始生物体单层，变异需能引入多层神经元，且要求"突变型丰富、稳定性高"。
3. **展示个体神经结构**：点击个体看其架构解构。
4. **增强选择压力**（用户设计）：选型交配 + 存活奖励 alpha。

## Part A — 统计面板：历史 + 存活回合数分布

### A1. 服务端历史持久化（修"无数据"根因）

`Ecosystem` 增加 `self.history: list[dict]`，每回合 `round_end` 后追加一条：
`{round, alive, avg_acc, natural_rate, survival_rate, alpha}`。

- `/api/state` 返回 `history`（全量序列，前端只取最近 200 条）。
- `_last_stats()` 补齐 `avg_acc`（当前缺，导致刷新后头部正确率显示 "—"）。
- 前端 `loadState` 用 `history` 预填充 `correctSeries / popSeries / naturalSeries` → **打开页面即有曲线**。
- 移除 `drawCurves` 的 `if (n < 2) return` 门槛：n=1 画单点，n=0 显示占位文字"等待推演…"。

### A2. 存活回合数分布（每回合直方图）

服务端在 `step_round` 繁殖后、回合结束统计时，对当前存活种群按年龄分箱：

```python
age_hist = [0] * survival_rounds          # 下标 i = 存活 i+1 回合
for g in self.pop:
    if 1 <= g.age <= self.survival_rounds:
        age_hist[g.age - 1] += 1
```

- **不统计**年龄 0 的当回合新生个体（它们还没喂食、未"存活"过）；新生数单独以 `newborns` 字段返回。
- `round_end` stats 增加 `age_hist` 与 `newborns`；`/api/state` 的 stats 同样携带当前快照（刷新可见）。
- 前端在**种群统计面板**新增一张**竖条直方图**（`x = 存活回合数 1..上限`，`y = 个体数`），每回合更新；面板顶部标注"新生 N 只"。
- 直方图与 A1 的历史曲线并存（曲线维持 3 条：正确率/种群数/自然死亡率；`survival_rate`/`alpha` 只进头部读数，不加曲线——遵循用户设计文档约定）。

## Part B — 个体架构解构（点击个体）

### B1. 服务端暴露架构

- `Genome` 增加 `arch()` 属性：返回隐藏层神经元数列表 `[n1, n2, …, nk]`（多层引入后每层规模可变）。
- `/api/state` 的 population 条目增加 `arch: [n1, …, nk]`；`birth` 事件也带 `arch`（新生个体前端需立即画出）。
- 渲染成本低：每次前端重建种群/新生时各带一个短数组。

### B2. 前端解剖面板

- 保留现有：父母/出生回合/已活回合/本回合对错/产出层 10 通道脉冲分布。
- 新增**架构图**：`输入 784 → 层1 [n1] → 层2 [n2] → … → 产出 10`，每层一个色块并标注神经元数；层数 1 与多层视觉区分（单层灰、多层紫）。
- 未多层化前的个体（初始单层）也能正常渲染（arch=[100]）。

## Part C — 多层结构突变（核心）

### C1. 基因组表示：固定 → 变深度

```python
@dataclass
class Genome:
    name: str
    layers: list[np.ndarray]   # 隐藏层权重 [(784,n1), (n1,n2), …, (n_{k-1},n_k)]
    readout: np.ndarray        # (n_k, 10)
    born_gen: int = 0
    age: int = 0
    parents: tuple | None = None
```

- 初始/全灭重播仍由 `random_genome` 生成**单层 784→100→10**（多层只由变异产生）。
- 新增边界常量：`MIN_LAYERS=1`、`MAX_LAYERS=4`、`MIN_NEURONS=20`、`MAX_NEURONS=200`、`MAX_HIDDEN=400`（全部隐藏神经元硬上限，保回合耗时）。
- 隐藏层共享 `THETA_HIDDEN=12`、产出层 `THETA_READOUT=1.5`（暂不做逐层 theta，YAGNI）。

### C2. 前向泛化（numba JIT）

将单隐藏层 `_forward_core`（已 numba JIT，见 `eco_engine.py:113`）泛化为 `_forward_core_multi(S, layers, Wr, dims, leak, theta_h, theta_r, ref_period, n_t)`：

- `layers` 用 `numba.typed.List` 传变长形状的 2D 权重数组（numba 类型为 `List(Array(float64, 2, 'C'))`，元素形状可不同，nopython 模式支持）。
- 每时间步：`prev = S[:,:,t]` → 逐层 `LIF+WTA`（漏电→不应期清零→WTA 发放→不应期递减），层 l 的 one-hot 发放作为层 l+1 输入；最后一层发放进产出层。
- 沿用共享-S 优化（`forward_from_S`）与逐个体 `B=1` 调用模式不变。
- **性能预期**：深层输入极稀疏（前一层 WTA 每样本每步至多 1 个脉冲），第 0 层（784 输入）是主要成本；`MAX_HIDDEN` 保证上限。单层个体路径与现状持平。

### C3. 交叉：架构继承（双亲架构不同）

```
1. 架构父 = 存活回合数更长者（tie 掷硬币）→ 子代架构 = 架构父的层列表
2. 权重混合：逐层，仅当双亲该层"位置相同且形状兼容"做存活加权逐权重 50/50 + 均匀扰动 + 0.1% 重初始化 + 活跃列归一化；
   不兼容层直接取架构父（形状确定，绝无维度错配）
3. 之后施加结构突变（见 C4）
```

- 保证子代架构始终落在已验证亲代的可行邻域（稳定性）；新奇性全交给突变。
- 存活加权语义沿用：`weight_a=a.age, weight_b=b.age`。

### C4. 五类结构突变（出生时施加，概率分层）

| 突变 | 做法 | 破坏性 | 概率(建议) |
|---|---|---|---|
| ① 静默神经元诞生 | 某层扩 +d 列（权重 U(-0.02,0.02)，**不列归一化→近零不放电**），下一层同步扩 +d 行 | 极小 | 0.40 |
| ② 层复制/恒等插入 | 取隐藏层 W(n_in,n_out)，紧跟插入 (n_out→n_out)=I·scale+噪声 → W₂∘W₁≈W | 小 | 0.15 |
| ③ 层合并 | 相邻 W₁(a→b)、W₂(b→c) 合成 W=W₂@W₁(a→c)，删中间 LIF | 中 | 0.10 |
| ④ 神经元剪枝 | 删某层不发火/幅值最小的列 + 下一层对应行 | 中 | 0.10 |
| ⑤ 随机整层插入 | 插入全新随机层（列归一化到 W_NORM） | 大 | 0.03 |
| 权重级 | 沿用：存活加权 50/50 + U(-σ,σ) + 0.1% 重初始化 | — | 每子代恒有 |

- **稳定性三保险**：静默神经元不归一化（保持静默）；列归一化只作用于活跃列——**新常量 `NORM_ACTIVE_EPS=1.0`**：列 L2 范数 ≥1.0 的列归一化到 W_NORM，范数 <1.0 的列（静默诞生/近零突变）跳过、保持近零；层数/每层/总量全部钳制在 C1 边界内。
- 每次繁殖的子代：结构突变按概率独立判定，可 0 次（保持亲代架构）也可多次（叠加，受上限约束）。
- 常量：`P_GROW=0.40 / P_SPLIT=0.15 / P_MERGE=0.10 / P_PRUNE=0.10 / P_ADDRANDOM=0.03`、`NORM_ACTIVE_EPS=1.0`。

### C5. 诚实预期

- 深层个体初生时第二层起"喂不饱"（第一层 WTA 吐脉冲稀疏，第二层难达 θ=12）→ 多层个体大概率早期死亡，属合理淘汰压力。
- 纯权重进化下多层不太可能超越单层正确率（README 已有 2 层退化结论）；本次目标是**让多层结构能出现、能观测、能在生态中受选择**，存活分布直方图如实呈现。
- 若实测深层全哑，后续再引入 `input_gain` 补偿（STDP 系统已有此教训），本设计不预做。

## Part D — 选型交配 + 存活奖励 alpha（承接用户设计文档）

按其 `2026-08-09-eco-assortative-alpha-design.md` 实施，要点：

- `alpha = 1 / survival_rate`（种群级、全局、不设上限、∈[1,∞)），`brood = survival_rounds × N × alpha × density`；总产仔仍受承载力硬上限。
- 选型交配：第一只随机序取，第二只按 `w = (1-s) + s·exp(-Δage/ASSORT_TAU)` 加权采样（`ASSORT_TAU=2.0`）；`s=0` 退化为纯随机。
- 新增可调参数 `assort_strength`（0-1，默认 0.5），前端加滑条；stats 增加 `survival_rate`/`alpha`（round_end + `/api/state`）。
- 新增测试：`survival_alpha` 纯函数、`assortative_pairing_tends_similar_age`、`stats_has_survival_rate_alpha`、`config_assort_strength`。
- 与 C3 集成：选型交配确定配对后，仍走"架构继承交叉 + 结构突变"生子代。

## API 变更汇总

| 端/事件 | 变更 |
|---|---|
| `/api/state` | `history`（stats 序列）、population 条目加 `arch`、stats 加 `avg_acc`/`age_hist`/`newborns`/`survival_rate`/`alpha`、config 加 `assort_strength` |
| `round_end` stats | 加 `age_hist`、`newborns`、`survival_rate`、`alpha` |
| `birth` 事件 | 加 `arch` |
| `/api/config` | 支持 `assort_strength`（0-1） |

## 前端变更汇总（eco_game.html）

1. **种群统计面板**：曲线由服务器历史预填充、去掉 n<2 门槛；新增存活回合数竖条直方图 + 新生数标注。
2. **头部**：加存活率与 alpha 读数。
3. **环境参数面板**：加"选型强度 s"滑条（0-100，默认 50）。
4. **个体解剖面板**：新增架构图（输入→层块→产出）。

## 测试（eco_tests.py）

- **保持通过**：现有 12 个测试（含 numba 对照）；`test_columns_normalized`/`test_crossover_mixes_both_parents`/`test_mutation_is_rare`/`test_weighted_crossover` 需适配多层表示（单层个体仍应满足原断言）。
- **改写**：numba 对照测试泛化为多层 golden reference（`_forward_core_multi` vs 纯 numpy 参考）。
- **新增**：
  1. `test_multilayer_forward_matches_reference` — 多层 forward（含 2-3 层个体）与 numpy 参考一致。
  2. `test_structural_mutations_bounded` — 任意突变后层数/每层/总量都在边界内，且单层不塌成 0 层。
  3. `test_architecture_inheritance` — 双亲架构不同时，子代架构 = 存活更长亲代的架构（同寿 tie 随机其一）。
  4. `test_silent_birth_preserves_output` — 静默神经元诞生前后对同 S 的输出分布近似（行为保持）。
  5. `test_history_and_age_hist` — step 后 `/api/state` 含 history、stats 含 age_hist/newborns。
  6. 用户文档 4 个测试（survival_alpha / assortative / stats_survival_alpha / config_assort_strength）。

## 实施顺序（供 writing-plans 参考）

1. **引擎多层化**（C1-C2）：Genome.layers、random_genome、`_forward_core_multi`、forward/forward_from_S 泛化。
2. **交叉与结构突变**（C3-C4）：架构继承 + 五类突变 + 边界钳制。
3. **繁殖阶段整合**（D + C）：选型配对 + alpha + 调用新交叉。
4. **统计与 API**（A1-A2/B1）：history、age_hist、newborns、arch、survival_rate/alpha、config。
5. **前端**（A/B/D）：曲线预填充、直方图、头部读数、滑条、解剖架构图。
6. **测试**：适配 + 新增。
7. **验证**：全测试、冒烟、诚实基线、README 更新。

## 风险与边界

- **numba typed.List 传变长层**：实现前先做一个最小可行性验证（2 层小网络 JIT 跑通）；若 numba 0.66 不支持变长 List 的 nopython 索引，后备方案为"固定最大形状 + 掩码"（内存/性能略差，逻辑等价）。
- **性能**：多层种群回合耗时上升，`MAX_HIDDEN=400` 硬上限兜底；冒烟时量测并如实报告。
- **确定性**：所有新随机（结构突变、选型配对、alpha 无随机）由 `self.rng`/派生 seed 驱动，保持同 seed 可复现。
