# LIF 生态游戏 — 选型交配 + 存活奖励 alpha 设计

> 在 v2 回合制（`eco_engine.py` / `eco_server.py` / `eco_game.html`）基础上的繁殖机制扩展。两项改动相互独立：
> ① 选型交配：存活回合数相近的个体更倾向互相繁殖；② 存活奖励 alpha：繁殖倍数从 N 变为 N × alpha。

## 背景

v2 生态（纯权重进化、一生无学习）运行诚实结论：正确率贴随机线（best ≈ 0.33）。本设计为增强选择压力的两个繁殖机制：

- **需求 1（选型交配）**：目前存活者 `shuffle` 后相邻随机配对，高龄与低龄混配。改为"存活回合数高者更倾向与同样高者繁殖"，集中存活基因。
- **需求 2（存活奖励 alpha）**：目前繁殖倍数为 N（`每对产仔数 = 存活回合数 × N`）。加入 alpha 因子，最终倍数 = N × alpha，奖励挺过残酷回合的存活者。

## 决策摘要（澄清后确认）

| 决策点 | 结论 |
|---|---|
| "存活率"定义 | **种群级每回合存活率** = 本回合存活者数 / 回合开始种群数 |
| alpha 算法 | `alpha = 1 / survival_rate`，**全局**（全回合所有配对共用），**不设上限**，∈ [1, ∞) |
| alpha 进公式 | **保留存活回合数因子**：`brood = survival_rounds × N × alpha × density` |
| 选型交配机制 | **年龄相似度加权配对**（软）：第一只随机序取，第二只按年龄相似度加权采样 |
| 选型强度调节 | 新增滑条 **选型强度 s**（0-100%，默认 50%），s=0 纯随机（= 现状），s=100% 纯按年龄相似度 |
| 可观察性 | stats 加 `survival_rate`/`alpha` 读数；**只加读数、不加新曲线** |

## 机制

### ① 存活奖励 alpha（种群级，全局）

在 `step_round` 繁殖阶段计算：

```python
start_pop = len(self.pop)                     # 喂食前种群数（繁殖阶段 self.pop 未变）
survival_rate = len(survivors) / start_pop if start_pop > 0 else 0.0
alpha = 1.0 / survival_rate if survival_rate > 0 else 1.0
```

- `survival_rate = 1`（全员存活）→ alpha=1，与原行为一致；随机权重下每回合约 10% 存活 → alpha≈10。
- 不设上限：极端崩溃（如 1% 存活）时 alpha=100，但**总产仔仍受承载力硬上限** `target = min(room, pairs × brood)` 约束，不会超容量。

每对产仔数公式：

```python
brood = int(round(self.survival_rounds * self.n_repro * alpha * density))
```

### ② 选型交配（软，可调强度）

替换当前 `shuffle → 相邻配对`（`eco_engine.py:312-313`）：

```python
self.rng.shuffle(survivors)
pairs = []
while len(survivors) >= 2:
    a = survivors[0]                                      # 随机序取第一只（全员等概率当 a）
    rest = survivors[1:]
    d = np.abs(np.array([g.age for g in rest]) - a.age)   # 与其余存活者年龄差
    w = (1 - s) + s * np.exp(-d / ASSORT_TAU)             # s=选型强度(0-1)，ASSORT_TAU=2.0
    b_idx = int(self.rng.choice(len(rest), p=w / w.sum()))
    b = rest.pop(b_idx)                                   # 移除 b；rest 由 survivors[1:] 切片而来，a 已在切片时排除
    pairs.append((a, b))
    survivors = rest                                      # 移除 a 与 b 后继续
```

- **s=0** → 权重恒等 → 均匀随机配对，统计上等价于原 `shuffle+相邻`（同样是均匀随机完美匹配）。
- **s=100%** → 纯年龄相似度：`|Δage|=0` 权重 1.0、`|Δage|=1` → 0.61、`|Δage|=2` → 0.37，同龄配对约是同差 2 岁的 2.7 倍。
- 奇数存活者：最后一只不配对（与原行为一致）。
- **交叉权重不变**：仍 `weight_a=a.age, weight_b=b.age`。配对后双亲年龄相近 → 后代近似 50/50，选型本身承担"集中高龄基因"的角色。
- 性能：最坏 O(n²)（capacity=5000 时约 1200 万次 exp，numpy 向量化，估数百 ms/回合）；默认 500 容量无感知。若大容量明显变慢，后备方案为"按 age 排序后在 ±窗口内采样"（O(n)），本设计不预做。

## 参数与 API

- **新增可调参数 `assort_strength`**（0.0-1.0，默认 0.5）：`set_config` 校验 0≤s≤1，`_config` 返回，经 `/api/config` 下发。
- 新增常量 `ASSORT_TAU = 2.0`（年龄相似度核宽，回合单位，固定不可调）。
- **stats 扩展**：`round_end` stats 增加 `survival_rate`（0-1）、`alpha`（≥1）。同时存入 `self.last_survival_rate` / `self.last_alpha`，并在 `_last_stats`（`/api/state` 用）中一并返回，保证刷新后读数仍显示。

## 前端改动（eco_game.html）

- **环境参数面板**加一行滑条"选型强度 s"：0-100，step 5，默认 50 → `postConfig({assort_strength: v/100})`；`syncConfigSliders` 反向同步（`assort_strength` 乘 100 显示）。
- **头部统计行**加存活率与 alpha 读数：`种群 X · 正确率 Y% · 存活率 Z% · α W`（Z 百分比 1 位小数，W 保留 1 位小数）。
- **不加新曲线**（4 条曲线保持现状：正确率/种群数/自然死亡率）。

## 测试（eco_tests.py）

现有 12 个测试应全部保持通过（配对方式变化不改事件 schema、承载力上限、名字唯一性、同 seed 可复现）。

新增：
1. `test_survival_alpha` — 抽成纯函数 `survival_alpha(n_survivors, n_start)`：全员存活→1.0；半存活→2.0；0 存活→1.0（防御）。
2. `test_assortative_pairing_tends_similar_age` — 构造年龄两极分化的存活者（半 age=1 / 半 age=20），s=1.0 多轮配对后平均 |Δage| 显著小于 s=0 的平均 |Δage|；验证"倾向"生效。
3. `test_stats_has_survival_rate_alpha` — `step_round` 后 stats 含 `survival_rate`/`alpha`，且存活者>0 时 `alpha == 1/survival_rate`。
4. `test_config_assort_strength` — `set_config(assort_strength=0.8)` 生效并回读；非法值（-0.1 / 1.5）被忽略。

## 诚实预期

- 随机权重下每回合约 10% 存活 → alpha≈10，繁殖放大近 10 倍；但种群受承载力封顶，效果表现为"崩溃后更快回填到满"，而非种群无限增长。
- 选型交配的实际强度受限：随机权重下大部分存活者年龄集中在 1-3 回合（每回合 ~90% 死亡），年龄分布窄，s 与 tau 的区分度有限；效果需在高龄段（长寿命配置 + 低死亡率）才明显。忠实实现、如实呈现。
