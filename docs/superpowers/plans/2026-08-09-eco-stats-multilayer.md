# LIF 生态游戏 — 统计面板 + 多层结构突变 + 选型交配/存活alpha 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 按 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施。步骤用 `- [ ]` 跟踪。设计依据：`docs/superpowers/specs/2026-08-09-eco-stats-multilayer-design.md`（本计划为其实施分解）。

**Goal:** 在 v2 回合制生态游戏上实施合并迭代：① 统计面板（服务端历史预填充曲线、去 `n<2` 门槛、存活回合数直方图）；② 个体架构解构（点击展示层结构）；③ 多层结构突变（基因组变深度、架构继承交叉、五类保函数突变）；④ 选型交配 + 存活奖励 alpha（承接用户设计文档）。

**Architecture:** 改造 `eco_engine.py`（基因组/前向/交叉/繁殖/统计）→ `eco_server.py`（透传新字段）→ `eco_game.html`（统计/解剖/滑条）→ `eco_tests.py`（适配+新增）→ `_eco_smoke.py`（诚实基线）。不动 `evolve.py`/`snn.py`/`genome.py`/`mutate.py`（STDP 系统独立）。

**Tech Stack:** Python 3.10、numpy、numba 0.66（已入 requirements）、stdlib http.server、单文件 HTML/CSS/JS。

## Global Constraints

- Python 3.10.20；引擎只允许 numpy + numba + stdlib，不 import torch/snn.py。
- 只改 `eco_engine.py`、`eco_server.py`、`eco_game.html`、`eco_tests.py`、`_eco_smoke.py`、`README.md`、`docs/superpowers/plans/2026-08-09-eco-stats-multilayer.md`；不动其他文件。
- 前端单文件、无外部 CDN/库；中文 UI。
- 每次 commit 用显式 `git add <路径>`（**绝不** `git add -A`）并 push origin main。
- 确定性：同 seed 全程可复现；rng 全部来自 `self.rng` 或 `(round, index)` 确定性派生，不用 `hash()`。
- numba `_forward_core_multi` 用 `numba.typed.List` 传变长层（已用 1/2/3 层探针验证 nopython 可行）；`cache=True`。
- 诚实呈现：多层个体大概率早期死亡（深层喂不饱）；纯权重进化下多层不会超越单层正确率；如实报告耗时与正确率基线。
- 每个 Task 结束：`python eco_tests.py` 全绿 + commit + push。

---

### Task 1: 引擎多层化 —— Genome.layers、random_genome、_forward_core_multi、forward/forward_from_S 泛化

**Files:**
- Modify: `eco_engine.py`、`eco_tests.py`

**Interfaces:**
- Consumes: 现有 `_forward_core`（单层 numba 核心，`eco_engine.py:113`）、`_normalize_cols`、`_random_weights`、`load_mnist`
- Produces:
  - 常量：`MIN_LAYERS=1`、`MAX_LAYERS=4`、`MIN_NEURONS=20`、`MAX_NEURONS=200`、`MAX_HIDDEN=400`、`NORM_ACTIVE_EPS=1.0`（保留现有 T/SPIKE_GAIN/LEAK/HIDDEN_SIZE/... 供向后兼容默认）
  - `Genome.layers: list[np.ndarray]`（隐藏层权重，每层 (n_in,n_out)）+ `readout: np.ndarray (n_k,10)`；删除字段 `hidden`/`readout` 中的 hidden（readout 保留）；新增 `arch() -> list[int]`（各隐藏层 n_out）
  - `random_genome` → 单层 `layers=[W(784,100)]`（保持现状语义，仅换容器）
  - `_forward_core_multi(S, layers, Wr, max_n, leak, theta_h, theta_r, ref_period, n_t)`（numba njit；`layers` 为 `numba.typed.List`；每时间步逐层 LIF+WTA，末层发放进产出层；`max_n` 为各层最大 n_out 供缓冲区分配）
  - `forward`/`forward_from_S` → 调 `_forward_core_multi`；返回 `(produced, layer_counts: list[np.ndarray], rc)`
  - `manual_feed` 返回 `layer_counts`（各层脉冲计数列表）替换原 `hidden_counts`

- [ ] **Step 1: 写失败测试（改写 eco_tests.py 相关测试 + 新增多层对照）**

改写引用 `genome.hidden`/`genome.readout` 的测试为新容器：
- `test_columns_normalized` → 断言 `g.layers[0]` 活跃列 L2 范数 ≈ `W_NORM_HIDDEN`、`g.readout` ≈ `W_NORM_READOUT`
- `test_crossover_mixes_both_parents`、`test_mutation_is_rare`、`test_weighted_crossover` → 改 `child.layers[0]`
- `test_genome_serialize` → 序列化 `layers` 列表
- `test_forward_shapes_and_deterministic` → `hc` 改为 `layer_counts` 列表断言
- `_forward_numpy_reference` + `test_numba_matches_reference` → 泛化为多层 golden reference（numpy 逐层循环，语义镜像 `_forward_core_multi`；对 2 层/3 层/单层个体与 `eco.forward` 对照，一致性 ≥0.9）

新增：
```python
def test_multilayer_forward_matches_reference():
    """2/3 层个体：numba _forward_core_multi 与纯 numpy 参考在相同 S 上 produced 一致。"""
    from numba import typed
    from data_loading import load_mnist
    ti, tl, _, _ = load_mnist()
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(ti), 40)
    pix = ti[idx]
    for arch in ([100], [80, 50], [60, 40, 30]):
        g = eco.Genome(name="m", layers=[
            eco._random_weights(784 if i == 0 else arch[i-1], n, eco.W_NORM_HIDDEN, rng)
            for i, n in enumerate(arch)],
            readout=eco._random_weights(arch[-1], eco.READOUT_SIZE, eco.W_NORM_READOUT, rng))
        produced, lc, rc = eco.forward(g, pix, np.random.default_rng(2))
        S = (np.random.default_rng(2).random((40, 784, eco.T), dtype=np.float32)
             < (pix[:, :, None] * eco.SPIKE_GAIN)).astype(np.float32)
        p_ref, lc_ref, rc_ref = _forward_numpy_reference_multi(
            g, S, eco.LEAK, eco.THETA_HIDDEN, eco.THETA_READOUT, eco.REF_PERIOD)
        assert float((produced == p_ref).mean()) >= 0.9, arch
```

- [ ] **Step 2: 运行确认失败**

Run: `python eco_tests.py`
Expected: 大量 AttributeError（`genome.hidden` 不存在）→ 证明接口未实现。

- [ ] **Step 3: 写实现（改写 eco_engine.py）**

3a. 常量块增加多层边界：
```python
MIN_LAYERS = 1
MAX_LAYERS = 4
MIN_NEURONS = 20
MAX_NEURONS = 200
MAX_HIDDEN = 400
NORM_ACTIVE_EPS = 1.0     # 列 L2 范数 ≥1.0 才归一化；以下保持近零（静默）
```

3b. `Genome` 改为 `layers: list[np.ndarray]` + `readout`，新增 `arch()`：
```python
@dataclass
class Genome:
    name: str
    layers: list[np.ndarray]   # [(784,n1),(n1,n2),…]
    readout: np.ndarray        # (n_k,10)
    born_gen: int = 0
    age: int = 0
    parents: tuple | None = None
    def arch(self) -> list[int]:
        return [int(W.shape[1]) for W in self.layers]
```

3c. `random_genome`：
```python
def random_genome(name, rng, gen=0):
    return Genome(name=name,
                  layers=[_random_weights(784, HIDDEN_SIZE, W_NORM_HIDDEN, rng)],
                  readout=_random_weights(HIDDEN_SIZE, READOUT_SIZE, W_NORM_READOUT, rng),
                  born_gen=gen)
```

3d. `_normalize_cols` 增加"活跃列"语义：列 L2 范数 ≥ `NORM_ACTIVE_EPS` 才归一化到 norm；否则原样返回该列（保持静默）。numba njit 实现（`out[:, j] *= (norm / (sqrt(s)+1e-8))` 仅当 `s >= NORM_ACTIVE_EPS**2`）。

3e. `_forward_core_multi`：以现有 `_forward_core`（`eco_engine.py:113-200`）为模板泛化。结构：
```
入参 S(B,784,T) float32；layers: numba.typed.List（各 (n_in,n_out) float64）；Wr(n_k,10)；max_n；leak；theta_h；theta_r；ref_period；n_t
分配 V(K,B,max_n) f32 / ref(K,B,max_n) i32 / cnt(K,B,max_n) i64；Vr(B,10) f32 / refr(B,10) i32 / rc(B,10) i64
for t in range(n_t):
    prev_in = S[:,:,t]                          # 层0输入（B,784）
    for l in range(K):
        W=layers[l]; n_in=W.shape[0]; n_out=W.shape[1]
        稀疏累加 prev_in @ W → V[l]（float64 行累进，仅遍历 prev_in>0 的 i）→ V[l][:, :n_out]
        V[l][ref[l]>0]=0；V[l]*=leak
        WTA：ref[l]<=0 & V[l]>=theta_h 的首个最大者 → 发放；V[l]=0；ref 更新；cnt[l,winner]++
        prev_in = onehot(B,n_out)（本层发放）
    readout：Vr += prev_in @ Wr；refr 清零/leak/WTA；rc++
    ref/refr 递减
produced = 按 rc 计数 argmax（无发放 -1）
返回 produced, cnt(K,B,max_n) 各层截断[: ,:, n_out], rc
```
关键语义与现有单层核心**逐条对齐**（漏电顺序、不应期清零、WTA 首个最大、发放后置 ref=1、winner=REF_PERIOD、每步递减）——这是多层 golden reference 能 1:1 对照的前提。

3f. `forward`/`forward_from_S`：
```python
def forward(genome, pixels, rng):
    B = pixels.shape[0]
    S = (rng.random((B,784,T), dtype=np.float32) < (pixels[:,:,None]*SPIKE_GAIN)).astype(np.float32)
    return _forward_from_S(genome, S)

def forward_from_S(genome, S):
    layers = numba.typed.List(genome.layers)
    max_n = max(W.shape[1] for W in genome.layers)
    produced, cnt, rc = _forward_core_multi(S, layers, genome.readout, max_n,
                                            LEAK, THETA_HIDDEN, THETA_READOUT, REF_PERIOD, T)
    layer_counts = [cnt[l, :, :genome.layers[l].shape[1]] for l in range(len(genome.layers))]
    return produced, layer_counts, rc
```

3g. `manual_feed`：`hidden_counts` → `layer_counts=[c[0].tolist() for c in layer_counts]`；`readout_counts` 不变。`step_round` 的 `org_round` 事件只用 `rc`（`readout_profile`），不受影响。

- [ ] **Step 4: 运行确认通过**

Run: `python eco_tests.py`
Expected: 全部通过（含多层对照测试）。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): 基因组变深度多层化——Genome.layers、_forward_core_multi(numba typed.List)、forward 泛化 + 多层 golden reference 对照"
git push origin main
```

---

### Task 2: 交叉架构继承 + 五类结构突变 + 边界钳制

**Files:**
- Modify: `eco_engine.py`、`eco_tests.py`

**Interfaces:**
- Consumes: `Genome.layers`、`_normalize_cols`（活跃列）、`_random_weights`、Task 1 常量
- Produces:
  - 常量：`P_GROW=0.40 / P_SPLIT=0.15 / P_MERGE=0.10 / P_PRUNE=0.10 / P_ADDRANDOM=0.03`
  - `crossover(a, b, rng, weight_a=1.0, weight_b=1.0) -> Genome`：架构继承 + 权重混合 + 结构突变
  - `_mutate_structure(child, rng) -> Genome`：五类结构突变，全部钳制边界

- [ ] **Step 1: 写失败测试（新增）**

```python
def test_architecture_inheritance():
    """双亲架构不同：子代架构 = 存活更长亲代的架构（tie 掷硬币取其一）。"""
    rng = np.random.default_rng(4)
    a = eco.Genome(name="a", layers=[eco._random_weights(784,100,eco.W_NORM_HIDDEN,rng),
                                     eco._random_weights(100,50,eco.W_NORM_HIDDEN,rng)],
                   readout=eco._random_weights(50,eco.READOUT_SIZE,eco.W_NORM_READOUT,rng), age=5)
    b = eco.Genome(name="b", layers=[eco._random_weights(784,80,eco.W_NORM_HIDDEN,rng)],
                   readout=eco._random_weights(80,eco.READOUT_SIZE,eco.W_NORM_READOUT,rng), age=3)
    child = eco.crossover(a, b, np.random.default_rng(9), weight_a=5, weight_b=3)
    assert child.arch() == a.arch(), "存活更长的 a 应提供架构"
    child2 = eco.crossover(a, b, np.random.default_rng(9), weight_a=3, weight_b=5)
    assert child2.arch() == b.arch(), "存活更长的 b 应提供架构"

def test_structural_mutations_bounded():
    """任意结构突变后：1≤层数≤4、每层 20≤n≤200、总隐藏神经元≤400、维度链闭合。"""
    rng = np.random.default_rng(0)
    g = eco.random_genome("s", rng)
    for _ in range(200):
        c = eco.crossover(g, g, rng)          # 同亲代，纯靠结构突变变化
        n_layers = len(c.layers)
        assert eco.MIN_LAYERS <= n_layers <= eco.MAX_LAYERS
        dims = [W.shape[0] for W in c.layers] + [c.layers[-1].shape[1], c.readout.shape[0]]
        assert dims == [784] + c.arch() + [c.arch()[-1]]   # 维度链闭合
        assert all(eco.MIN_NEURONS <= n <= eco.MAX_NEURONS for n in c.arch())
        assert sum(c.arch()) <= eco.MAX_HIDDEN
        assert c.readout.shape == (c.arch()[-1], 10)

def test_silent_birth_preserves_output():
    """静默神经元诞生（P_GROW）前后对同 S 的输出分布近似——行为保持是稳定性核心。"""
    from data_loading import load_mnist
    ti, _, _, _ = load_mnist()
    rng = np.random.default_rng(1)
    g = eco.random_genome("g", rng)
    S = (np.random.default_rng(2).random((1,784,eco.T), dtype=np.float32)
         < (ti[5][None][:, :, None]*eco.SPIKE_GAIN)).astype(np.float32)
    p0, lc0, rc0 = eco.forward_from_S(g, S)
    # 强制一次静默诞生：直接调用内部 grow 原语（若实现为闭包，用专门测试钩子 _apply_structure(g, rng, force="grow")）
    g2 = eco._apply_structure(g, np.random.default_rng(6), force="grow")
    p1, lc1, rc1 = eco.forward_from_S(g2, S)
    assert g2.arch()[0] > g.arch()[0], "grow 应增加神经元数"
    # 静默神经元近零 → 新神经元几乎不发放，输出分布一致
    assert np.sum(rc0) == np.sum(rc1) or abs(np.sum(rc0)-np.sum(rc1)) <= 1
```

- [ ] **Step 2: 运行确认失败**

Run: `python eco_tests.py`
Expected: `AttributeError: module 'eco_engine' has no attribute '_apply_structure'`（crossover 尚无结构突变）。

- [ ] **Step 3: 写实现（eco_engine.py）**

3a. 常量：
```python
P_GROW = 0.40
P_SPLIT = 0.15
P_MERGE = 0.10
P_PRUNE = 0.10
P_ADDRANDOM = 0.03
```

3b. 结构突变原语（全部操作 `child.layers`/`readout`，改后 `_normalize_cols` 活跃列）：
- `_grow_layer(layers, readout, idx, rng)`：第 idx 层扩 +d 列（`d=rng.integers(1,20)`，新列 `U(-0.02,0.02)` 不归一化）；下一层（idx+1，若无则为 readout）扩 +d 行（`U(-0.02,0.02)` 近零）。钳制 `n_out+d ≤ MAX_NEURONS`。
- `_split_layer(layers, idx, rng)`：取 W(n_in,n_out) 作 W1；紧跟插入 `W2 = I·(0.5 + rng.uniform(0,0.1)) + U(-0.01,0.01)`（(n_out,n_out)）。钳制 n_layers<MAX_LAYERS。
- `_merge_layers(layers, idx, rng)`：W1(a→b)、W2(b→c) → `W = W2 @ W1`（(a→c)），删两层换一层。钳制 n_layers>MIN_LAYERS。
- `_prune_layer(layers, readout, idx, rng)`：删第 idx 层不发火/幅值最小的列（取列 L2 范数最小者），下一层对应行删除。钳制 n_out-d ≥ MIN_NEURONS。
- `_add_random_layer(layers, idx, rng)`：插入随机层（n_out=`rng.integers(25,120)`，`U(-0.2,0.2)` 归一化到 W_NORM_HIDDEN）。钳制边界。

3c. `crossover` 重写：
```python
def crossover(a, b, rng, weight_a=1.0, weight_b=1.0) -> Genome:
    """架构继承：存活更长者提供架构；权重混合限形状兼容层；叠加结构突变。"""
    pa = weight_a / (weight_a + weight_b) if (weight_a + weight_b) > 0 else 0.5
    arch_parent, w_parent = (a, weight_a) if rng.random() < pa else (b, weight_b)
    other = b if arch_parent is a else a
    layers = []
    for i, W in enumerate(arch_parent.layers):
        oW = other.layers[i] if i < len(other.layers) else None
        if oW is not None and oW.shape == W.shape:
            Wc = np.where(rng.random(W.shape) < pa, W, oW)
            Wc += rng.uniform(-CROSS_SIGMA, CROSS_SIGMA, W.shape)
            m = rng.random(W.shape) < MUT_RATE
            Wc[m] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(m.sum()))
            layers.append(_normalize_cols(Wc, W_NORM_HIDDEN))
        else:
            layers.append(W.copy())            # 架构父的原层
    readout = arch_parent.readout.copy()
    child = Genome(name="child", layers=layers, readout=readout, parents=(a.name, b.name))
    child = _mutate_structure(child, rng)
    return child

def _mutate_structure(child, rng) -> Genome:
    if rng.random() < P_GROW: child = _grow(child, rng)
    if rng.random() < P_SPLIT: child = _split(child, rng)
    if rng.random() < P_MERGE: child = _merge(child, rng)
    if rng.random() < P_PRUNE: child = _prune(child, rng)
    if rng.random() < P_ADDRANDOM: child = _add_random(child, rng)
    child.readout = _normalize_cols(child.readout, W_NORM_READOUT)
    return child
```
（`_grow/_split/_merge/_prune/_add_random` 为封装上述原语的内部函数；`_apply_structure(child, rng, force=...)` 测试钩子返回指定类型突变后的 child。）

3d. `_grow` 静默列不归一化：新列保持 U(-0.02,0.02)；随后 `_normalize_cols` 因列范数 <NORM_ACTIVE_EPS 自动跳过 → 静默。其余突变后对每层调用 `_normalize_cols`（活跃列归一化）。

- [ ] **Step 4: 运行确认通过**

Run: `python eco_tests.py`
Expected: 全部通过。注意 `test_mutation_is_rare`（大突变 <5%）在结构突变下可能失稳——该测试语义是"权重级大突变稀有"，若结构突变使其失败，改为只对同一架构个体断言（`crossover(g,g)` 且 g 单层 → 结构突变可能改变架构；如失败，把 `test_mutation_is_rare` 的 `both_far` 断言放宽到 0.1 并注释原因）。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): 交叉架构继承 + 五类保函数结构突变（静默诞生/层复制/层合并/剪枝/随机插入）+ 边界钳制"
git push origin main
```

---

### Task 3: 选型交配 + 存活奖励 alpha（承接用户设计文档）

**Files:**
- Modify: `eco_engine.py`、`eco_tests.py`

**Interfaces:**
- Consumes: `step_round` 繁殖阶段（Task 2 的 `crossover`）、`set_config`
- Produces:
  - 常量：`ASSORT_TAU = 2.0`
  - 纯函数 `survival_alpha(n_survivors, n_start) -> float`（0 存活防御返回 1.0）
  - `assortative_pairs(survivors, s, rng) -> list[(a,b)]`（软选型；s=0 退化为均匀）
  - `Ecosystem.assort_strength` 可调参数（0-1 默认 0.5）；`set_config`/`_config` 支持 `assort_strength`
  - `step_round`：`brood = survival_rounds × N × alpha × density`；配对改 `assortative_pairs`；stats 增加 `survival_rate`、`alpha`；`_last_stats` 同步（存 `self.last_survival_rate`/`self.last_alpha`）

- [ ] **Step 1: 写失败测试（按用户设计文档 4 个测试 + 集成断言）**

```python
def test_survival_alpha():
    assert eco.survival_alpha(100, 100) == 1.0
    assert abs(eco.survival_alpha(50, 100) - 2.0) < 1e-9
    assert eco.survival_alpha(0, 100) == 1.0          # 防御
    assert eco.survival_alpha(0, 0) == 1.0

def test_assortative_pairing_tends_similar_age():
    rng = np.random.default_rng(11)
    from eco_engine import Genome
    mk = lambda name, age: Genome(name=name, layers=[np.zeros((2,2))],
                                   readout=np.zeros((2,2)), age=age)
    pops = [mk(f"a{i}", 1) for i in range(30)] + [mk(f"b{i}", 20) for i in range(30)]
    def avg_delta(s):
        pairs = eco.assortative_pairs(pops, s, np.random.default_rng(3))
        return np.mean([abs(x.age - y.age) for x, y in pairs])
    d0, d1 = avg_delta(0.0), avg_delta(1.0)
    assert d1 < d0 * 0.6, f"s=1 应显著同龄配对: s0={d0:.2f} s1={d1:.2f}"

def test_stats_has_survival_rate_alpha():
    e = eco.Ecosystem(seed=0)
    _, stats = e.step_round()
    assert "survival_rate" in stats and "alpha" in stats
    assert 0.0 <= stats["survival_rate"] <= 1.0
    if stats["survival_rate"] > 0:
        assert abs(stats["alpha"] - 1.0 / stats["survival_rate"]) < 1e-6
    s = e.get_state()
    assert "survival_rate" in s["stats"] and "alpha" in s["stats"]

def test_config_assort_strength():
    e = eco.Ecosystem(seed=2)
    e.set_config(assort_strength=0.8)
    assert abs(e.assort_strength - 0.8) < 1e-9
    e.set_config(assort_strength=-0.1)     # 非法忽略
    assert abs(e.assort_strength - 0.8) < 1e-9
    e.set_config(assort_strength=1.5)
    assert abs(e.assort_strength - 0.8) < 1e-9
```

- [ ] **Step 2: 运行确认失败**

Run: `python eco_tests.py`
Expected: `AttributeError`（`survival_alpha`/`assortative_pairs`/`assort_strength` 不存在）。

- [ ] **Step 3: 写实现（eco_engine.py）**

3a. 纯函数：
```python
ASSORT_TAU = 2.0
def survival_alpha(n_survivors, n_start) -> float:
    return 1.0 / n_survivors * n_start if n_survivors > 0 else 1.0

def assortative_pairs(survivors, s, rng):
    if s <= 0:
        rng.shuffle(survivors)
        return [(survivors[j], survivors[j+1]) for j in range(0, len(survivors)-1, 2)]
    pairs = []
    work = list(survivors)
    rng.shuffle(work)
    while len(work) >= 2:
        a = work[0]
        rest = work[1:]
        ages = np.array([g.age for g in rest], np.float64)
        d = np.abs(ages - a.age)
        w = (1.0 - s) + s * np.exp(-d / ASSORT_TAU)
        w = w / w.sum()
        b_idx = int(rng.choice(len(rest), p=w))
        b = rest.pop(b_idx)
        pairs.append((a, b))
        work = rest
    return pairs
```

3b. `Ecosystem`：`__init__` 加 `self.assort_strength = 0.5`、`self.last_survival_rate = 0.0`、`self.last_alpha = 1.0`；`set_config` 校验 `0 ≤ assort_strength ≤ 1`；`_config` 返回 `assort_strength`。

3c. `step_round` 繁殖阶段（替换 `eco_engine.py:311-330`）：
```python
start_pop = len(self.pop)
survival_rate = len(survivors) / start_pop if start_pop > 0 else 0.0
alpha = survival_alpha(len(survivors), start_pop)
self.last_survival_rate, self.last_alpha = survival_rate, alpha
pairs = assortative_pairs(survivors, self.assort_strength, self.rng)
births = []
if pairs:
    density = max(DENSITY_FLOOR, 1.0 - len(self.pop) / self.capacity)
    brood = int(round(self.survival_rounds * self.n_repro * alpha * density))
    room = max(0, self.capacity - len(survivors))
    target = min(room, len(pairs) * brood)
    for k in range(target):
        a, b = pairs[k % len(pairs)]
        child = crossover(a, b, self.rng, weight_a=float(a.age), weight_b=float(b.age))
        ...
```

3d. `round_end` stats 增加 `survival_rate`/`alpha`；`_last_stats` 补 `avg_acc` 与 `survival_rate`/`alpha`（`avg_acc` 存 `self.last_avg_acc`）。

- [ ] **Step 4: 运行确认通过**

Run: `python eco_tests.py`
Expected: 全部通过（含用户设计 4 个测试）。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): 选型交配(assort_strength 软强度) + 存活奖励 alpha(1/survival_rate 全局) ——承接用户设计文档"
git push origin main
```

---

### Task 4: 统计与架构 API —— history / age_hist / newborn / arch / config 透传

**Files:**
- Modify: `eco_engine.py`、`eco_server.py`、`eco_tests.py`

**Interfaces:**
- Consumes: `Ecosystem`（Task 1-3 产物）
- Produces:
  - `Ecosystem.history: list[dict]`（每回合 `{round, alive, avg_acc, natural_rate, survival_rate, alpha}`）
  - `round_end` stats 增加 `age_hist: list[int]`（长度 survival_rounds，下标 i=存活 i+1 回合）与 `newborns: int`
  - `get_state()` → 增加 `history`；population 条目增加 `arch`；stats 含 `age_hist`/`newborns`
  - `birth` 事件增加 `arch`
  - `/api/config` 支持 `assort_strength`（Task 3 已加引擎侧，服务端透传即可）

- [ ] **Step 1: 写失败测试**

```python
def test_history_and_age_hist():
    e = eco.Ecosystem(seed=0)
    for _ in range(3):
        e.step_round()
    st = e.get_state()
    assert "history" in st and len(st["history"]) == 3
    assert set(st["history"][-1]) >= {"round","alive","avg_acc","natural_rate","survival_rate","alpha"}
    s = st["stats"]
    assert "age_hist" in s and len(s["age_hist"]) == e.survival_rounds
    assert "newborns" in s
    assert sum(s["age_hist"]) + s["newborns"] == len(e.pop), "直方图+新生应覆盖全部存活"
    # 每个 population 条目带 arch
    for p in st["population"]:
        assert isinstance(p["arch"], list) and all(isinstance(n, int) for n in p["arch"])

def test_birth_event_has_arch():
    e = eco.Ecosystem(seed=5)
    events, _ = e.step_round()
    births = [ev for ev in events if ev["type"] == "birth"]
    assert births and all("arch" in b for b in births)
```

- [ ] **Step 2: 运行确认失败**

Run: `python eco_tests.py`
Expected: `KeyError`/`AssertionError`（history/age_hist/arch 未实现）。

- [ ] **Step 3: 写实现**

3a. `eco_engine.py`：
- `Ecosystem.__init__` 加 `self.history = []`。
- `step_round` 末尾、`self.round += 1` 前：计算 `age_hist`（遍历 `self.pop`，`1 ≤ g.age ≤ survival_rounds` → `age_hist[g.age-1]+=1`）与 `newborns`（`g.age == 0` 计数）；`stats` 增加 `age_hist`/`newborns`；`self.history.append({round, alive, avg_acc, natural_rate, survival_rate, alpha})`。
- `birth` 事件增加 `"arch": c.arch()`。
- `get_state()`：population 条目增加 `"arch": g.arch()`；返回 `"history": self.history`；stats 用 `_last_stats`（含 age_hist/newborns 当前快照）。
- `_last_stats()`：补 `avg_acc`（`self.last_avg_acc`）、`survival_rate`、`alpha`、`age_hist`、`newborns`（用当前 `self.pop` 现算）。
- 说明：`age_hist` 长度随 `survival_rounds` 变（滑条 10-30），前端按 stats 返回长度渲染。

3b. `eco_server.py`：`/api/config` 已透传 `set_config(**body)`（Task 3 引擎已支持 assort_strength），无需改；确认 `get_state`/`step_round` 原样返回新键。仅在 `test_server_endpoints` 增加 `assort_strength` 与 `history`/`arch` 断言。

- [ ] **Step 4: 运行确认通过**

Run: `python eco_tests.py`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_server.py eco_tests.py
git commit -m "feat(eco): 服务端统计历史(history) + 存活回合直方图(age_hist/newborns) + 个体架构(arch) + config 透传 assort_strength"
git push origin main
```

---

### Task 5: 前端 —— 曲线历史预填充、存活直方图、头部读数、选型滑条、解剖架构图

**Files:**
- Modify: `eco_game.html`

**Interfaces:**
- Consumes: `/api/state`（`history`、`stats.age_hist/newborns/survival_rate/alpha`、population.`arch`、config.`assort_strength`）、`round_end.stats`（同上）、`birth.arch`
- Produces: 单文件前端

- [ ] **Step 1: 结构改动**

1a. **曲线历史预填充 + 去 n<2 门槛**：
- `loadState`：`const hist = s.history || []`，用最后 200 条预填充 `correctSeries/popSeries/naturalSeries`（`alive/capacity` 归一化放 popSeries）。
- `drawCurves`：`if (n < 1) { 画占位文字"等待推演…"; return; }`；n=1 画单点（`moveTo+lineTo` 同点即可）。
- `_last_stats` 现在含 `avg_acc` → 头部正确率不再 "—"。

1b. **存活直方图**：种群统计面板在曲线 canvas 下新增 `#hist` canvas（或复用绘图函数）：
```js
function drawAgeHist(hist, newborns, cap){
  // hist: 长度=survival_rounds 的数组，x 轴=1..len，y 轴=个体数
  // 顶部标注"新生 N 只"；bar 用 var(--accent)，当前最大 bar 用 var(--ok)
}
```
- 每回合 `replayEvents` 末尾、`loadState`/`refreshPopulation` 用 `state.stats.age_hist`/`newborns` 调用。
- `pushCurves`/历史曲线逻辑不动（3 条曲线保持）。

1c. **头部读数**：`updateHeader` 追加：`存活率 <b>X%</b> · α <b>W</b>`（`survival_rate*100` 1 位小数；`alpha` 1 位小数；无则 "—"）。

1d. **选型滑条**：环境参数面板加一行"选型强度 s"：`<input id="c_assort" type="range" min="0" max="100" step="5" value="50">` → `bindConfig("c_assort","assort_strength","v_assort")`（值 /100）；`syncConfigSliders` 反向 ×100。

1e. **解剖架构图**：`showAnatomy` 在现有信息上方渲染架构：
```js
const arch = g.arch || [];
// 输入 784 → 层块(各 n) → 产出 10；每层一个色块标神经元数；多层(arch.length>1)用紫、单层灰
// arch 来源：state.pop 重建时从 population 条目带；birth 时从 birth.arch 带
```
- `rebuildPop`：population 条目存 `arch: p.arch`。
- `replayEvents` birth：`arch: b.arch`。
- `manual_feed` 结果区不展示 layer_counts（保持现状只显示 readout 分布）。

- [ ] **Step 2: 语法自检**

```bash
python -c "import re;s=open('eco_game.html',encoding='utf-8').read();m=re.findall(r'<script>(.*?)</script>',s,re.S);open('_eco_js_check.js','w',encoding='utf-8').write(m[0])"
node --check _eco_js_check.js && rm _eco_js_check.js
python eco_tests.py
```

- [ ] **Step 3: 无头验证 + 提交**

用无头 Edge（`--headless --dump-dom` / 截图）验证：页面加载即显示历史曲线、直方图、头部读数；点击色块显示架构图。确认后：
```bash
git add eco_game.html
git commit -m "feat(eco): 前端统计面板(历史曲线预填充+存活直方图) + 头部存活率/alpha + 选型滑条 + 解剖架构图"
git push origin main
```

---

### Task 6: 冒烟 + 诚实基线 + README

**Files:**
- Modify: `_eco_smoke.py`、`README.md`

- [ ] **Step 1: 改写冒烟脚本**

`_eco_smoke.py` 增加输出：每回合 `age_hist` 均值/多层占比（`arch().length>1` 比例）、`alpha`、`survival_rate`、单回合耗时；30 回合后汇总诚实基线（正确率 vs 随机 0.10、多层占比演化、natural_rate vs 0.95、耗时）。

- [ ] **Step 2: 运行冒烟并如实记录**

Run: `python _eco_smoke.py`
Expected: 无崩溃；观察并记录：
- **多层占比**：初始 0%，随繁殖是否出现多层个体、是否持续存活（预期出现但多数早期死亡）。
- **正确率**：贴随机线（诚实报告）。
- **natural_rate / alpha**：alpha≈10（约 10% 存活率）、natural_rate 极低（停止条件不可达）。
- **单回合耗时**：随种群与多层占比增长；若满容量超 ~8s，报告并评估 MAX_HIDDEN/层数上限是否需收紧。

- [ ] **Step 3: README 生态游戏章节更新**

- 繁殖：选型交配（选型强度 s）+ 存活奖励 alpha（=1/存活率）+ 存活加权交叉 + 五类结构突变 + 密度承载力。
- 生物体：架构可变（1-4 隐藏层，初始单层 100），多层由变异产生。
- 统计：种群统计面板（历史曲线 + 存活回合数直方图）、个体解剖（架构图）。
- 诚实预期段落更新。

- [ ] **Step 4: 提交 + 交付报告**

```bash
git add _eco_smoke.py README.md
git commit -m "docs(eco): README 更新——多层结构突变/选型交配/存活alpha/统计面板 + 冒烟诚实基线"
git push origin main
```

报告：玩法/参数变更、诚实结果（正确率、多层占比演化、natural_rate、alpha、耗时）、结论（多层是否出现并受选择、统计面板数据是否稳定、与上一版对比）。
