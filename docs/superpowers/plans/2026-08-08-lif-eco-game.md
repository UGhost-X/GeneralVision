# LIF 生态游戏（喂食-产出-淘汰-有性繁殖）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可视化生态游戏：LIF 生物体吃数字、产出数字，吃对率高的存活并做有性繁殖（权重随基因交叉/变异遗传），吃对率低的淘汰。

**Architecture:** 纯 numpy 生态引擎 `eco_engine.py`（无学习、权重出生即固定）驱动有性繁殖演化，产出标准化事件流；`eco_server.py`（stdlib http.server）按需推演每天的事件并托管前端；`eco_game.html`（单文件原生 JS + Canvas）播放培养皿动画、手动喂食、统计曲线与个体解剖视图。引擎/服务/前端通过事件 JSON 解耦。

**Tech Stack:** Python 3.10（仅 numpy，不依赖 torch——引擎独立自包含）、Python stdlib `http.server`、单文件 HTML/CSS/原生 JS（无外部依赖）。

## Global Constraints

- Python 3.10.20，仅允许 numpy 与标准库；引擎不 import torch/snn.py（保持轻量与自包含，泊松编码公式与 snn.py 一致：`rand < pixels*0.6`）。
- 复用现有 `data_loading.load_mnist()`（返回 `(train_img, train_lbl, test_img, test_lbl)`，图像 `(N,784)` float32 ∈[0,1]）。
- **不改动任何现有文件**（`snn.py`、`evolve.py`、`data_loading.py`、并行会话的 `export_snn_demo.py`/`snn_weights.js` 等一律不动）。只新建文件。
- 每次 commit 必须 `git push origin main`（CLAUDE.md 工作流）。`git add` 只用显式路径，**绝不** `git add -A`（避免卷入并行会话未跟踪文件）。
- 前端为单文件、无外部 CDN/字体/库；注释与文案用中文。
- 确定性：Ecosystem 以 seed 初始化，同 seed 全程可复现。

---

### Task 1: eco_engine 骨架 —— 基因组、随机权重、有性繁殖交叉

**Files:**
- Create: `eco_engine.py`
- Test: `eco_tests.py`

**Interfaces:**
- Consumes: `data_loading.load_mnist()`
- Produces:
  - 常量：`POP_CAP=60, FOOD_COUNT=50, T=200, SPIKE_GAIN=0.6, LEAK=0.94, HIDDEN_SIZE=100, READOUT_SIZE=10, REF_PERIOD=4, THETA_HIDDEN, THETA_READOUT, W_NORM_HIDDEN=16.0, W_NORM_READOUT=3.0, W_INIT_RANGE=0.2, CROSS_SIGMA=0.01, MUT_RATE=0.001, BOTTOM_DEATH=0.30, MAX_AGE=15, INIT_POP=40`
  - `@dataclass Genome`: 字段 `name:str, hidden:np.ndarray(784,100), readout:np.ndarray(100,10), born_gen:int=0, age:int=0, parents:tuple|None=None`
  - `random_genome(name:str, rng, gen:int=0) -> Genome`
  - `crossover(a:Genome, b:Genome, rng) -> Genome`（逐权重 50/50 取父/母 + N(0,CROSS_SIGMA) + MUT_RATE 概率重置 + 列归一化）

- [ ] **Step 1: 写失败测试**

```python
# eco_tests.py
"""LIF 生态引擎测试。运行：python eco_tests.py"""
import numpy as np
import eco_engine as eco

def test_crossover_mixes_both_parents():
    rng = np.random.default_rng(7)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child = eco.crossover(a, b, rng)
    assert child.hidden.shape == (784, 100), child.hidden.shape
    assert child.readout.shape == (100, 10), child.readout.shape
    # 逐权重取父/母：每个权重应"更接近"其中一方（噪声 σ=0.01 远小于双亲差距）
    for Wc, Wa, Wb in [(child.hidden, a.hidden, b.hidden),
                       (child.readout, a.readout, b.readout)]:
        closer_a = (np.abs(Wc - Wa) < np.abs(Wc - Wb)).mean()
        assert 0.35 < closer_a < 0.65, f"closer_a={closer_a}"

def test_columns_normalized():
    g = eco.random_genome("n", np.random.default_rng(0))
    norms = np.linalg.norm(g.hidden, axis=0)
    assert np.allclose(norms, eco.W_NORM_HIDDEN, atol=1e-4), norms[:5]
    rnorms = np.linalg.norm(g.readout, axis=0)
    assert np.allclose(rnorms, eco.W_NORM_READOUT, atol=1e-4), rnorms[:5]

def test_mutation_is_rare():
    rng = np.random.default_rng(11)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child = eco.crossover(a, b, rng)
    d_a = np.abs(child.hidden - a.hidden)
    d_b = np.abs(child.hidden - b.hidden)
    from_a = (d_a < d_b).mean()                 # 一半来自 a，一半来自 b
    both_far = ((d_a > 0.3) & (d_b > 0.3)).mean()  # 大突变应极罕见
    assert 0.35 < from_a < 0.65, from_a
    assert both_far < 0.05, both_far

def test_genome_serialize():
    g = eco.random_genome("s", np.random.default_rng(3))
    d = {"name": g.name, "hidden": g.hidden.tolist(), "readout": g.readout.tolist(),
         "born_gen": g.born_gen, "age": g.age}
    import json
    s = json.dumps(d)
    assert "hidden" in s and "readout" in s
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python eco_tests.py`
Expected: `ModuleNotFoundError: No module named 'eco_engine'`

- [ ] **Step 3: 写最小实现**

```python
# eco_engine.py（本任务先写：常量、Genome、normalize、random_genome、crossover）
"""LIF 生态游戏引擎：喂食-产出-淘汰-有性繁殖（纯 numpy，一生无学习）。

生物体 = LIF 网络（784 输入 → 100 隐藏神经元 → 10 产出神经元），所有权重
出生即随机、一生固定。适应度 = 每天吃对率。有性繁殖 = 逐权重 50/50 取双亲
+ 高斯扰动 + 千分之一大突变。输出标准化事件流供前端播放。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# ---- 游戏与网络参数（阈值在 Task 2 校准） ----
POP_CAP = 60
FOOD_COUNT = 50
T = 200
SPIKE_GAIN = 0.6
LEAK = 0.94
HIDDEN_SIZE = 100
READOUT_SIZE = 10
REF_PERIOD = 4
THETA_HIDDEN = 12.0
THETA_READOUT = 1.5
W_NORM_HIDDEN = 16.0
W_NORM_READOUT = 3.0
W_INIT_RANGE = 0.2
CROSS_SIGMA = 0.01
MUT_RATE = 0.001
BOTTOM_DEATH = 0.30
MAX_AGE = 15
INIT_POP = 40


@dataclass
class Genome:
    """一个生物体：架构固定（784→100→10），权重即基因。"""
    name: str
    hidden: np.ndarray            # (784, 100) 输入→隐藏
    readout: np.ndarray           # (100, 10)  隐藏→产出
    born_gen: int = 0
    age: int = 0
    parents: tuple | None = None


def _normalize_cols(W: np.ndarray, norm: float) -> np.ndarray:
    """每列归一化到指定 L2 范数（保证随机权重动力学不爆/不哑）。"""
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-8)
    return W * (norm / col_norms)


def _random_weights(n_in: int, n_out: int, norm: float, rng: np.random.Generator) -> np.ndarray:
    W = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, (n_in, n_out))
    return _normalize_cols(W, norm)


def random_genome(name: str, rng: np.random.Generator, gen: int = 0) -> Genome:
    return Genome(
        name=name,
        hidden=_random_weights(784, HIDDEN_SIZE, W_NORM_HIDDEN, rng),
        readout=_random_weights(HIDDEN_SIZE, READOUT_SIZE, W_NORM_READOUT, rng),
        born_gen=gen,
    )


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """有性繁殖：逐权重 50/50 取父/母 + 全体高斯扰动 + 千分之一大突变。"""
    hidden = np.where(rng.random(a.hidden.shape) < 0.5, a.hidden, b.hidden)
    hidden += rng.normal(0.0, CROSS_SIGMA, hidden.shape)
    mh = rng.random(hidden.shape) < MUT_RATE
    hidden[mh] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(mh.sum()))
    hidden = _normalize_cols(hidden, W_NORM_HIDDEN)

    readout = np.where(rng.random(a.readout.shape) < 0.5, a.readout, b.readout)
    readout += rng.normal(0.0, CROSS_SIGMA, readout.shape)
    mr = rng.random(readout.shape) < MUT_RATE
    readout[mr] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(mr.sum()))
    readout = _normalize_cols(readout, W_NORM_READOUT)

    return Genome(name="child", hidden=hidden, readout=readout,
                  parents=(a.name, b.name))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python eco_tests.py`
Expected: 全部断言通过，无输出或打印 "ALL TESTS PASSED"。若 `test_crossover_mixes_both_parents` 的区间断言因噪声不满足，将阈值放宽到 `0.30 < frac < 0.70`（样本量 78400 下 50/50 的标准差 ~0.18%，放宽是为防止个别种子巧合）。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): 生态引擎骨架——基因组、随机权重、有性繁殖交叉"
git push origin main
```

---

### Task 2: eco_engine 前向 —— LIF 放电 + 产出数字 + 阈值校准

**Files:**
- Modify: `eco_engine.py`
- Test: `eco_tests.py`

**Interfaces:**
- Produces: `forward(genome: Genome, pixels: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]` 返回 `(produced[B] int, hidden_counts (B,100) int, readout_counts (B,10) int)`；`produced=-1` 表示产出层整场未发放。

- [ ] **Step 1: 写失败测试**

```python
def test_forward_shapes_and_deterministic():
    rng = np.random.default_rng(5)
    g = eco.random_genome("f", rng)
    pix = np.zeros((3, 784), np.float32); pix[0, 200:250] = 1.0
    produced, hc, rc = eco.forward(g, pix, np.random.default_rng(5))
    assert produced.shape == (3,) and hc.shape == (3, 100) and rc.shape == (3, 10)
    assert produced.dtype == np.int64 and hc.dtype == np.int64
    produced2, _, _ = eco.forward(g, pix, np.random.default_rng(5))
    assert np.array_equal(produced, produced2), "同 seed 应可复现"

def test_forward_firing_sane():
    """对随机 digit 批次：隐藏层应发放（非全零）、产出不应过于集中/未发放过多。"""
    from data_loading import load_mnist
    ti, tl, _, _ = load_mnist()
    rng = np.random.default_rng(1)
    g = eco.random_genome("f2", rng)
    idx = rng.integers(0, len(ti), 40)
    produced, hc, rc = eco.forward(g, ti[idx], np.random.default_rng(2))
    assert hc.sum() > 0, "隐藏层整场无发放——动力学哑了"
    none_frac = float((produced == -1).mean())
    assert none_frac < 0.7, f"产出层未发放比例过高 {none_frac:.2f}"
    real = produced[produced != -1]
    assert len(np.unique(real)) >= 3, f"产出数字过于集中: {np.unique(real)}"
    assert (rc.sum(axis=0) > 0).sum() >= 2, f"产出层几乎只有单一通道发放: {rc.sum(axis=0)}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python eco_tests.py`
Expected: `AttributeError: module 'eco_engine' has no attribute 'forward'`

- [ ] **Step 3: 写实现（追加到 eco_engine.py）**

```python
def forward(genome: Genome, pixels: np.ndarray, rng: np.random.Generator
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pixels: (B,784) ∈[0,1]。返回 (produced, hidden_counts, readout_counts)。

    泊松编码 → 隐藏层 LIF+WTA（无 STDP、无 homeostasis，纯放电）→ 产出层 LIF。
    produced = 产出层累计发放最多的数字；-1 表示整场未发放。
    逐样本重置 V/refr，与 snn.py step() 语义一致。
    """
    B = pixels.shape[0]
    S = (rng.random((B, 784, T)) < (pixels[:, :, None] * SPIKE_GAIN)).astype(np.float32)
    Vh = np.zeros((B, HIDDEN_SIZE), np.float32)
    refh = np.zeros((B, HIDDEN_SIZE), np.int32)
    Vr = np.zeros((B, READOUT_SIZE), np.float32)
    refr = np.zeros((B, READOUT_SIZE), np.int32)
    hc = np.zeros((B, HIDDEN_SIZE), np.int64)
    rc = np.zeros((B, READOUT_SIZE), np.int64)
    Wh, Wr = genome.hidden, genome.readout
    for t in range(T):
        # ---- 隐藏层 ----
        Vh += S[:, :, t] @ Wh
        Vh[refh > 0] = 0.0
        Vh *= LEAK
        elig = (refh <= 0) & (Vh >= THETA_HIDDEN)
        fire_rows = np.nonzero(elig.any(axis=1))[0]
        hspk = np.zeros((B, HIDDEN_SIZE), np.float32)
        if fire_rows.size:
            win = np.where(elig, Vh, -np.inf).argmax(axis=1)[fire_rows]
            Vh[fire_rows] = 0.0
            was_idle = refh[fire_rows] <= 0
            refh[fire_rows] = np.where(was_idle, 1, refh[fire_rows])
            refh[fire_rows, win] = REF_PERIOD
            hspk[fire_rows, win] = 1.0
            hc[fire_rows, win] += 1
        refh = np.maximum(refh - 1, 0)
        # ---- 产出层 ----
        Vr += hspk @ Wr
        Vr[refr > 0] = 0.0
        Vr *= LEAK
        eligr = (refr <= 0) & (Vr >= THETA_READOUT)
        fire_r = np.nonzero(eligr.any(axis=1))[0]
        if fire_r.size:
            winr = np.where(eligr, Vr, -np.inf).argmax(axis=1)[fire_r]
            Vr[fire_r] = 0.0
            was_idle_r = refr[fire_r] <= 0
            refr[fire_r] = np.where(was_idle_r, 1, refr[fire_r])
            refr[fire_r, winr] = REF_PERIOD
            rc[fire_r, winr] += 1
        refr = np.maximum(refr - 1, 0)
    produced = np.where(rc.sum(axis=1) > 0, rc.argmax(axis=1), -1)
    return produced, hc, rc
```

- [ ] **Step 4: 运行测试，若动力学退化则校准阈值**

Run: `python eco_tests.py`
Expected:
- 若 `test_forward_firing_sane` 报"隐藏层整场无发放"→ 降低 `THETA_HIDDEN`（如 12→8）或提高 `SPIKE_GAIN`，重跑。
- 若报"产出层几乎只有单一通道发放"→ 降低 `THETA_READOUT`（如 1.5→0.8）或降低 `W_NORM_READOUT`，重跑。
- 若报"产出过于集中/未发放过多"→ 同理微调两阈值。
- 校准原则：隐藏层每场总发放 ~20-120，产出未发放比例 <20%，产出通道分布不全集中。**把最终校准后的常量值写回 eco_engine.py 顶部注释并提交**。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): LIF 前向——泊松编码+隐藏层放电+产出层读数，阈值校准"
git push origin main
```

---

### Task 3: eco_engine 生态循环 —— 每天喂食、判定、淘汰、有性繁殖、事件流

**Files:**
- Modify: `eco_engine.py`
- Test: `eco_tests.py`

**Interfaces:**
- Produces:
  - `class Ecosystem`: `__init__(self, seed:int=0)`；`step_day() -> tuple[list[dict], dict]`；`get_state() -> dict`；`manual_feed(name:str, digit:int) -> dict`；`get_digit_image(idx:int) -> dict`
  - 事件 schema（前端消费）：`day_begin {type,day,food_idx:[50],food_labels:[50]}`；`org_day {type,name,acc,produced:[50],readout_profile:[10]}`；`death {type,name}`；`birth {type,name,parents:[2],gen}`；`day_end {type,stats}`
  - `stats = {day, alive, avg_acc, best_acc, best_name, median_acc, worst_acc}`
  - `get_state() = {day, config:{...}, population:[{name,age,born_gen,parents,alive}], stats}`

- [ ] **Step 1: 写失败测试**

```python
def test_day_loop_invariants():
    eco_ = eco.Ecosystem(seed=0)
    assert len(eco_.pop) == eco.INIT_POP
    for day in range(3):
        events, stats = eco_.step_day()
        types = [e["type"] for e in events]
        assert types[0] == "day_begin" and types[-1] == "day_end"
        assert len(eco_.pop) == eco.POP_CAP, f"day{day} 种群 {len(eco_.pop)}"
        assert 0.0 <= stats["avg_acc"] <= 1.0
        assert stats["alive"] == eco.POP_CAP
        names = {g.name for g in eco_.pop}
        assert len(names) == eco.POP_CAP, "重名"
        for e in events:
            if e["type"] == "org_day":
                assert len(e["produced"]) == eco.FOOD_COUNT
                assert len(e["readout_profile"]) == eco.READOUT_SIZE
            if e["type"] == "birth":
                assert len(e["parents"]) == 2
    # 手动喂食
    best = max(eco_.pop, key=lambda g: eco_.stats_cache.get(g.name, 0))
    r = eco_.manual_feed(best.name, 3)
    assert r["label"] == 3 and r["produced"] in list(range(-1, 10))
    assert len(r["food_pixels"]) == 784 and len(r["readout_counts"]) == 10
    # 可复现：同 seed 重建，前 2 天轨迹应一致（_day_fingerprint 记录 avg 与种群名）
    def _run2(seed):
        e = eco.Ecosystem(seed=seed)
        out = []
        for _ in range(2):
            e.step_day()
            out.append(e._day_fingerprint())
        return out
    assert _run2(9) == _run2(9), "同 seed 应可复现"
```

`stats_cache` 与 `_day_fingerprint` 是引擎内部成员（Step 3 定义）：`stats_cache: dict[str,float]` 记录每体最近一天吃对率；`_day_fingerprint()` 返回当日 avg_acc 与种群名的元组。

- [ ] **Step 2: 运行测试确认失败**

Run: `python eco_tests.py`
Expected: `AttributeError: module 'eco_engine' has no attribute 'Ecosystem'`

- [ ] **Step 3: 写实现（追加到 eco_engine.py）**

```python
class Ecosystem:
    """生态主循环。同 seed 全程可复现（所有随机性来自 self.rng 顺序抽取）。"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.counter = 0
        self.day = 0
        self.stats_cache: dict[str, float] = {}
        self._img, self._lbl, _, _ = load_mnist()
        self.pop: list[Genome] = [
            random_genome(f"eco#{i}", self.rng) for i in range(INIT_POP)
        ]
        self.counter = INIT_POP

    # ---- 内部 ----
    def _next_name(self) -> str:
        n = f"eco#{self.counter}"
        self.counter += 1
        return n

    def _day_fingerprint(self) -> tuple:
        return (round(float(np.mean(list(self.stats_cache.values()))), 6),
                tuple(sorted(g.name for g in self.pop)))

    def step_day(self) -> tuple[list[dict], dict]:
        events: list[dict] = []
        food_idx = self.rng.integers(0, len(self._img), FOOD_COUNT)
        food_lbl = self._lbl[food_idx]
        events.append({"type": "day_begin", "day": self.day,
                       "food_idx": food_idx.tolist(),
                       "food_labels": food_lbl.tolist()})

        accs: dict[str, float] = {}
        for i, g in enumerate(self.pop):
            produced, _hc, rc = forward(g, self._img[food_idx],
                                        np.random.default_rng(self.day * 1_000_003 + i))
            acc = float((produced == food_lbl).mean())
            g.age += 1
            self.stats_cache[g.name] = acc
            accs[g.name] = acc
            events.append({"type": "org_day", "name": g.name,
                           "acc": round(acc, 4),
                           "produced": produced.tolist(),
                           "readout_profile": rc.mean(axis=0).round(2).tolist()})

        order = sorted(self.pop, key=lambda g: -accs[g.name])
        n_survive = max(1, len(order) - round(len(order) * BOTTOM_DEATH))  # 存活顶部 70%
        bottom = set(id(g) for g in order[n_survive:])                     # 底部 30% 饿死
        aged = set(id(g) for g in order[:n_survive] if g.age > MAX_AGE)    # 顶部高龄也走
        survivors = [g for g in self.pop if id(g) not in bottom and id(g) not in aged]
        for g in self.pop:
            if id(g) not in bottom and id(g) not in aged:
                continue
            events.append({"type": "death", "name": g.name})
            self.stats_cache.pop(g.name, None)

        need = POP_CAP - len(survivors)
        if need > 0 and survivors:
            w = np.array([accs[g.name] + 1e-6 for g in survivors], np.float64)
            probs = w / w.sum()
            while len(survivors) < POP_CAP:
                if len(survivors) >= 2:
                    p1, p2 = self.rng.choice(len(survivors), 2, replace=False, p=probs)
                else:
                    p1 = p2 = 0
                child = crossover(survivors[p1], survivors[p2], self.rng)
                child.name = self._next_name()
                child.born_gen = self.day
                survivors.append(child)
                events.append({"type": "birth", "name": child.name,
                               "parents": [child.parents[0], child.parents[1]],
                               "gen": child.born_gen})
        self.pop = survivors

        acc_arr = np.array([accs.get(g.name, 0.0) for g in self.pop], np.float64)
        stats = {
            "day": self.day,
            "alive": len(self.pop),
            "avg_acc": round(float(acc_arr.mean()), 4),
            "best_acc": round(float(acc_arr.max()), 4),
            "best_name": self.pop[int(acc_arr.argmax())].name,
            "median_acc": round(float(np.median(acc_arr)), 4),
            "worst_acc": round(float(acc_arr.min()), 4),
        }
        events.append({"type": "day_end", "stats": stats})
        self.day += 1
        return events, stats

    def get_state(self) -> dict:
        return {
            "day": self.day,
            "config": {"pop_cap": POP_CAP, "food_count": FOOD_COUNT, "T": T,
                       "max_age": MAX_AGE, "bottom_death": BOTTOM_DEATH,
                       "mutation_rate": MUT_RATE},
            "population": [{"name": g.name, "age": g.age, "born_gen": g.born_gen,
                            "parents": list(g.parents) if g.parents else None,
                            "alive": True} for g in self.pop],
            "stats": self._last_stats,
        }

    def get_digit_image(self, idx: int) -> dict:
        return {"pixels": self._img[idx].tolist(), "label": int(self._lbl[idx])}

    def manual_feed(self, name: str, digit: int) -> dict:
        g = next(x for x in self.pop if x.name == name)
        cand = np.nonzero(self._lbl == digit)[0]
        idx = int(self.rng.choice(cand))
        produced, hc, rc = forward(g, self._img[idx][None],
                                   np.random.default_rng(int(self.rng.integers(0, 2**31))))
        return {"food_pixels": self._img[idx].tolist(), "label": digit,
                "produced": int(produced[0]),
                "correct": bool(produced[0] == digit),
                "hidden_counts": hc[0].tolist(),
                "readout_counts": rc[0].tolist()}

    @property
    def _last_stats(self) -> dict:
        if not self.pop:
            return {"day": self.day, "alive": 0, "avg_acc": 0.0, "best_acc": 0.0,
                    "best_name": "", "median_acc": 0.0, "worst_acc": 0.0}
        acc_arr = np.array([self.stats_cache.get(g.name, 0.0) for g in self.pop], np.float64)
        return {"day": self.day, "alive": len(self.pop),
                "avg_acc": round(float(acc_arr.mean()), 4),
                "best_acc": round(float(acc_arr.max()), 4),
                "best_name": self.pop[int(acc_arr.argmax())].name,
                "median_acc": round(float(np.median(acc_arr)), 4),
                "worst_acc": round(float(acc_arr.min()), 4)}
```

注意：`forward` 内部用 `rng.random((B,784,T))` 一次性生成全部泊松脉冲，内存峰值 = `FOOD_COUNT×784×200×4B ≈ 31MB`/次调用，可接受。Step 3 中 `forward` 的 rng 由 day + name 派生，保证跨世代确定性（同 seed 复现）。

- [ ] **Step 4: 运行测试确认通过**

Run: `python eco_tests.py`
Expected: 全部断言通过。若 `test_day_loop_invariants` 中 `best = max(eco_.pop, ...)` 因 `stats_cache` 全为 0 仍取到第一只，可接受（manual_feed 只要求存在该 name）。若运行过慢（>30s），确认单日耗时（打印 `time`），供 Task 4 优化决策。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): 生态循环——每日喂食判定、底部淘汰+寿命、有性繁殖回填、事件流"
git push origin main
```

---

### Task 4: 引擎冒烟跑 —— 校准参数、量测耗时、记录诚实基线

**Files:**
- Create: `_eco_smoke.py`
- Modify: `eco_engine.py`（如需调 FOOD_COUNT/T）

**Interfaces:**
- Consumes: `Ecosystem`
- Produces: 运行 20 天，打印每日 `day alive avg best worst` 与单日耗时；记录最优吃对率。

- [ ] **Step 1: 写冒烟脚本**

```python
# _eco_smoke.py
"""引擎冒烟跑：20 天，打印种群统计与耗时，验证生态动力学并记录诚实基线。"""
import time
import numpy as np
import eco_engine as eco

def main():
    eco_ = eco.Ecosystem(seed=0)
    t0 = time.time()
    prev_best = -1
    for d in range(20):
        t = time.time()
        events, stats = eco_.step_day()
        dt = time.time() - t
        names = sorted(set(g.name for g in eco_.pop))
        gens = np.array([g.born_gen for g in eco_.pop])
        ages = np.array([g.age for g in eco_.pop])
        print(f"day {stats['day']:>3} | alive={stats['alive']:>3} "
              f"avg={stats['avg_acc']:.3f} best={stats['best_acc']:.3f} "
              f"med={stats['median_acc']:.3f} worst={stats['worst_acc']:.3f} "
              f"| 唯一名={len(names)} 平均年龄={ages.mean():.1f} "
              f"最老={ages.max()} | {dt:.2f}s")
        if stats["best_acc"] > prev_best:
            prev_best = stats["best_acc"]
    print(f"\n=== 冒烟完成 === 最优吃对率={prev_best:.3f} "
          f"(随机基线≈0.10) 总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行冒烟**

Run: `python _eco_smoke.py`
Expected: 20 天跑完无崩溃；观察：
- **单日耗时**：若 >6s，把 `FOOD_COUNT` 降到 30 或 `T` 降到 150（写回 eco_engine.py 顶部并注明原因），重跑。
- **种群动力学**：`alive` 恒 60；`avg_acc` 应随天缓慢变化（可升可平）；`最老` 不应超过 MAX_AGE+1（说明寿命淘汰生效）。
- **诚实基线记录**：把最优吃对率记到本任务提交说明与最终报告里，如实呈现（很可能仅略高于随机 0.10，这符合设计预期的"生态游戏演示、不追超 STDP"）。

- [ ] **Step 3: 提交**

```bash
git add _eco_smoke.py eco_engine.py
git commit -m "chore(eco): 引擎冒烟跑——校准耗时/参数，记录诚实基线"
git push origin main
```

---

### Task 5: eco_server —— stdlib HTTP 服务与 API

**Files:**
- Create: `eco_server.py`
- Test: 追加到 `eco_tests.py`（`test_server_endpoints`）

**Interfaces:**
- Consumes: `Ecosystem`
- Produces:
  - `GET /` → `eco_game.html`
  - `GET /api/state` → Ecosystem.get_state()
  - `POST /api/step` → `{"day":N,"events":[...],"stats":{...}}`（引擎推演一天）
  - `GET /api/digit_image/<idx>` → Ecosystem.get_digit_image(idx)
  - `POST /api/manual_feed` body `{"digit":n,"name":?}` → Ecosystem.manual_feed(name or best, digit)
  - 命令行：`python eco_server.py --port 8765 --seed 0`

- [ ] **Step 1: 写失败测试**

```python
import threading, json, urllib.request, urllib.error
def test_server_endpoints():
    from eco_server import run_server_in_thread, PORT_DEFAULT
    port, thread = run_server_in_thread(seed=0)
    base = f"http://127.0.0.1:{port}"
    try:
        s = json.load(urllib.request.urlopen(base + "/api/state"))
        assert s["config"]["pop_cap"] == eco.POP_CAP
        assert len(s["population"]) == eco.INIT_POP
        req = urllib.request.Request(base + "/api/step", method="POST")
        r = json.load(urllib.request.urlopen(req))
        assert r["stats"]["alive"] == eco.POP_CAP
        assert r["events"][0]["type"] == "day_begin"
        img = json.load(urllib.request.urlopen(base + "/api/digit_image/0"))
        assert len(img["pixels"]) == 784
        body = json.dumps({"digit": 4, "name": s["population"][0]["name"]}).encode()
        rq = urllib.request.Request(base + "/api/manual_feed", data=body, method="POST",
                                    headers={"Content-Type": "application/json"})
        mf = json.load(urllib.request.urlopen(rq))
        assert mf["label"] == 4 and len(mf["readout_counts"]) == 10
        html = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert "<canvas" in html or "id=\"dish\"" in html
    finally:
        thread.shutdown(); thread.join(timeout=5)
```

`run_server_in_thread` 由 eco_server 提供：用 `ThreadingHTTPServer` 绑定端口 0（随机空闲端口），返回 (port, server_thread)。需要 `server.server_close()` 关闭。

- [ ] **Step 2: 运行测试确认失败**

Run: `python eco_tests.py`
Expected: `ModuleNotFoundError: No module named 'eco_server'`

- [ ] **Step 3: 写实现**

```python
# eco_server.py
"""LIF 生态游戏本地服务：托管前端 + 推演/状态 API（纯 stdlib http.server）。"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import eco_engine as eco

PORT_DEFAULT = 8765
ROOT = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(ROOT, "eco_game.html")


class EcoHandler(BaseHTTPRequestHandler):
    engine: eco.Ecosystem  # 类属性由 make_server 注入
    lock = threading.Lock()

    # ---- helpers ----
    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        p = self.path
        if p in ("/", "/index.html"):
            try:
                data = open(HTML, "rb").read()
            except FileNotFoundError:
                self._json({"error": "eco_game.html 不存在"}, 500); return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p == "/api/state":
            with self.lock:
                self._json(self.engine.get_state())
        elif p.startswith("/api/digit_image/"):
            idx = int(p.rsplit("/", 1)[1])
            with self.lock:
                self._json(self.engine.get_digit_image(idx))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/step":
            with self.lock:
                events, stats = self.engine.step_day()
            self._json({"day": stats["day"], "events": events, "stats": stats})
        elif self.path == "/api/manual_feed":
            body = self._read_body()
            digit = int(body.get("digit", 0))
            with self.lock:
                st = self.engine.get_state()
                name = body.get("name") or st["stats"]["best_name"]
                try:
                    self._json(self.engine.manual_feed(name, digit))
                except StopIteration:
                    self._json({"error": f"no organism named {name}"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):  # 静音访问日志，避免刷屏
        pass


def _build_server(seed: int = 0, port: int = 0):
    engine = eco.Ecosystem(seed=seed)
    handler = type("Handler", (EcoHandler,), {"engine": engine})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def run_server_in_thread(seed: int = 0, port: int = 0):
    """测试用：随机空闲端口，返回 (port, server)。"""
    server = _build_server(seed=seed, port=port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server.server_address[1], server


def main():
    ap = argparse.ArgumentParser(description="LIF 生态游戏服务")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    server = _build_server(seed=args.seed, port=args.port)
    print(f"LIF 生态游戏已启动: http://127.0.0.1:{args.port}  (seed={args.seed})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python eco_tests.py`
Expected: `test_server_endpoints` 通过（注意它内部调 `POST /api/step` 会推演一天，属于预期副作用）。若 `ThreadingHTTPServer` 端口被占用报错，重试或换端口。

- [ ] **Step 5: 提交**

```bash
git add eco_server.py eco_tests.py
git commit -m "feat(eco): HTTP 服务与 API——状态/推演/喂食/数字图像"
git push origin main
```

---

### Task 6: eco_game.html —— 界面骨架（培养皿网格、控制条、统计曲线）

**Files:**
- Create: `eco_game.html`

**Interfaces:**
- Consumes: `/api/state`, `/api/step`, `/api/digit_image/<idx>`, `/api/manual_feed`（本任务先接 state + step，手动喂食与解剖在 Task 7）
- Produces: 单文件前端，无外部依赖

- [ ] **Step 1: 写 HTML 骨架 + 核心 JS（状态加载、网格渲染、曲线绘制）**

```html
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>🧫 LIF 生态游戏</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#e6edf3;
          --ok:#3fb950; --bad:#f85149; --dim:#8b949e; --accent:#58a6ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.5 "Segoe UI", system-ui, sans-serif; }
  header { display:flex; align-items:center; gap:14px; padding:10px 18px;
           border-bottom:1px solid var(--line); position:sticky; top:0;
           background:var(--bg); z-index:5; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; }
  .btn { background:var(--panel); color:var(--txt); border:1px solid var(--line);
         border-radius:6px; padding:4px 10px; cursor:pointer; }
  .btn:hover { border-color:var(--accent); }
  #statsLabel { color:var(--dim); font-size:12px; }
  main { display:grid; grid-template-columns: 1fr 320px; gap:12px; padding:12px; }
  #dish { display:grid; grid-template-columns: repeat(auto-fill,minmax(118px,1fr));
          gap:8px; align-content:start; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:8px; position:relative; transition:opacity .6s, transform .4s; }
  .card.dead { opacity:.25; filter:grayscale(1); transform:scale(.92); }
  .card .nm { font-size:11px; color:var(--dim); }
  .card .big { font-size:30px; font-weight:700; text-align:center; margin:4px 0; }
  .card .bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .card .bar > i { display:block; height:100%; background:var(--ok); }
  .card .meta { font-size:10px; color:var(--dim); display:flex; justify-content:space-between; }
  #side { display:flex; flex-direction:column; gap:12px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
  #curves { width:100%; height:180px; }
  .feedbtn { width:34px; height:34px; margin:2px; font-size:15px; cursor:pointer;
             background:var(--panel); color:var(--txt); border:1px solid var(--line);
             border-radius:6px; }
  .feedbtn:hover { border-color:var(--accent); }
  #manualResult { margin-top:8px; font-size:13px; }
  #manualResult canvas { image-rendering:pixelated; background:#000; border-radius:4px; }
  .mm { font-size:11px; color:var(--dim); margin-top:6px; }
</style>
</head>
<body>
<header>
  <h1>🧫 LIF 生态游戏</h1>
  <span id="dayLabel" class="dim"></span>
  <button class="btn" id="playBtn">⏸ 自动</button>
  <button class="btn" id="stepBtn">⏭ 推演一天</button>
  <label>速度 <input id="speed" type="range" min="50" max="1500" value="600" style="width:120px"></label>
  <span id="statsLabel"></span>
</header>
<main>
  <div id="dish"></div>
  <aside id="side">
    <div class="panel"><b>种群统计</b>
      <canvas id="curves"></canvas>
      <div class="mm" id="curveLegend">蓝=平均吃对率　绿=最优吃对率　灰=存活数</div>
    </div>
    <div class="panel"><b>手动喂食</b>
      <div id="manualBtns"></div>
      <div id="manualResult"></div>
    </div>
    <div class="panel"><b>个体解剖</b><div id="anatomy">点击任意生物体查看</div></div>
  </aside>
</main>
<script>
"use strict";
const $ = s => document.querySelector(s);
const state = { day:0, pop:new Map(), stats:null, auto:true, speed:600,
                accSeries:[], bestSeries:[], aliveSeries:[] };

// ---------- 渲染 ----------
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function el(tag, cls, html){ const e=document.createElement(tag); if(cls)e.className=cls;
  if(html!=null)e.innerHTML=html; return e; }

function cardHtml(g){
  const acc = state.stats ? (state.pop.get(g.name)?.acc ?? 0) : 0;
  return `<div class="nm">${esc(g.name)}</div>
    <div class="big">${state.pop.get(g.name)?.lastProduced ?? '·'}</div>
    <div class="bar"><i style="width:${Math.round(acc*100)}%"></i></div>
    <div class="meta"><span>吃对 ${(acc*100).toFixed(0)}%</span><span>${g.age} 天</span></div>`;
}

function renderCards(){
  const dish = $("#dish"); dish.innerHTML = "";
  for (const g of state.pop.values()){
    const c = el("div", "card" + (g.alive ? "" : " dead"));
    c.dataset.name = g.name;
    c.innerHTML = cardHtml(g);
    c.addEventListener("click", () => showAnatomy(g.name));
    dish.appendChild(c);
  }
}

function drawCurves(){
  const cv = $("#curves"), ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * devicePixelRatio;
  const H = cv.height = cv.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio); ctx.clearRect(0,0,W,H);
  const n = state.accSeries.length; if (n < 2) return;
  const plot = (series, color) => {
    const cw = cv.clientWidth, ch = cv.clientHeight;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    series.forEach((v,i) => { const x = i/(n-1)*cw, y = ch - v*ch;
      i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); });
    ctx.stroke();
  };
  plot(state.accSeries, "#58a6ff");
  plot(state.bestSeries, "#3fb950");
}

function updateHeader(){
  $("#dayLabel").textContent = `第 ${state.day} 天`;
  const s = state.stats;
  $("#statsLabel").innerHTML = s
    ? `种群 <b>${s.alive}</b> · 平均吃对率 <b>${(s.avg_acc*100).toFixed(1)}%</b> · 最优 <b>${esc(s.best_name)} ${(s.best_acc*100).toFixed(1)}%</b>`
    : "加载中…";
}

async function loadState(){
  const s = await (await fetch("/api/state")).json();
  state.day = s.day; state.stats = s.stats;
  for (const p of s.population) state.pop.set(p.name, {...p, acc:0, lastProduced:"·"});
  renderCards(); drawCurves(); updateHeader();
}

// ---------- 推演 ----------
async function stepDay(){
  const r = await (await fetch("/api/step", {method:"POST"})).json();
  state.day = r.day;
  replayEvents(r.events, r.stats);
}

function replayEvents(events, stats){
  state.stats = stats;
  for (const e of events){
    if (e.type === "org_day"){
      const g = state.pop.get(e.name); if(!g) continue;
      g.acc = e.acc; g.lastProduced = e.produced[0] === -1 ? "?" : e.produced[0];
      g.readoutProfile = e.readout_profile;
    } else if (e.type === "death"){
      const g = state.pop.get(e.name); if(g) g.alive = false;
    } else if (e.type === "birth"){
      state.pop.set(e.name, {name:e.name, age:0, born_gen:e.gen, parents:e.parents,
                             alive:true, acc:0, lastProduced:"·"});
    } else if (e.type === "day_end"){
      state.stats = e.stats;
      state.accSeries.push(e.stats.avg_acc);
      state.bestSeries.push(e.stats.best_acc);
      state.aliveSeries.push(e.stats.alive);
    }
  }
  renderCards(); drawCurves(); updateHeader();
}

// ---------- 控制 ----------
let lastStep = 0;
async function autoLoop(){
  if (!state.auto) return;
  const now = performance.now();
  if (now - lastStep >= state.speed){ lastStep = now; await stepDay(); }
  requestAnimationFrame(autoLoop);
}
$("#playBtn").addEventListener("click", () => {
  state.auto = !state.auto;
  $("#playBtn").textContent = state.auto ? "⏸ 自动" : "▶ 手动";
  if (state.auto) requestAnimationFrame(autoLoop);
});
$("#stepBtn").addEventListener("click", () => { state.auto = false;
  $("#playBtn").textContent = "▶ 手动"; stepDay(); });
$("#speed").addEventListener("input", e => state.speed = +e.target.value);

// ---------- 骨架预留（Task 7 实现）----------
function showAnatomy(name){ /* Task 7 */ }

(async function init(){
  $("#dayLabel").textContent = "加载中…";
  await loadState();
  requestAnimationFrame(autoLoop);
})();
</script>
</body>
</html>
```

- [ ] **Step 2: JS 语法自检**

```bash
# 提取 <script> 内容做语法检查
python -c "import re;s=open('eco_game.html',encoding='utf-8').read();m=re.findall(r'<script>(.*?)</script>',s,re.S);open('_eco_js_check.js','w',encoding='utf-8').write(m[0])"
node --check _eco_js_check.js
rm _eco_js_check.js
```
Expected: `node --check` 无输出（语法 OK）。

- [ ] **Step 3: 起服务人工确认**

```bash
python eco_server.py --port 8765
```
Expected: 浏览器打开 `http://127.0.0.1:8765` 能看到培养皿网格（40 张卡）、顶部统计、"第 0 天"；点"推演一天"网格吃对率与曲线更新；"自动"模式按设定速度逐天推演。若 grid 为空，检查 console 报错并修复。

- [ ] **Step 4: 提交**

```bash
git add eco_game.html
git commit -m "feat(eco): 游戏前端骨架——培养皿网格、推演控制、统计曲线"
git push origin main
```

---

### Task 7: eco_game.html —— 喂食动画、生死动画、手动喂食、个体解剖

**Files:**
- Modify: `eco_game.html`

**Interfaces:**
- Consumes: `org_day.produced[]`（与本日 food_labels 比对出 ✓/✗）、`death`/`birth` 事件、`/api/manual_feed`、`org_day.readout_profile`
- Produces: 完整游戏体验

- [ ] **Step 1: 实现喂食动画（点击个体逐条重放当天 ✓/✗）**

在 `showAnatomy` 与新增 `playFeedTrace(name)` 中：用 `state.foodLabels`（在 replayEvents 的 `day_begin` 里记录）+ 该个体 `org_day.produced[]`，逐个快速渲染"吃进 X → 吐出 Y → ✓/✗"徽标。给 `replayEvents` 增加 `day_begin` 分支：

```js
} else if (e.type === "day_begin"){
  state.foodLabels = e.food_labels;
  state.day = e.day;
}
```

解剖视图（`showAnatomy(name)`）内容：名字、父母、出生代、年龄、今日吃对率、产出神经元脉冲柱状图（`readoutProfile`，10 根柱子，Canvas 或 div 条）、"逐条回放今日喂食"按钮。

```js
function showAnatomy(name){
  const g = state.pop.get(name); if (!g) return;
  const box = $("#anatomy");
  box.innerHTML = `<b>${esc(name)}</b>
    <div class="mm">父母：${g.parents ? esc(g.parents.join(" × ")) : "（初始个体）"}
    · 出生第 ${g.born_gen} 天 · 已活 ${g.age} 天 · 今日吃对 ${(g.acc*100).toFixed(0)}%</div>
    <div style="display:flex;gap:2px;height:70px;align-items:flex-end;margin:8px 0">${
      (g.readoutProfile||[]).map(v =>
        `<div style="flex:1;background:${v>0?'#58a6ff':'#30363d'};height:${Math.min(100,v*10)}%"
              title="产出神经元${(g.readoutProfile||[]).indexOf(v)} 脉冲${v}"></div>`).join("")}
    </div>
    <button class="btn" onclick="playFeedTrace('${name}')">🎬 回放今日喂食</button>
    <div id="trace"></div>`;
}
```

`playFeedTrace`：用一个 300ms 定时器逐条播放 `foodLabels[i] → produced[i]`，带 ✓/✗ 着色，写入 `#trace`。

- [ ] **Step 2: 实现生死动画**

`replayEvents` 中 `death` 分支在卡片加 `.dead` 类（CSS 已写好淡出/灰度/缩放），并加短暂高亮；`birth` 分支新卡以 `@keyframes popin` 放大出现。在 `<style>` 增加：

```css
@keyframes popin { from { transform:scale(.6); opacity:0; } to { transform:scale(1); opacity:1; } }
.card.birth { animation:popin .5s ease; border-color:var(--ok); }
.card .big { transition:color .2s; }
.card.right .big { color:var(--ok); } .card.wrong .big { color:var(--bad); }
```

`org_day` 分支：按 `g.lastProduced === 对应 food label` 给卡片加 `right`/`wrong` 一闪（0.4s 后移除）。

- [ ] **Step 3: 实现手动喂食**

```js
function drawDigit(cv, pixels){
  const s = 8, ctx = cv.getContext("2d");
  cv.width = 28*s; cv.height = 28*s; ctx.clearRect(0,0,cv.width,cv.height);
  for (let y=0;y<28;y++) for (let x=0;x<28;x++){
    const v = pixels[y*28+x]; ctx.fillStyle = `rgb(${Math.round(v*255)},${Math.round(v*255)},${Math.round(v*255)})`;
    ctx.fillRect(x*s, y*s, s, s);
  }
}
async function manualFeed(digit){
  const r = await (await fetch("/api/manual_feed", {method:"POST",
    headers:{"Content-Type":"application/json"}, body:JSON.stringify({digit})})).json();
  const box = $("#manualResult");
  const label = r.label, prod = r.produced, ok = r.correct;
  box.innerHTML = `<canvas></canvas>
    <div class="mm" style="font-size:16px">喂进 <b>${label}</b> → 吐出 <b>${prod === -1 ? "（没吐出）" : prod}</b>
    <span style="color:${ok ? "var(--ok)" : "var(--bad)"}">${ok ? "✓" : "✗"}</span></div>`;
  drawDigit(box.querySelector("canvas"), r.food_pixels);
}
// init 里生成 0-9 按钮
for (let d=0; d<10; d++){
  const b = el("button", "feedbtn", String(d));
  b.addEventListener("click", () => manualFeed(d));
  $("#manualBtns").appendChild(b);
}
```

- [ ] **Step 4: JS 语法自检 + 浏览器人工验收**

Step 2 的 node --check 命令重跑一遍；然后 `python eco_server.py --port 8765`，人工验收：喂食动画、生死动画、手动喂食、点击个体解剖、曲线随推演增长。修复发现的问题。

- [ ] **Step 5: 提交**

```bash
git add eco_game.html
git commit -m "feat(eco): 游戏动画——喂食回放、生死动画、手动喂食、个体解剖"
git push origin main
```

---

### Task 8: README 更新 + 端到端验收 + 诚实结果报告

**Files:**
- Modify: `README.md`（新增生态游戏章节）
- Create: 无

- [ ] **Step 1: README 增加生态游戏说明**

在 README 追加一节：

```markdown
## LIF 生态游戏（喂食-产出-淘汰-有性繁殖）

游戏式神经进化：生物体（LIF 网络 784→100→10）吃数字、产出数字，吃对率高的存活并做有性繁殖（权重逐权重交叉 + 扰动 + 突变），吃对率低的淘汰。一生不做任何学习，权重出生即随机。

运行：
    python eco_server.py --port 8765
浏览器打开 http://127.0.0.1:8765 即可观看培养皿动画、手动喂食、个体解剖与统计曲线。

与 STDP 进化系统（evolve.py）的区别：本游戏是纯权重遗传进化（无 STDP/无学习），
STDP 系统是架构进化 + 一生 STDP 自学。两者独立共存。
```

- [ ] **Step 2: 端到端验收**

```bash
python eco_tests.py                 # 全量测试
python _eco_smoke.py                # 引擎冒烟（观察单日耗时与最优吃对率）
python eco_server.py --port 8765 &  # 起服务
# 浏览器人工验收游戏界面
```
Expected: 测试全过；冒烟给出诚实基线；界面动画完整可玩。

- [ ] **Step 3: 提交**

```bash
git add README.md
git commit -m "docs(eco): README 补充 LIF 生态游戏运行说明"
git push origin main
```

- [ ] **Step 4: 交付报告**

给用户的最终报告需包含：
1. 玩法说明（喂食→产出→判定→淘汰→有性繁殖）与操作方式
2. 引擎/服务/前端三个文件的职责与事件流
3. **诚实结果**：冒烟跑 20 天的最优吃对率 vs 随机基线 0.10；种群动力学（avg 曲线、存活、寿命淘汰是否生效）
4. 结论：生态机制是否按设计工作、权重遗传进化能否带来吃对率提升、与 STDP 系统的对比
5. 后续方向（多层、玩家干预、家谱可视化等）
