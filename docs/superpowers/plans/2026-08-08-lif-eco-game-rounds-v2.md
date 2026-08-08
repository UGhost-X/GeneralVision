# LIF 生态游戏 v2 回合制重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已交付的 v1 生态游戏（天制/固定 60/按吃对率加权繁殖）重构为**回合制**生态游戏：一回合喂 1 个数字、产出错误的个体当回合死亡（非自然）、活到存活回合数上限自然死亡、存活者随机两两配对繁殖（存活时长加权交叉）、密度依赖承载力封顶、累计自然死亡率 ≥95% 停止、全灭重播。

**Architecture:** 改造现有三文件（`eco_engine.py`/`eco_server.py`/`eco_game.html`），不新建文件结构。引擎回合循环产出事件流；服务端透传配置与事件；前端改为 canvas 培养皿（只显示存活者）+ 回合控制。v1 的 STDP 进化系统（`evolve.py`）不受影响。

**Tech Stack:** Python 3.10（纯 numpy）、stdlib http.server、单文件 HTML/CSS/JS。

## Global Constraints

- Python 3.10.20；引擎只允许 numpy + stdlib，不 import torch/snn.py。
- 只改 `eco_engine.py`、`eco_server.py`、`eco_game.html`、`eco_tests.py`、`_eco_smoke.py`；不动其他文件（尤其并行会话的 `export_snn_demo.py`、`snn_weights.js`、`snn_demo.html`、`docs/superpowers/plans/2026-08-08-snn-infer-demo.md`、`.claude/`）。
- 前端为单文件、无外部 CDN/库；中文 UI。
- 每次 commit 用显式 `git add <路径>`（**绝不** `git add -A`）并 push origin main。
- 确定性：同 seed 全程可复现；rng 全部来自 self.rng 或 (round, index) 确定性派生，不用 `hash()`。
- 诚实呈现：随机权重下"喂 1 个错者死"→ 每回合约 90% 非自然死亡，95% 自然死亡停止条件基本不可达；如实报告实际 natural_rate。

---

### Task 1: 引擎回合制核心 —— step_round、自然/非自然死亡、存活加权交叉、密度承载力、停止计数、全灭重播

**Files:**
- Modify: `eco_engine.py`（常量、crossover、Ecosystem 重构）
- Test: `eco_tests.py`

**Interfaces:**
- Consumes: 现有 `Genome`、`_normalize_cols`、`_random_weights`、`random_genome`、`forward`、`load_mnist`
- Produces:
  - 常量：`T=120, SURVIVAL_ROUNDS=20, N_REPRO=50, CAPACITY=1000, INIT_POP=60`（删除 `POP_CAP/FOOD_COUNT/BOTTOM_DEATH/MAX_AGE`）
  - `crossover(a, b, rng, weight_a=1.0, weight_b=1.0) -> Genome`（逐权重以 `pa=weight_a/(weight_a+weight_b)` 取 a + 高斯扰动 + 突变 + 列归一化；默认权重相等即 50/50）
  - `death_cause(g: Genome, correct: bool, survival_rounds: int) -> str | None`（g.age+=1；错→"unnatural"；age>survival_rounds→"natural"；否则 None）
  - `class Ecosystem`：`set_config(**kw)`、`step_round() -> (events, stats)`、`get_state()`、`get_digit_image`、`manual_feed`、`_round_fingerprint`
  - 事件 schema：`round_begin{type,round,food_idx,food_label}`、`org_round{type,name,produced,correct,age}`、`death{type,name,cause}`、`birth{type,name,parents:[2],gen}`、`reseed{type,count}`、`round_end{type,stats}`
  - `stats = {round, alive, avg_acc, natural_deaths, total_deaths, natural_rate, stopped}`

- [ ] **Step 1: 写失败测试（追加/改写 eco_tests.py）**

```python
# ---- 替换 test_day_loop_invariants 为回合制版本，并新增以下测试 ----

def test_round_loop_invariants():
    eco_ = eco.Ecosystem(seed=0)
    assert len(eco_.pop) == eco.INIT_POP
    for _round in range(4):
        events, stats = eco_.step_round()
        types = [e["type"] for e in events]
        assert types[0] == "round_begin" and types[-1] == "round_end"
        assert len(eco_.pop) <= eco_.capacity, f"round{_round} 种群 {len(eco_.pop)} 超承载力"
        assert 0.0 <= stats["avg_acc"] <= 1.0
        assert 0.0 <= stats["natural_rate"] <= 1.0
        names = {g.name for g in eco_.pop}
        assert len(names) == len(eco_.pop), "重名"
        for e in events:
            if e["type"] == "org_round":
                assert "produced" in e and "correct" in e and "age" in e
            if e["type"] == "death":
                assert e["cause"] in ("natural", "unnatural")
            if e["type"] == "birth":
                assert len(e["parents"]) == 2
    # 手动喂食
    g0 = eco_.pop[0]
    r = eco_.manual_feed(g0.name, 3)
    assert r["label"] == 3 and r["produced"] in list(range(-1, 10))
    assert len(r["food_pixels"]) == 784 and len(r["readout_counts"]) == 10
    # 可复现
    def _run2(seed):
        e = eco.Ecosystem(seed=seed)
        out = []
        for _ in range(2):
            e.step_round()
            out.append(e._round_fingerprint())
        return out
    assert _run2(9) == _run2(9), "同 seed 应可复现"

def test_round_never_exceeds_capacity():
    """多回合后种群应 ≤ 承载力（密度依赖 + 硬上限）。"""
    e = eco.Ecosystem(seed=1)
    for _ in range(8):
        e.step_round()
        assert len(e.pop) <= e.capacity, f"种群 {len(e.pop)} > 承载力 {e.capacity}"
    assert e.total_deaths > 0, "应有死亡记录"
    assert len(e.pop) >= 1, "不应灭绝到 0（全灭重播应恢复）"

def test_weighted_crossover():
    """存活加权交叉：weight_a 越大，后代越接近 a。"""
    rng = np.random.default_rng(7)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child_50 = eco.crossover(a, b, np.random.default_rng(8), weight_a=1, weight_b=1)
    close50 = (np.abs(child_50.hidden - a.hidden) < np.abs(child_50.hidden - b.hidden)).mean()
    assert 0.35 < close50 < 0.65, close50
    child_90 = eco.crossover(a, b, np.random.default_rng(8), weight_a=9, weight_b=1)
    close90 = (np.abs(child_90.hidden - a.hidden) < np.abs(child_90.hidden - b.hidden)).mean()
    assert close90 > 0.75, f"weight_a=9 应显著偏向 a: {close90}"

def test_death_cause():
    """死亡分类：错→unnatural；对但超龄→natural；对且未超龄→存活。"""
    g = eco.random_genome("d", np.random.default_rng(0))
    assert eco.death_cause(g, False, 20) == "unnatural"
    g2 = eco.random_genome("d2", np.random.default_rng(1)); g2.age = 20
    assert eco.death_cause(g2, True, 20) == "natural"
    g3 = eco.random_genome("d3", np.random.default_rng(2)); g3.age = 1
    assert eco.death_cause(g3, True, 20) is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python eco_tests.py`
Expected: `AttributeError: module 'eco_engine' has no attribute 'step_round'`（旧 `test_day_loop_invariants` 也失败，因接口改动——本任务一并改写它，见 Step 1）

- [ ] **Step 3: 写实现（改写 eco_engine.py）**

常量块替换为：

```python
# ---- 游戏参数（v2 回合制） ----
T = 120                  # 仿真步数（泊松编码，Task 2/4 v1 已校准）
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
CROSS_SIGMA = 0.01       # 有性繁殖高斯扰动 σ
MUT_RATE = 0.001         # 千分之一大突变
SURVIVAL_ROUNDS = 20     # 自然寿命上限（回合）
N_REPRO = 50             # 每对每次繁殖数量 = 存活回合数 × N
CAPACITY = 1000          # 环境承载力（种群上限）
INIT_POP = 60            # 初始/全灭重播种群数
```

`crossover` 增加存活加权（默认权重相等即 50/50，保持向后兼容）：

```python
def crossover(a: Genome, b: Genome, rng: np.random.Generator,
              weight_a: float = 1.0, weight_b: float = 1.0) -> Genome:
    """有性繁殖：逐权重以 pa 取 a（存活时长加权）+ 高斯扰动 + 千分之一大突变。

    默认 weight_a=weight_b → 50/50；weight_a 越大后代越偏 a（活越久的个体基因占比越大）。
    """
    wsum = weight_a + weight_b
    pa = weight_a / wsum if wsum > 0 else 0.5
    hidden = np.where(rng.random(a.hidden.shape) < pa, a.hidden, b.hidden)
    hidden += rng.normal(0.0, CROSS_SIGMA, hidden.shape)
    mh = rng.random(hidden.shape) < MUT_RATE
    hidden[mh] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(mh.sum()))
    hidden = _normalize_cols(hidden, W_NORM_HIDDEN)

    readout = np.where(rng.random(a.readout.shape) < pa, a.readout, b.readout)
    readout += rng.normal(0.0, CROSS_SIGMA, readout.shape)
    mr = rng.random(readout.shape) < MUT_RATE
    readout[mr] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(mr.sum()))
    readout = _normalize_cols(readout, W_NORM_READOUT)

    return Genome(name="child", hidden=hidden, readout=readout,
                  parents=(a.name, b.name))


def death_cause(g: Genome, correct: bool, survival_rounds: int) -> str | None:
    """返回本回合的死亡原因：'natural' | 'unnatural' | None（存活）。"""
    g.age += 1
    if not correct:
        return "unnatural"
    if g.age > survival_rounds:
        return "natural"
    return None
```

`Ecosystem` 重构（`__init__`、`set_config`、`step_round`、`get_state`、`manual_feed` 等）：

```python
class Ecosystem:
    """回合制生态主循环。同 seed 全程可复现。"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.counter = 0
        self.round = 0
        # 可调参数（前端经 /api/config 修改）
        self.survival_rounds = SURVIVAL_ROUNDS
        self.n_repro = N_REPRO
        self.capacity = CAPACITY
        self.initial_pop = INIT_POP
        # 停止条件计数
        self.natural_deaths = 0
        self.total_deaths = 0
        self.stopped = False
        self._img, self._lbl, _, _ = load_mnist()
        self.pop: list[Genome] = [
            random_genome(f"eco#{i}", self.rng, gen=0) for i in range(self.initial_pop)
        ]
        self.counter = self.initial_pop

    def set_config(self, **kw) -> dict:
        """更新可调参数（前端调用）。合法值直接写入；非法忽略。"""
        if "survival_rounds" in kw and 10 <= kw["survival_rounds"] <= 30:
            self.survival_rounds = int(kw["survival_rounds"])
        if "n_repro" in kw and 10 <= kw["n_repro"] <= 100:
            self.n_repro = int(kw["n_repro"])
        if "capacity" in kw and 100 <= kw["capacity"] <= 5000:
            self.capacity = int(kw["capacity"])
        if "initial_pop" in kw and 60 <= kw["initial_pop"] <= 1000:
            self.initial_pop = int(kw["initial_pop"])
        return self._config()

    def _config(self) -> dict:
        return {"survival_rounds": self.survival_rounds, "n_repro": self.n_repro,
                "capacity": self.capacity, "initial_pop": self.initial_pop}

    def _next_name(self) -> str:
        n = f"eco#{self.counter}"
        self.counter += 1
        return n

    def _round_fingerprint(self) -> tuple:
        return (round(float(self.natural_deaths) / max(1, self.total_deaths), 6),
                len(self.pop), tuple(sorted(g.name for g in self.pop)))

    def step_round(self) -> tuple[list[dict], dict]:
        events: list[dict] = []
        food_idx = int(self.rng.integers(0, len(self._img)))
        food_lbl = int(self._lbl[food_idx])
        events.append({"type": "round_begin", "round": self.round,
                       "food_idx": food_idx, "food_label": food_lbl})

        survivors: list[Genome] = []
        avg_correct = 0.0
        for i, g in enumerate(self.pop):
            produced, _hc, rc = forward(g, self._img[food_idx][None],
                                        np.random.default_rng(self.round * 1_000_003 + i))
            produced = int(produced[0])
            correct = (produced == food_lbl)
            cause = death_cause(g, correct, self.survival_rounds)
            events.append({"type": "org_round", "name": g.name, "produced": produced,
                           "correct": bool(correct), "age": g.age,
                           "readout_profile": rc.mean(axis=0).round(2).tolist()})
            avg_correct += float(correct)
            if cause is not None:
                self.total_deaths += 1
                if cause == "natural":
                    self.natural_deaths += 1
                events.append({"type": "death", "name": g.name, "cause": cause})
            else:
                survivors.append(g)
        avg_correct /= max(1, len(self.pop))

        # ---- 随机两两配对繁殖（存活时长加权交叉 + 密度依赖 + 承载力） ----
        self.rng.shuffle(survivors)
        pairs = [(survivors[j], survivors[j + 1]) for j in range(0, len(survivors) - 1, 2)]
        births: list[Genome] = []
        if pairs:
            density = max(0.0, 1.0 - len(self.pop) / self.capacity)
            brood = int(round(self.survival_rounds * self.n_repro * density))
            room = max(0, self.capacity - len(survivors))
            target = min(room, len(pairs) * brood)
            for k in range(target):
                a, b = pairs[k % len(pairs)]
                child = crossover(a, b, self.rng, weight_a=float(a.age), weight_b=float(b.age))
                child.name = self._next_name()
                child.born_gen = self.round
                births.append(child)
        survivors.extend(births)
        self.pop = survivors
        for c in births:
            events.append({"type": "birth", "name": c.name,
                           "parents": [c.parents[0], c.parents[1]], "gen": c.born_gen})

        # ---- 全灭重播 ----
        if not self.pop:
            self.pop = [random_genome(f"eco#{self.counter + i}", self.rng, gen=self.round)
                        for i in range(self.initial_pop)]
            self.counter += self.initial_pop
            events.append({"type": "reseed", "count": self.initial_pop})

        # ---- 停止条件：累计自然死亡 / 累计总死亡 ≥ 95% ----
        natural_rate = (self.natural_deaths / self.total_deaths) if self.total_deaths > 0 else 0.0
        self.stopped = self.stopped or (natural_rate >= 0.95)
        stats = {"round": self.round, "alive": len(self.pop),
                 "avg_acc": round(avg_correct, 4),
                 "natural_deaths": self.natural_deaths,
                 "total_deaths": self.total_deaths,
                 "natural_rate": round(natural_rate, 4),
                 "stopped": self.stopped}
        events.append({"type": "round_end", "stats": stats})
        self.round += 1
        return events, stats

    def get_state(self) -> dict:
        return {"round": self.round, "config": self._config(),
                "population": [{"name": g.name, "age": g.age, "born_gen": g.born_gen,
                                "parents": list(g.parents) if g.parents else None,
                                "alive": True} for g in self.pop],
                "stats": self._last_stats()}

    def _last_stats(self) -> dict:
        natural_rate = (self.natural_deaths / self.total_deaths) if self.total_deaths > 0 else 0.0
        return {"round": self.round, "alive": len(self.pop),
                "natural_deaths": self.natural_deaths,
                "total_deaths": self.total_deaths,
                "natural_rate": round(natural_rate, 4),
                "stopped": self.stopped}
```

（`get_digit_image`、`manual_feed` 保持不变；`_last_stats` 改为方法，`get_state()` 内以 `self._last_stats()` 调用。）

- [ ] **Step 4: 运行测试确认通过**

Run: `python eco_tests.py`
Expected: 全部通过（旧 `test_day_loop_invariants` 已被 `test_round_loop_invariants` 替代；`test_server_endpoints` 尚引用旧 config 键会失败——它在 Task 3 更新，本任务先删除或临时跳过该测试中的 config 断言，见备注）。

> 备注：`test_server_endpoints` 断言 `config.pop_cap` 与 `day_begin` 事件，本任务后这些键已变。Task 1 先把该测试的这两个断言改为 `config.survival_rounds` 存在 与 `round_begin`，使其通过；其余端点断言保留。

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): 回合制核心——step_round、自然/非自然死亡、存活加权交叉、密度承载力、停止计数"
git push origin main
```

---

### Task 2: 引擎冒烟 + 诚实基线

**Files:**
- Modify: `_eco_smoke.py`

**Interfaces:**
- Consumes: `Ecosystem.step_round`

- [ ] **Step 1: 改写冒烟脚本**

```python
# _eco_smoke.py
"""回合制引擎冒烟：跑 30 回合，记录每回合种群/正确率/自然死亡率/耗时与诚实基线。"""
import time
import numpy as np
import eco_engine as eco

def main():
    e = eco.Ecosystem(seed=0)
    t0 = time.time()
    prev_best = 0.0
    for r in range(30):
        t = time.time()
        events, stats = e.step_round()
        dt = time.time() - t
        prev_best = max(prev_best, stats["avg_acc"])
        print(f"round {stats['round']:>3} | alive={stats['alive']:>5} "
              f"correct={stats['avg_acc']:.3f} natural_rate={stats['natural_rate']:.4f} "
              f"n_deaths={stats['total_deaths']:>5} | {dt:.2f}s")
    print(f"\n=== 冒烟完成 === 最优单回合正确率={prev_best:.3f} "
          f"(随机基线≈0.10) 总耗时 {time.time()-t0:.0f}s "
          f"最终 natural_rate={stats['natural_rate']:.4f}（目标 0.95）")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行冒烟**

Run: `python _eco_smoke.py`
Expected: 30 回合跑完无崩溃；观察并如实记录：
- **单回合耗时 vs 种群**：种群应从小增长到承载力附近（~1000），单回合耗时随之增长（每只 ~2-3ms，1000 只 ~2-3s）。若 >10s 或种群增长异常，检查密度公式；若远超可玩范围，报告并建议调 `CAPACITY` 默认值。
- **correct 率**：应在 0.08-0.12 附近（随机基线），如实记录。
- **natural_rate**：应接近 0（随机权重下几乎无人活到存活回合数上限）——**如实呈现"停止条件基本不可达"**。
- **`test_round_never_exceeds_capacity` 是否成立**：种群 ≤ 承载力。

- [ ] **Step 3: 提交**

```bash
git add _eco_smoke.py
git commit -m "chore(eco): 回合制引擎冒烟——量测耗时与诚实基线"
git push origin main
```

---

### Task 3: 服务端 —— /api/config、事件与配置透传

**Files:**
- Modify: `eco_server.py`、`eco_tests.py`

**Interfaces:**
- Consumes: `Ecosystem.set_config/get_state/step_round/get_digit_image/manual_feed`
- Produces:
  - `POST /api/config` body `{survival_rounds?, n_repro?, capacity?, initial_pop?}` → `{"config": {...}}`（非法值忽略）
  - `GET /api/state` → `{round, config:{survival_rounds,n_repro,capacity,initial_pop}, population, stats:{round,alive,natural_rate,...}}`
  - `POST /api/step` → `{round, events, stats}`（事件类型：round_begin/org_round/death(cause)/birth/reseed/round_end）

- [ ] **Step 1: 更新 `test_server_endpoints`**（eco_tests.py 中 config 与事件断言）

```python
def test_server_endpoints():
    import json, urllib.request
    from eco_server import run_server_in_thread
    port, server = run_server_in_thread(seed=0)
    base = f"http://127.0.0.1:{port}"
    try:
        s = json.load(urllib.request.urlopen(base + "/api/state"))
        assert s["config"]["survival_rounds"] == eco.SURVIVAL_ROUNDS
        assert len(s["population"]) == eco.INIT_POP
        req = urllib.request.Request(base + "/api/step", method="POST")
        r = json.load(urllib.request.urlopen(req))
        assert r["events"][0]["type"] == "round_begin"
        assert r["stats"]["natural_rate"] >= 0.0
        img = json.load(urllib.request.urlopen(base + "/api/digit_image/0"))
        assert len(img["pixels"]) == 784
        body = json.dumps({"digit": 4, "name": s["population"][0]["name"]}).encode()
        rq = urllib.request.Request(base + "/api/manual_feed", data=body, method="POST",
                                    headers={"Content-Type": "application/json"})
        mf = json.load(urllib.request.urlopen(rq))
        assert mf["label"] == 4 and len(mf["readout_counts"]) == 10
        cfg = json.dumps({"n_repro": 60}).encode()
        cr = urllib.request.Request(base + "/api/config", data=cfg, method="POST",
                                    headers={"Content-Type": "application/json"})
        c2 = json.load(urllib.request.urlopen(cr))
        assert c2["config"]["n_repro"] == 60
        html = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert 'id="dish"' in html and "<canvas" in html
    finally:
        server.shutdown()
        server.server_close()
```

- [ ] **Step 2: 运行确认失败**

Run: `python eco_tests.py`
Expected: `test_server_endpoints` 失败（`/api/config` 404）

- [ ] **Step 3: 写实现（eco_server.py 增加 do_POST /api/config 分支，并把 state/step 的返回键对齐）**

在 `do_POST` 中加入：

```python
        elif self.path == "/api/config":
            body = self._read_body() or {}
            with self.lock:
                cfg = self.engine.set_config(**body)
            self._json({"config": cfg})
```

同时确认 `GET /api/state` 与 `POST /api/step` 直接返回 `engine.get_state()` / `step_round()` 的结果即可（键已由引擎提供）。

- [ ] **Step 4: 运行测试确认通过**

Run: `python eco_tests.py`
Expected: 全部通过

- [ ] **Step 5: 提交**

```bash
git add eco_server.py eco_tests.py
git commit -m "feat(eco): 服务端 /api/config 与回合制事件透传"
git push origin main
```

---

### Task 4: 前端回合制重构 —— canvas 培养皿、只显示存活、新控制、统计与停止指示

**Files:**
- Rewrite: `eco_game.html`

**Interfaces:**
- Consumes: `/api/state`、`/api/step`、`/api/digit_image/<idx>`、`/api/manual_feed`、`/api/config`
- Produces: 单文件前端（canvas 培养皿 + 回合控制 + 统计曲线 + 手动喂食 + 解剖）

- [ ] **Step 1: 结构重写**

重写 `eco_game.html`（保留 v1 暗色风格与总体布局），实现：

**Header**：`第 N 回合`、⏸ 自动 / ⏭ 推演一回合、速度滑条（1-10 秒/回合，标签"投喂速度"）、停止指示条（`natural_rate` 进度条，目标 95%，≥95% 时提示"达成 95% 自然死亡目标，已停止"）。

**控制面板（可调参数，调 `POST /api/config`）**：
- 存活回合数：`<input type="range" min="10" max="30">` 默认 20
- 繁殖倍数 N：`<input type="range" min="10" max="100">` 默认 50
- 承载力：`<input type="range" min="100" max="5000" step="100">` 默认 1000
- 初始种群：`<input type="range" min="60" max="1000" step="10">` 默认 60
- 改动即发 `/api/config`，并在 `#statsLabel` 提示"参数已更新"。

**Canvas 培养皿 `#dish`**：按存活者数量自适应网格（`ceil(sqrt(alive))` 列），每只一个色块：
- 本回合产出正确 → 绿色；错误 → 红色；等待喂食 → 灰蓝。
- 色块尺寸随种群自适应（种群大则格子小，最小 4px）。
- 死亡：色块当回合从 canvas 上**移除**（不留灰块）——canvas 每回合按存活者重绘。
- 点击色块 → 解剖视图（见 Step 2）。

**统计曲线**（canvas，蓝=单回合正确率、绿=种群数（右轴或按承载力归一）、紫=natural_rate）。

**手动喂食**：0-9 按钮 → `POST /api/manual_feed {digit, name: 当前点击个体或随机存活者}` → 显示喂进/吐出/✓✗ + 绘制数字。

**核心 JS 结构**：
```js
const state = { round:0, pop:new Map(), config:null, stats:null, auto:true,
                speed:5000, foodLabel:null, naturalSeries:[], correctSeries:[],
                popSeries:[], stepping:false, selected:null };
async function loadState(){ /* fetch /api/state; 填 config 滑块、pop map、round、stats */ }
async function stepRound(){ /* POST /api/step; replayEvents(r.events, r.stats) */ }
function replayEvents(events, stats){ /* round_begin→foodLabel; org_round→更新个体对错/年龄;
    death→从 pop 移除并标记动画; birth→加入 pop; reseed→清空重建; round_end→stats/曲线/停止判断 */ }
function drawDish(){ /* canvas 按 state.pop 存活者画色块网格 */ }
function drawCurves(){ /* 正确率/种群/natural_rate 三线 */ }
// 自动：setTimeout 按 speed(ms) 推演一回合（速度 1-10s 映射 1000-10000ms）
```

- [ ] **Step 2: 交互与动画**

- 死亡动画：死亡个体色块短暂变暗（1 帧）后从 canvas 移除（符合"去掉灰块"）。
- 喂食动画：每回合 `round_begin` 显示食物数字；`org_round` 逐个体在 canvas 上闪烁对/错色；整个回合动画时长受速度滑条控制（1-10s）。
- 解剖视图：点击色块 → 弹出面板显示该个体 `readout_profile` 柱状图（10 根）、父母、出生回合、当前存活回合数、本回合产出/对错。
- 停止条件：`round_end.stats.stopped === true` → 暂停自动推演并提示。

- [ ] **Step 3: 语法自检 + 人工确认**

```bash
python -c "import re;s=open('eco_game.html',encoding='utf-8').read();m=re.findall(r'<script>(.*?)</script>',s,re.S);open('_eco_js_check.js','w',encoding='utf-8').write(m[0])"
node --check _eco_js_check.js && rm _eco_js_check.js
python eco_tests.py
python eco_server.py --port 8765   # 浏览器人工验收（画面/控制/动画/手动喂食/解剖/停止条）
```

- [ ] **Step 4: 提交**

```bash
git add eco_game.html
git commit -m "feat(eco): 前端回合制重构——canvas 培养皿只显示存活、回合控制、停止指示"
git push origin main
```

---

### Task 5: 端到端验收 + 诚实报告

**Files:**
- Modify: `README.md`（生态游戏章节更新为回合制说明）

- [ ] **Step 1: README 生态游戏章节更新为回合制**

把 README 的生态游戏段落改为：一回合喂 1 个数字、产出错误的当回合死亡（非自然）、活到存活回合数上限自然死亡、存活者随机两两繁殖（存活时长加权交叉）、密度依赖承载力、累计自然死亡 ≥95% 停止。可调参数列表（投喂速度 1-10s、存活回合数 10-30、N 10-100、承载力 100-5000、初始种群 60-1000）。

- [ ] **Step 2: 全量验收**

```bash
python eco_tests.py          # 全量测试
python _eco_smoke.py         # 30 回合冒烟，记录诚实基线
python eco_server.py --port 8765  # 起服务，人工浏览器验收回合制界面
```

- [ ] **Step 3: 提交**

```bash
git add README.md
git commit -m "docs(eco): README 更新为回合制生态游戏说明"
git push origin main
```

- [ ] **Step 4: 交付报告**

给用户报告：玩法（回合制）、可调参数、诚实结果（correct 率 vs 随机基线、natural_rate vs 95% 目标的实际差距、种群增长曲线、单回合耗时）、结论（机制是否按设计、停止条件为何基本不可达）、与 v1/STDP 的对比。
