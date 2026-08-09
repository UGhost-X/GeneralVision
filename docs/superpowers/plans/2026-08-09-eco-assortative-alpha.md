# 选型交配 + 存活奖励 alpha 实施计划

> **Goal:** 在 v2 回合制生态游戏上实现两项繁殖机制：① 选型交配（存活回合数相近者更倾向互相繁殖，年龄相似度加权配对 + 可调强度 s）；② 存活奖励 alpha（`alpha = 1/每回合存活率`，`brood = 存活回合数 × N × alpha × density`）。规格：`docs/superpowers/specs/2026-08-09-eco-assortative-alpha-design.md`。

**Architecture:** 改造 `eco_engine.py`（引擎：纯函数 `survival_alpha` + `assortative_pairs`、`Ecosystem` 参数/配对/公式/stats）、`eco_tests.py`（新增 4 测试）、`eco_game.html`（选型强度滑条 + 头部存活率/α 读数）。不新建文件结构。

**Tech Stack:** Python 3.10（纯 numpy）、stdlib http.server、单文件 HTML/JS。

## Global Constraints

- 引擎只允许 numpy + stdlib；只改 `eco_engine.py`、`eco_tests.py`、`eco_game.html`。
- 确定性：同 seed 全程可复现；新增 rng 调用全部走 `self.rng` 或确定性派生。
- 前端单文件、无 CDN、中文 UI。
- 每次 commit 用显式 `git add <路径>` 并 push origin main。
- 诚实呈现：随机权重下每回合约 10% 存活 → alpha≈10；选型交配受存活者年龄分布窄（多为 1-3 回合）限制，效果有限；如实报告。

---

### Task 1: 引擎 —— survival_alpha、assortative_pairs、assort_strength、stats 扩展

**Files:**
- Modify: `eco_engine.py`
- Test: `eco_tests.py`

**Interfaces:**
- 新增常量：`ASSORT_TAU = 2.0`（年龄相似度核宽，固定）、`ASSORT_STRENGTH = 0.5`（默认选型强度）
- 新增纯函数：
  - `survival_alpha(n_survivors, n_start) -> float`：`sr = n_survivors/n_start`；`sr>0 → 1/sr`；否则 1.0
  - `assortative_pairs(survivors: list[Genome], strength, rng, tau=ASSORT_TAU) -> list[(Genome,Genome)]`：随机序取第一只，第二只按 `w = (1-strength) + strength·exp(-|Δage|/tau)` 加权采样；strength=0 等价均匀随机（= 现状）
- `Ecosystem`：`__init__` 加 `self.assort_strength`；`set_config` 校验 `0<=assort_strength<=1`；`_config` 返回；`step_round` 用 `assortative_pairs` 配对 + alpha 进 brood + stats 加 `survival_rate`/`alpha`；`_last_stats` 返回两者（存 `self.last_survival_rate`/`self.last_alpha`）

- [ ] **Step 1: 写失败测试（追加 eco_tests.py）**

```python
def test_survival_alpha():
    assert eco.survival_alpha(10, 10) == 1.0
    assert abs(eco.survival_alpha(5, 10) - 2.0) < 1e-9
    assert eco.survival_alpha(0, 10) == 1.0     # 0 存活防御
    assert eco.survival_alpha(3, 0) == 1.0      # 空回合防御

def test_assortative_pairing_tends_similar_age():
    rng = np.random.default_rng(42)
    survivors = [eco.random_genome(f"lo{i}", rng) for i in range(20)]
    for g in survivors: g.age = 1
    survivors += [eco.random_genome(f"hi{i}", rng) for i in range(20)]
    for g in survivors[20:]: g.age = 20
    def mean_abs_diff(strength):
        diffs = []
        for s in range(30):
            pairs = eco.assortative_pairs(survivors, strength,
                                          np.random.default_rng(1000 + s))
            diffs += [abs(a.age - b.age) for a, b in pairs]
        return float(np.mean(diffs))
    d_random = mean_abs_diff(0.0)
    d_assort = mean_abs_diff(1.0)
    assert d_assort < d_random * 0.6, f"选型 {d_assort:.2f} 应显著小于随机 {d_random:.2f}"

def test_stats_has_survival_rate_alpha():
    e = eco.Ecosystem(seed=2)
    for _ in range(3):
        _, stats = e.step_round()
        assert "survival_rate" in stats and "alpha" in stats
        if stats["survival_rate"] > 0:
            assert abs(stats["alpha"] - 1.0 / stats["survival_rate"]) < 1e-6
    st = e.get_state()["stats"]
    assert "survival_rate" in st and "alpha" in st

def test_config_assort_strength():
    e = eco.Ecosystem(seed=4)
    e.set_config(assort_strength=0.8)
    assert abs(e.assort_strength - 0.8) < 1e-9
    assert abs(e._config()["assort_strength"] - 0.8) < 1e-9
    e.set_config(assort_strength=-0.1)
    assert abs(e.assort_strength - 0.8) < 1e-9   # 非法值忽略
    e.set_config(assort_strength=1.5)
    assert abs(e.assort_strength - 0.8) < 1e-9
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe eco_tests.py`
Expected: `AttributeError: module 'eco_engine' has no attribute 'survival_alpha'`

- [ ] **Step 3: 写实现（eco_engine.py）**

常量块加：
```python
ASSORT_TAU = 2.0      # 选型交配：年龄相似度核宽（回合），固定不可调
ASSORT_STRENGTH = 0.5 # 默认选型强度 s（0-1）
```

模块级纯函数（`crossover` 之后）：
```python
def survival_alpha(n_survivors: int, n_start: int) -> float:
    """存活奖励 alpha = 1/每回合存活率（种群级）。全员存活→1.0；0 存活→1.0（防御）。"""
    if n_start <= 0:
        return 1.0
    sr = n_survivors / n_start
    return 1.0 / sr if sr > 0 else 1.0


def assortative_pairs(survivors: list[Genome], strength: float,
                      rng: np.random.Generator, tau: float = ASSORT_TAU
                      ) -> list[tuple[Genome, Genome]]:
    """选型交配：年龄相似度加权配对。

    随机序取第一只 a；第二只从剩余存活者中按 w=(1-strength)+strength·exp(-|Δage|/tau)
    加权采样。strength=0 → 权重恒等 → 均匀随机（等价原 shuffle+相邻）；strength=1 → 纯按年龄相似度。
    """
    pool = list(survivors)
    rng.shuffle(pool)
    pairs = []
    while len(pool) >= 2:
        a = pool[0]
        rest = pool[1:]                                # 切片即排除 a
        d = np.abs(np.array([g.age for g in rest], dtype=float) - a.age)
        w = (1.0 - strength) + strength * np.exp(-d / tau)
        b_idx = int(rng.choice(len(rest), p=w / w.sum()))
        b = rest.pop(b_idx)                            # 从 rest 移除 b
        pairs.append((a, b))
        pool = rest
    return pairs
```

`Ecosystem.__init__` 加：
```python
self.assort_strength = ASSORT_STRENGTH
self.last_survival_rate = 0.0
self.last_alpha = 1.0
```

`set_config` 加：
```python
if "assort_strength" in kw and 0.0 <= kw["assort_strength"] <= 1.0:
    self.assort_strength = float(kw["assort_strength"])
```
`_config` 加 `"assort_strength": self.assort_strength`。

`step_round` 喂食/死亡循环后、繁殖段改为：
```python
        # ---- 存活奖励 alpha：存活率 = 本回合存活者 / 回合开始种群 ----
        start_pop = len(self.pop)
        alpha = survival_alpha(len(survivors), start_pop)
        self.last_survival_rate = (len(survivors) / start_pop) if start_pop > 0 else 0.0
        self.last_alpha = alpha

        # ---- 选型交配繁殖（年龄相似度加权配对 + 存活加权交叉 + 密度依赖 + 承载力） ----
        pairs = assortative_pairs(survivors, self.assort_strength, self.rng)
        births: list[Genome] = []
        if pairs:
            density = max(DENSITY_FLOOR, 1.0 - start_pop / self.capacity)
            brood = int(round(self.survival_rounds * self.n_repro * alpha * density))
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
```

stats 加两键：
```python
        stats = {"round": self.round, "alive": len(self.pop),
                 "avg_acc": round(avg_correct, 4),
                 "survival_rate": round(self.last_survival_rate, 4),
                 "alpha": round(self.last_alpha, 3),
                 "natural_deaths": self.natural_deaths,
                 "total_deaths": self.total_deaths,
                 "natural_rate": round(natural_rate, 4),
                 "stopped": self.stopped}
```
`_last_stats` 加 `"survival_rate": self.last_survival_rate, "alpha": self.last_alpha`。

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe eco_tests.py`
Expected: 全部通过（12 旧 + 4 新）

- [ ] **Step 5: 提交**

```bash
git add eco_engine.py eco_tests.py
git commit -m "feat(eco): 选型交配(年龄相似度加权配对+assort_strength) + 存活奖励alpha(1/存活率，brood×alpha)"
git push origin main
```

---

### Task 2: 前端 —— 选型强度滑条 + 存活率/α 读数

**Files:**
- Modify: `eco_game.html`

**Interfaces:**
- Consumes: `/api/config`（`assort_strength`）、`/api/state`、`/api/step` stats（`survival_rate`/`alpha`）

- [ ] **Step 1: 环境参数面板加滑条（繁殖倍数 N 行之后）**

```html
<div class="cfgrow"><label>选型强度 s <span id="v_assort">50</span></label>
  <input type="range" id="c_assort" min="0" max="100" step="5" value="50"></div>
```

- [ ] **Step 2: syncConfigSliders 支持缩放（assort_strength 0-1 ↔ 滑条 0-100）**

```js
const map = { c_survival:["survival_rounds","v_survival"], c_nrepro:["n_repro","v_nrepro"],
              c_capacity:["capacity","v_capacity"], c_initial:["initial_pop","v_initial"],
              c_assort:["assort_strength","v_assort",100] };
for (const id in map){
  const [key, vlabel, scale] = map[id];
  const v = (scale || 1) * c[key];
  document.getElementById(id).value = v;
  document.getElementById(vlabel).textContent = Math.round(v);
}
```

`bindConfig` 加 scale 参数（发送时除以 scale）：
```js
function bindConfig(id, key, vlabel, scale){
  const el = document.getElementById(id);
  el.addEventListener("input", () => {
    document.getElementById(vlabel).textContent = el.value;
    clearTimeout(el._t);
    el._t = setTimeout(() => postConfig({[key]: +el.value / (scale || 1)}), 250);
  });
}
```
初始化处加：`bindConfig("c_assort","assort_strength","v_assort", 100);`

- [ ] **Step 3: 头部统计行加存活率/α（去掉重复的自然死亡%）**

```js
const sr = (s.survival_rate != null) ? (s.survival_rate*100).toFixed(1)+"%" : "—";
const al = (s.alpha != null) ? "α "+s.alpha.toFixed(1) : "—";
$("#statsLabel").innerHTML = `种群 <b>${s.alive}</b> · 正确率 <b>${acc}</b> · 存活率 <b>${sr}</b> · ${al}`;
```

- [ ] **Step 4: JS 语法自检 + 全量测试**

```bash
node --check <(sed -n '/<script>/,/<\/script>/p' eco_game.html | sed '1d;$d')
.venv\Scripts\python.exe eco_tests.py
```

- [ ] **Step 5: 提交**

```bash
git add eco_game.html
git commit -m "feat(eco): 前端选型强度滑条 + 头部存活率/alpha 读数"
git push origin main
```

---

### Task 3: 验收 + 诚实报告

- [ ] **Step 1: 跑冒烟观察 alpha/选型效果**

```bash
.venv\Scripts\python.exe _eco_smoke.py
```
观察：alpha 是否约 10（随机权重）、存活率 ~0.1、种群仍 ≤ 承载力；选型强度默认 0.5 下无崩溃。

- [ ] **Step 2: 起服务人工抽查**

```bash
.venv\Scripts\python.exe eco_server.py --port 8765
```

- [ ] **Step 3: 交付报告**

汇报：alpha/存活率读数、选型配对统计验证结果、诚实预期（alpha≈10、选型受窄年龄分布限制）、测试全绿。
