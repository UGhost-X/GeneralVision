# 在线学习 P1：学习骨架实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 LIF 生态游戏中的个体拥有生命周期内学习能力：基因型(先天)/表型(学习后)分离，读出层在线监督 delta rule 学习，出生成熟期发育，学习参数作为可遗传基因供进化调优。

**Architecture:** `Genome` 保留为基因型（可遗传可突变），新增三个学习基因字段；`Organism` 新增 `learned_weights` 表型数组（出生=基因型副本，死亡释放）。新增 `forward_learn()` 返回最后一层隐藏率供读出层学习；新增 `_phenotype_genomes()` 构造指向表型的轻量 Genome shim，使现有 forward/census 管道零改动地用上表型。`step()` 喂食后对全部存活个体做读出层 delta rule 学习 + 漂移回基因型；新生个体出生后跑成熟期发育。体检评估改用表型。

**Tech Stack:** Python 3.10 (uv/.venv), numpy 2.2, numba 0.66, pytest 8。eco_engine 为纯 numpy+numba，不依赖 torch。

## Global Constraints

- 所有代码修改前必须提交并推送当前改动到 `origin main`（CLAUDE.md 工作流）。
- 提交信息用中文（仓库惯例，参考近期 commit）。
- 保持 eco_engine.py 纯 numpy+numba（无 torch 依赖）。
- `Genome.weights` 永远是**基因型**（先天）；学习只写 `Organism.learned_weights`（表型）。韦斯曼屏障。
- 学习基因值域：`readout_lr ∈ [0,1]`、`hidden_plasticity ∈ [0,1]`（P1 仅存在，P2 生效）、`plasticity_drift ∈ [0,0.5]`。
- 食物流=测试集（评测），成熟期学习流=训练集（诚实边界）。
- 每个任务结束必须跑通对应 pytest，且提交+推送。

---

### Task 1: pytest 脚手架 + 引擎冒烟测试

**Files:**
- Create: `pytest.ini`
- Create: `tests/test_eco_learning.py`（仅冒烟测试 + 共享 eco fixture）
- Modify: `requirements.txt`（加 pytest 开发依赖注释）
- 运行：`uv pip install pytest`

**Interfaces:**
- Consumes: 无（新脚手架）。
- Produces: `pytest.ini`（testpaths=tests）；`tests/test_eco_learning.py` 提供模块级 fixture `eco`（后续任务复用）；pytest 已装入 `.venv`。

- [ ] **Step 1: 安装 pytest 并写冒烟测试**

```bash
uv pip install "pytest>=8.0"
```

创建 `pytest.ini`：
```ini
[pytest]
testpaths = tests
```

在 `requirements.txt` 末尾追加（开发依赖注释）：
```
# 开发：测试
pytest>=8.0
```

创建 `tests/test_eco_learning.py`：
```python
"""LIF 生态游戏在线学习 P1 测试。"""

import numpy as np
import pytest

from eco_engine import EcoConfig, Ecosystem


@pytest.fixture(scope="module")
def eco() -> Ecosystem:
    """共享一个最小生态实例（首次构造含奠基筛选，之后 reset 廉价）。"""
    return Ecosystem(config=EcoConfig(init_pop=100), seed=7)


def test_smoke_step(eco):
    eco.reset()
    events = eco.step()
    assert events["round"] == 1
    assert events["population_after"] > 0
    assert "accuracy" in events
    assert eco.state()["population_size"] > 0
```

- [ ] **Step 2: 运行确认（预期先失败于缺 pytest）**

Run: `python -m pytest tests/test_eco_learning.py -v`
Expected: pytest 已安装则测试通过；若报 numpy 版本或导入错误，先解决环境问题。

- [ ] **Step 3: 提交**

```bash
git add pytest.ini tests/test_eco_learning.py requirements.txt
git commit -m "测试：新增 pytest 脚手架与引擎冒烟测试

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

### Task 2: 学习基因 + 基因型/表型字段

**Files:**
- Modify: `eco_engine.py` — 常量区（新增学习基因值域与尺度）；`Genome`（新增 3 字段 + to_dict）；`random_genome`（初始化学习基因）；`_mutate_genome`（学习基因变异 + 两处 Genome 构造补字段）；`crossover`（学习基因平均 + 构造补字段）；`Organism`（新增 learned_weights/acc_ema/samples_learned + `_learning_amount` + to_dict）；`_kill`（释放表型）；`_seed_population`（表型初始化）

**Interfaces:**
- Consumes: `Organism`（Task 2 定义），`Genome` 新字段。
- Produces: `Genome.readout_lr / hidden_plasticity / plasticity_drift`；`Organism.learned_weights / acc_ema / samples_learned`；`Organism._learning_amount() -> float`；常量 `READOUT_LR_SCALE=0.05`、`MATURITY_SAMPLES=6`、值域常量。`random_genome` 初始 `readout_lr≈0.2`、`hidden_plasticity≈0.05`、`plasticity_drift≈0.0`。

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_eco_learning.py`）

```python
from eco_engine import (
    EcoConfig,
    Ecosystem,
    crossover,
    random_genome,
)


def test_genome_has_learning_genes():
    g = random_genome(np.random.default_rng(0))
    assert 0.0 <= g.readout_lr <= 1.0
    assert 0.0 <= g.hidden_plasticity <= 1.0
    assert 0.0 <= g.plasticity_drift <= 0.5
    d = g.to_dict()
    assert d["readout_lr"] == g.readout_lr


def test_crossover_keeps_learning_genes():
    a = random_genome(np.random.default_rng(1))
    b = random_genome(np.random.default_rng(2))
    c = crossover(a, b, 1.0, 1.0, np.random.default_rng(3), 0.5, 0.5)
    assert 0.0 <= c.readout_lr <= 1.0
    assert 0.0 <= c.hidden_plasticity <= 1.0
    assert 0.0 <= c.plasticity_drift <= 0.5


def test_organism_has_learned_weights(eco):
    eco.reset()
    alive = [o for o in eco.population if o.alive]
    assert alive
    o = alive[0]
    assert o.learned_weights is not None
    assert o.learned_weights.shape == o.genome.weights.shape
    assert np.allclose(o.learned_weights, o.genome.weights)
    d = o.to_dict()
    assert "readout_lr" in d and "learning_amount" in d
    assert d["learning_amount"] == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_eco_learning.py -k "learning_genes or crossover_keeps or has_learned" -v`
Expected: FAIL（AttributeError: readout_lr / learned_weights 不存在）。

- [ ] **Step 3: 实现 —— 常量区**（`eco_engine.py` 顶部，`POP_GROWTH = 0.10` 之后插入）

```python
# 在线学习（P1）
READOUT_LR_SCALE = 0.05          # 读出层 delta 学习实际步长 = readout_lr * 此值
MATURITY_SAMPLES = 6             # 出生成熟期训练样本数
READOUT_LR_MIN, READOUT_LR_MAX = 0.0, 1.0
HIDDEN_PLASTICITY_MIN, HIDDEN_PLASTICITY_MAX = 0.0, 1.0
PLASTICITY_DRIFT_MIN, PLASTICITY_DRIFT_MAX = 0.0, 0.5
```

- [ ] **Step 4: 实现 —— Genome 字段 + to_dict**（`Genome` dataclass `threshold_scale: float = 1.0` 后追加；to_dict 返回字典追加三键）

```python
    readout_lr: float = 0.2
    hidden_plasticity: float = 0.05
    plasticity_drift: float = 0.0
```
to_dict 在 `"threshold_scale": self.threshold_scale,` 后追加：
```python
            "readout_lr": self.readout_lr,
            "hidden_plasticity": self.hidden_plasticity,
            "plasticity_drift": self.plasticity_drift,
```

- [ ] **Step 5: 实现 —— random_genome 初始化**（`random_genome` 返回 `Genome(...)` 时在 `threshold_scale=...` 后追加）

```python
        readout_lr=float(np.clip(0.2 + rng.normal(0.0, 0.10), 0.0, 1.0)),
        hidden_plasticity=float(np.clip(0.05 + rng.normal(0.0, 0.03), 0.0, 1.0)),
        plasticity_drift=float(np.clip(0.0 + rng.normal(0.0, 0.05), 0.0, 0.5)),
```

- [ ] **Step 6: 实现 —— _mutate_genome**（三处改动）

(1) 在 `threshold_scale = float(genome.threshold_scale)` 后加局部副本：
```python
    readout_lr = float(genome.readout_lr)
    hidden_plasticity = float(genome.hidden_plasticity)
    plasticity_drift = float(genome.plasticity_drift)
```
(2) 在 `trait_mutation` 分支内（`next_mutation_rate = ...` 之后）追加学习基因变异：
```python
        readout_lr = max(0.0, min(1.0, readout_lr * math.exp(rng.normal(0.0, 0.3))))
        hidden_plasticity = max(
            0.0, min(1.0, hidden_plasticity * math.exp(rng.normal(0.0, 0.5)))
        )
        plasticity_drift = max(
            0.0, min(0.5, plasticity_drift * math.exp(rng.normal(0.0, 0.3)))
        )
```
(3) 两处 `Genome(...)` 构造（无变异提前返回 + 最终返回）在 `threshold_scale,` 后各追加：
```python
            readout_lr,
            hidden_plasticity,
            plasticity_drift,
```

- [ ] **Step 7: 实现 —— crossover**（构造 `Genome(...)` 在 `float((donor.threshold_scale + other.threshold_scale) / 2.0),` 后追加）

```python
            float((donor.readout_lr + other.readout_lr) / 2.0),
            float((donor.hidden_plasticity + other.hidden_plasticity) / 2.0),
            float((donor.plasticity_drift + other.plasticity_drift) / 2.0),
```

- [ ] **Step 8: 实现 —— Organism 字段与 to_dict**（`Organism` dataclass `digit_accuracies: List[float] = field(default_factory=list)` 后追加）

```python
    learned_weights: Optional[np.ndarray] = None
    acc_ema: float = 0.5
    samples_learned: int = 0
```
在 `Organism` 类内 `to_dict` 方法前加辅助方法：
```python
    def _learning_amount(self) -> float:
        if self.learned_weights is None:
            return 0.0
        denom = float(np.linalg.norm(self.genome.weights)) + 1e-8
        return float(
            np.linalg.norm(self.learned_weights - self.genome.weights)
        ) / denom
```
to_dict 字典在 `"threshold_scale": ...` 后追加：
```python
            "readout_lr": self.genome.readout_lr,
            "hidden_plasticity": self.genome.hidden_plasticity,
            "plasticity_drift": self.genome.plasticity_drift,
            "learning_amount": self._learning_amount(),
            "samples_learned": self.samples_learned,
            "acc_ema": self.acc_ema,
```

- [ ] **Step 9: 实现 —— _kill 释放表型**（`_kill` 内 `organism.death_reason = reason` 后追加）

```python
        organism.learned_weights = None
```

- [ ] **Step 10: 实现 —— _seed_population 表型初始化**（`_seed_population` 中改为新建 `seed_genome` 变量并初始化 learned_weights）

替换：
```python
        for genome, digit in self._founder_specs:
            organism = Organism(
                uid=self._next_uid,
                genome=Genome(
                    tuple(genome.layer_sizes),
                    genome.weights.copy(),
                    genome.longevity_bonus,
                    genome.fecundity,
                    genome.wrong_tolerance,
                    genome.mutation_rate,
                    genome.wta_k,
                    genome.leak,
                    genome.input_gain,
                    genome.threshold_scale,
                ),
```
为：
```python
        for founder_genome, digit in self._founder_specs:
            seed_genome = Genome(
                tuple(founder_genome.layer_sizes),
                founder_genome.weights.copy(),
                founder_genome.longevity_bonus,
                founder_genome.fecundity,
                founder_genome.wrong_tolerance,
                founder_genome.mutation_rate,
                founder_genome.wta_k,
                founder_genome.leak,
                founder_genome.input_gain,
                founder_genome.threshold_scale,
                founder_genome.readout_lr,
                founder_genome.hidden_plasticity,
                founder_genome.plasticity_drift,
            )
            organism = Organism(
                uid=self._next_uid,
                genome=seed_genome,
                learned_weights=seed_genome.weights.copy(),
```
（其余字段 `born_round/correct/prediction/last_digit/digit_preference` 保持原样。）

- [ ] **Step 11: 运行测试确认通过**

Run: `python -m pytest tests/test_eco_learning.py -k "learning_genes or crossover_keeps or has_learned" -v`
Expected: PASS。

- [ ] **Step 12: 提交**

```bash
git add eco_engine.py tests/test_eco_learning.py
git commit -m "学习：新增学习基因与基因型/表型分离字段

- Genome 新增 readout_lr/hidden_plasticity/plasticity_drift（可遗传可变异可交叉）
- Organism 新增 learned_weights/acc_ema/samples_learned，死亡释放表型内存
- 学习量指标 _learning_amount 供前端展示

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

### Task 3: 表型 shim + forward_learn + 接线

**Files:**
- Modify: `eco_engine.py` — 在 `forward` 之后新增 `forward_learn`；在 `Organism` 类定义后新增 `_phenotype_genomes`；`step()` 改用 `forward_learn(_phenotype_genomes(alive_before), spikes)`；`_run_census` 评估与重拟合改用 `_phenotype_genomes(...)`

**Interfaces:**
- Consumes: `Genome`/`Organism` 新字段（Task 2），`_forward_group_vectorized`（返回 logits, hidden_rates），`_forward_numba_batch`（返回 logits, hidden_counts），`_softmax`，`_output_offset`。
- Produces: `forward_learn(genomes, spikes) -> (predictions:int[n], rates:(n,10), hidden_rates:List[np.ndarray])`，`hidden_rates[i]` 形状为 `(layer_sizes[-1],)`；`_phenotype_genomes(organisms) -> List[Genome]`（shim 指向表型）。`step()` 前向与体检评估均用表型。

- [ ] **Step 1: 写失败测试**（追加）

```python
from eco_engine import forward, forward_learn, _phenotype_genomes


def test_forward_learn_shapes(eco):
    eco.reset()
    alive = [o for o in eco.population if o.alive][:10]
    spikes = eco._sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    shims = _phenotype_genomes(alive)
    preds, rates, hidden_rates = forward_learn(shims, spikes)
    assert len(preds) == len(alive)
    assert rates.shape == (len(alive), 10)
    for i, o in enumerate(alive):
        assert hidden_rates[i].shape == (o.genome.layer_sizes[-1],)


def test_shims_match_genotype_when_unlearned(eco):
    eco.reset()
    alive = [o for o in eco.population if o.alive][:10]
    spikes = eco._sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    genomes = [o.genome for o in alive]
    shims = _phenotype_genomes(alive)
    p1, r1, _ = forward(genomes, spikes)
    p2, r2, _ = forward_learn(shims, spikes)
    assert np.array_equal(p1, p2)
    assert np.allclose(r1, r2)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_eco_learning.py -k "forward_learn or shims_match" -v`
Expected: FAIL（ImportError: cannot import name 'forward_learn' / '_phenotype_genomes'）。

- [ ] **Step 3: 实现 —— forward_learn**（在 `forward` 函数定义之后插入）

```python
def forward_learn(
    genomes: Sequence[Genome],
    spikes: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """同 forward，但额外返回每个个体的最后一层隐藏率（供读出层学习）。

    Returns:
        (predictions, rates, hidden_rates)：hidden_rates[i] 形状为
        (genomes[i].layer_sizes[-1],)，与输入顺序对齐。
    """
    spike_idx, spike_vals = spikes
    if not genomes:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0, 10), dtype=np.float32),
            [],
        )

    groups: Dict[Tuple[int, ...], List[int]] = {}
    for idx, genome in enumerate(genomes):
        groups.setdefault(tuple(genome.layer_sizes), []).append(idx)

    predictions = np.empty(len(genomes), dtype=np.int32)
    rates = np.empty((len(genomes), 10), dtype=np.float32)
    hidden_rates: List[Optional[np.ndarray]] = [None] * len(genomes)
    for layer_sizes, indices in groups.items():
        selected = [genomes[i] for i in indices]
        if len(selected) >= 8:
            (
                mats,
                biases,
                out_mat,
                out_bias,
                wta_ks,
                leaks,
                input_gains,
                threshold_scales,
            ) = _stack_genome_group(selected, layer_sizes)
            logits, hidden_rates_group = _forward_group_vectorized(
                mats,
                biases,
                out_mat,
                out_bias,
                wta_ks,
                leaks,
                input_gains,
                threshold_scales,
                spike_idx,
                spike_vals,
            )
            predictions[indices] = logits.argmax(axis=1).astype(np.int32)
            rates[indices] = _softmax(logits)
            for j, idx in enumerate(indices):
                hidden_rates[idx] = hidden_rates_group[j]
        else:
            logits, hidden_counts = _forward_numba_batch(
                selected, spike_idx, spike_vals
            )
            predictions[indices] = logits.argmax(axis=1).astype(np.int32)
            rates[indices] = _softmax(logits)
            n_steps = spike_idx.shape[0]
            last_n = layer_sizes[-1]
            for j, idx in enumerate(indices):
                hidden_rates[idx] = (
                    hidden_counts[j, :last_n] / float(n_steps)
                )
    return predictions, rates, [h for h in hidden_rates if h is not None]
```

- [ ] **Step 4: 实现 —— _phenotype_genomes**（在 `Organism` 类定义之后插入）

```python
def _phenotype_genomes(organisms: Sequence[Organism]) -> List[Genome]:
    """构造指向表型(学习后权重)的轻量 Genome shim，供 forward/体检使用。

    shim 复用基因型的结构与性状字段，仅把 .weights 指向 learned_weights；
    不拷贝权重数组，写回即修改个体表型。
    """
    result: List[Genome] = []
    for organism in organisms:
        g = organism.genome
        weights = (
            organism.learned_weights
            if organism.learned_weights is not None
            else g.weights
        )
        result.append(
            Genome(
                tuple(g.layer_sizes),
                weights,
                g.longevity_bonus,
                g.fecundity,
                g.wrong_tolerance,
                g.mutation_rate,
                g.wta_k,
                g.leak,
                g.input_gain,
                g.threshold_scale,
                g.readout_lr,
                g.hidden_plasticity,
                g.plasticity_drift,
            )
        )
    return result
```

- [ ] **Step 5: 实现 —— step() 前向改用表型**（`step()` 内替换）

替换：
```python
        predictions, _ = forward([o.genome for o in alive_before], spikes)
```
为：
```python
        predictions, rates, hidden_rates = forward_learn(
            _phenotype_genomes(alive_before), spikes
        )
```
（`rates`/`hidden_rates` 本任务暂未使用，Task 4 学习用；先随前向一并算出，避免二次前向。）

- [ ] **Step 6: 实现 —— 体检评估与重拟合改用表型**（`_run_census` 内三处）

(1) 替换 `genomes = [organism.genome for organism in alive]` 为 `genomes = _phenotype_genomes(alive)`。
(2) 替换 `self._fit_readouts([organism.genome for organism in top_organisms], ...)` 为 `self._fit_readouts(_phenotype_genomes(top_organisms), ...)`。
(3) 替换 top 个体重评分两行 `top_accuracy = self._score_digits([organism.genome for organism in top_organisms], spikes_cache)` 与 `top_confidence = self._confidence_digits([organism.genome for organism in top_organisms], spikes_cache)` 中的参数为 `_phenotype_genomes(top_organisms)`。

- [ ] **Step 7: 运行测试确认通过**

Run: `python -m pytest tests/test_eco_learning.py -k "forward_learn or shims_match or smoke_step" -v`
Expected: PASS。

- [ ] **Step 8: 提交**

```bash
git add eco_engine.py tests/test_eco_learning.py
git commit -m "学习：新增 forward_learn 与表型 shim，前向与体检改用表型

- forward_learn 返回最后一层隐藏率供读出层学习
- _phenotype_genomes 构造轻量 shim，现有管道零改动用上表型
- step/体检评估与重拟合均改用学习后权重（进化作用于先天+学习能力）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

### Task 4: 在线读出层 delta 学习

**Files:**
- Modify: `eco_engine.py` — `Ecosystem` 新增 `_learn_readout` 方法；`step()` 在判定对错后调用；`step()` 主循环更新 `acc_ema`

**Interfaces:**
- Consumes: `READOUT_LR_SCALE`（Task 2）、`forward_learn` 返回的 `rates`/`hidden_rates`（Task 3）、`_output_offset`。
- Produces: `Ecosystem._learn_readout(organisms, hidden_rates, rates, label) -> None`：对每个体的表型读出层做局部 delta 更新并漂移回基因型，递增 `samples_learned`。当 `readout_lr <= 0` 或 `learned_weights is None` 时跳过该个体。

- [ ] **Step 1: 写失败测试**（追加）

```python
def test_online_learning_changes_readout(eco):
    eco.reset()
    for o in eco.population:
        o.genome.readout_lr = 0.5
    alive = [o for o in eco.population if o.alive][:20]
    before = [o.learned_weights.copy() for o in alive]
    spikes = eco._sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    shims = _phenotype_genomes(alive)
    _, rates, hidden_rates = forward_learn(shims, spikes)
    label = int(eco._test_labels[0])
    eco._learn_readout(alive, hidden_rates, rates, label)
    for i, o in enumerate(alive):
        assert not np.allclose(before[i], o.learned_weights)
        assert o.samples_learned == 1


def test_zero_readout_lr_skips_learning(eco):
    eco.reset()
    for o in eco.population:
        o.genome.readout_lr = 0.0
    alive = [o for o in eco.population if o.alive][:20]
    before = [o.learned_weights.copy() for o in alive]
    spikes = eco._sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    shims = _phenotype_genomes(alive)
    _, rates, hidden_rates = forward_learn(shims, spikes)
    label = int(eco._test_labels[0])
    eco._learn_readout(alive, hidden_rates, rates, label)
    for i, o in enumerate(alive):
        assert np.allclose(before[i], o.learned_weights)
        assert o.samples_learned == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_eco_learning.py -k "online_learning or zero_readout_lr" -v`
Expected: FAIL（AttributeError: '_Ecosystem' object has no attribute '_learn_readout'）。

- [ ] **Step 3: 实现 —— _learn_readout**（`Ecosystem` 类内，`_run_census` 方法前插入）

```python
    def _learn_readout(
        self,
        organisms: Sequence[Organism],
        hidden_rates: Sequence[np.ndarray],
        rates: np.ndarray,
        label: int,
    ) -> None:
        """在线监督 delta rule：局部更新每个体的读出层表型权重。

        每个输出神经元只用自身误差(rates-onehot)与其输入活动(hidden_rate)，
        严格局部；更新后按 plasticity_drift 向基因型漂移（遗忘/稳态）。
        """
        target = np.zeros(10, dtype=np.float32)
        target[label] = 1.0
        for i, organism in enumerate(organisms):
            g = organism.genome
            lr = g.readout_lr * READOUT_LR_SCALE
            learned = organism.learned_weights
            if lr <= 0.0 or learned is None:
                continue
            err = rates[i] - target
            offset = _output_offset(g.layer_sizes)
            last_n = g.layer_sizes[-1]
            learned[offset : offset + last_n * 10] -= lr * np.outer(
                hidden_rates[i], err
            )
            learned[offset + last_n * 10 : offset + last_n * 10 + 10] -= (
                lr * err
            )
            drift = g.plasticity_drift
            if drift > 0.0:
                learned += drift * (g.weights - learned)
            organism.samples_learned += 1
```

- [ ] **Step 4: 实现 —— step() 主循环更新 acc_ema**（`step()` 内 `for idx, organism in enumerate(alive_before):` 循环里，`organism.last_digit = label` 后追加）

```python
            organism.acc_ema = (
                0.9 * organism.acc_ema + 0.1 * float(organism.correct)
            )
```

- [ ] **Step 5: 实现 —— EcoConfig 加 learning_on 开关**（`EcoConfig` dataclass `feed_interval: float = 5.0` 后追加）

```python
    learning_on: bool = True
```
`to_dict` 返回字典追加：
```python
            "learning_on": self.learning_on,
```
`update` 方法末尾追加：
```python
        if "learning_on" in values:
            self.learning_on = bool(values["learning_on"])
```

- [ ] **Step 6: 实现 —— step() 调用在线学习**（`step()` 内，在自然死亡判定循环之后、`survivors = [...]` 之前插入）

```python
        if self.config.learning_on:
            self._learn_readout(alive_before, hidden_rates, rates, label)
```
（`learning_on` 已在上一步加入 EcoConfig。）

- [ ] **Step 7: 运行测试确认通过**

Run: `python -m pytest tests/test_eco_learning.py -k "online_learning or zero_readout_lr or smoke_step" -v`
Expected: PASS。

- [ ] **Step 8: 提交**

```bash
git add eco_engine.py tests/test_eco_learning.py
git commit -m "学习：读出层在线监督 delta rule

- 每回合喂食后对全部存活个体局部更新读出层表型权重
- 按 plasticity_drift 向基因型漂移作稳态，samples_learned 累计
- EcoConfig 新增 learning_on 学习总开关（R0 对照组）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

### Task 5: 成熟期发育 + 配置开关

**Files:**
- Modify: `eco_engine.py` — `EcoConfig`（新增 `maturity_samples` + to_dict/update）；`Ecosystem` 新增 `_maturate`；`step()` 繁殖循环收集 newborns 并调用成熟期

**Interfaces:**
- Consumes: `EcoConfig.maturity_samples`（本任务）、`EcoConfig.learning_on`（Task 4）、`_learn_readout`/`forward_learn`/`_phenotype_genomes`（Task 3/4）、`_train_images`/`_train_labels`。
- Produces: `EcoConfig.maturity_samples:int=6`；`Ecosystem._maturate(newborns) -> None`：用 MATURITY_SAMPLES 张训练集图批量发育新生儿表型。

- [ ] **Step 1: 写失败测试**（追加）

```python
def test_maturate_updates_readout(eco):
    eco.reset()
    eco.config.maturity_samples = 3
    orgs = [o for o in eco.population if o.alive][:5]
    for o in orgs:
        o.genome.readout_lr = 0.5
    before = [o.learned_weights.copy() for o in orgs]
    eco._maturate(orgs)
    for i, o in enumerate(orgs):
        assert o.samples_learned == 3
        assert not np.allclose(before[i], o.learned_weights)


def test_step_respects_learning_switch(eco):
    eco.reset()
    eco.config.learning_on = False
    for o in eco.population:
        o.genome.readout_lr = 0.5
    before = {
        o.uid: o.learned_weights.copy()
        for o in eco.population
        if o.alive
    }
    eco.step()
    for o in eco.population:
        if o.alive and o.uid in before:
            assert np.allclose(before[o.uid], o.learned_weights)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_eco_learning.py -k "maturate or learning_switch" -v`
Expected: FAIL（AttributeError: maturity_samples / _maturate 不存在）。

- [ ] **Step 3: 实现 —— EcoConfig 加 maturity_samples 字段**（`EcoConfig` dataclass `feed_interval: float = 5.0` 后追加；`learning_on` 已在 Task 4 加入）

```python
    maturity_samples: int = MATURITY_SAMPLES
```
`to_dict` 返回字典追加：
```python
            "maturity_samples": self.maturity_samples,
```
`update` 方法末尾追加：
```python
        if "maturity_samples" in values:
            self.maturity_samples = int(
                min(50, max(0, int(values["maturity_samples"])))
            )
```

- [ ] **Step 4: 实现 —— _maturate**（`Ecosystem` 类内，`_learn_readout` 方法后插入）

```python
    def _maturate(self, newborns: Sequence[Organism]) -> None:
        """出生成熟期：用 MATURITY_SAMPLES 张训练集图批量发育新生儿表型。

        用训练集而非食物流，保证评测诚实；每张样本对全体新生儿批量前向，
        走同一个 _learn_readout（读出层 delta）。
        """
        k = self.config.maturity_samples
        if k <= 0 or not newborns:
            return
        for _ in range(k):
            label = int(self.rng.integers(10))
            candidates = np.flatnonzero(self._train_labels == label)
            if len(candidates) == 0:
                continue
            index = int(candidates[self.rng.integers(len(candidates))])
            image = self._train_images[index]
            spikes = _sample_spikes(image, self.rng)
            shims = _phenotype_genomes(newborns)
            _, rates, hidden_rates = forward_learn(shims, spikes)
            self._learn_readout(newborns, hidden_rates, rates, label)
```

- [ ] **Step 5: 实现 —— step() 收集 newborns 并成熟**（`step()` 内繁殖块）

(1) 在 `offspring = 0` 行后追加 `newborns: List[Organism] = []`。
(2) 繁殖循环内，把现有个体构造改为先算 `child_genome`，再构造 `child` 并收集：
替换：
```python
                    child = Organism(
                        uid=self._next_uid,
                        genome=crossover(
                            first.genome,
                            second.genome,
                            first.age,
                            second.age,
                            self.rng,
                            first.fitness,
                            second.fitness,
                        ),
                        born_round=self.round,
                        digit_preference=(
                            first.digit_preference
                            if self.rng.random() < 0.5
                            else second.digit_preference
                        ),
                    )
```
为：
```python
                    child_genome = crossover(
                        first.genome,
                        second.genome,
                        first.age,
                        second.age,
                        self.rng,
                        first.fitness,
                        second.fitness,
                    )
                    child = Organism(
                        uid=self._next_uid,
                        genome=child_genome,
                        born_round=self.round,
                        digit_preference=(
                            first.digit_preference
                            if self.rng.random() < 0.5
                            else second.digit_preference
                        ),
                        learned_weights=child_genome.weights.copy(),
                    )
```
(3) 在 `self.population.append(child)` 后追加 `newborns.append(child)`。
(4) 在繁殖 `if survivors and target_pop > len(survivors):` 块结束后（`for organism in survivors:` 未配对计数循环之前）插入：
```python
        if self.config.learning_on:
            self._maturate(newborns)
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_eco_learning.py -k "maturate or learning_switch or smoke_step" -v`
Expected: PASS（smoke_step 因新增字段仍应通过）。

- [ ] **Step 7: 提交**

```bash
git add eco_engine.py tests/test_eco_learning.py
git commit -m "学习：出生成熟期发育与学习总开关

- EcoConfig 新增 maturity_samples/learning_on，经 update 支持前端滑块
- 新生儿出生用训练集图批量发育表型，食物流与学习流分离
- learning_on 关闭时在线学习与成熟期均不生效（R0 对照组）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

### Task 6: 前端解剖面板展示学习字段

**Files:**
- Modify: `eco_game.html` — `renderInspect` 网格与取值

**Interfaces:**
- Consumes: `Organism.to_dict()` 新增字段：`readout_lr`、`hidden_plasticity`、`plasticity_drift`、`learning_amount`、`samples_learned`、`acc_ema`（Task 2）。`state.alive[]` 对象含这些键。
- Produces: 解剖面板可视化学习基因、学习量、学习样本数。

- [ ] **Step 1: 网格加 6 个单元**（`renderInspect` 模板字符串内，`最差数字` 单元之前追加）

```html
          <div class="cell"><span>读出学习率</span><strong id="inspectReadoutLr"></strong></div>
          <div class="cell"><span>隐藏可塑性</span><strong id="inspectHiddenPlasticity"></strong></div>
          <div class="cell"><span>漂移率</span><strong id="inspectDrift"></strong></div>
          <div class="cell"><span>学习量</span><strong id="inspectLearning"></strong></div>
          <div class="cell"><span>学习样本</span><strong id="inspectSamples"></strong></div>
          <div class="cell"><span>准确基线</span><strong id="inspectAccEma"></strong></div>
```

- [ ] **Step 2: 取值填充**（`renderInspect` 取值段，`inspectWorstDigit` 赋值后追加）

```javascript
      $("inspectReadoutLr").textContent = (organism.readout_lr ?? 0).toFixed(3);
      $("inspectHiddenPlasticity").textContent = (organism.hidden_plasticity ?? 0).toFixed(3);
      $("inspectDrift").textContent = (organism.plasticity_drift ?? 0).toFixed(3);
      $("inspectLearning").textContent = ((organism.learning_amount ?? 0) * 100).toFixed(1) + "%";
      $("inspectSamples").textContent = organism.samples_learned ?? 0;
      $("inspectAccEma").textContent = (organism.acc_ema ?? 0.5).toFixed(3);
```

- [ ] **Step 3: 手动验证**（启动服务并点击解剖）

Run: `python eco_server.py --port 8765`，浏览器打开 `http://127.0.0.1:8765`，跑几回合后点击个体。
Expected: 解剖面板显示六个学习字段且值合理（未学习个体学习量=0%，学习后 >0%）。

- [ ] **Step 4: 提交**

```bash
git add eco_game.html
git commit -m "前端：解剖面板展示学习基因/学习量/学习样本

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

### Task 7: 诚实验证 R0 vs R1

**Files:**
- Create: `scripts/compare_learning.py`
- 运行并如实记录结果到 `docs/superpowers/notes/2026-08-09-online-learning-p1-results.md`

**Interfaces:**
- Consumes: `EcoConfig.learning_on`（Task 5）、`Ecosystem` 全量接口。
- Produces: 对照组 accuracy 轨迹与结论（诚实报告）。

- [ ] **Step 1: 写对照脚本**（创建 `scripts/compare_learning.py`）

```python
"""P1 诚实验证：R0(纯遗传，learning_on=False) vs R1(在线学习，learning_on=True)。

固定 seed 各跑 N 回合，输出每回合 accuracy 与期末均值/最优。
"""

import argparse
import json

import numpy as np

from eco_engine import EcoConfig, Ecosystem


def run(seed: int, rounds: int, learning_on: bool) -> list:
    config = EcoConfig(init_pop=1000, learning_on=learning_on)
    eco = Ecosystem(config=config, seed=seed)
    accs = []
    for _ in range(rounds):
        events = eco.step()
        accs.append(float(events.get("accuracy", 0.0)))
    return accs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    results = {"rounds": args.rounds, "seeds": args.seeds, "runs": {}}
    for seed in args.seeds:
        for tag, learning_on in (("R0_no_learning", False), ("R1_learning", True)):
            accs = run(seed, args.rounds, learning_on)
            results["runs"][f"{tag}_seed{seed}"] = {
                "trajectory": accs,
                "mean": float(np.mean(accs)),
                "last10_mean": float(np.mean(accs[-10:])),
                "max": float(np.max(accs)),
            }
            print(
                f"seed={seed} {tag}: mean={np.mean(accs):.3f} "
                f"last10={np.mean(accs[-10:]):.3f} max={np.max(accs):.3f}"
            )
    with open("docs/superpowers/notes/2026-08-09-online-learning-p1-results.json", "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行对照**

Run: `mkdir -p docs/superpowers/notes && python scripts/compare_learning.py --rounds 30 --seeds 1 2 3`
Expected: 输出 R0/R1 均值、末段均值、最优。**如实记录**：若 R1 未优于 R0，如实写明，不夸大（用户明确要求诚实）。

- [ ] **Step 3: 写结果记录**（`docs/superpowers/notes/2026-08-09-online-learning-p1-results.md`，填入实际数字）

```markdown
# 在线学习 P1 验证结果

日期：2026-08-09
条件：init_pop=1000，rounds=30，seeds=1/2/3，R0=learning_on False，R1=True

## 结果

（从运行输出填入三行 R0/R1 的 mean / last10 / max）

## 结论

- 读出层在线学习是否显著提升 accuracy？如实回答。
- 学习增益主要来自成熟期还是每回合在线？可补充 A/B。
- 对 P2（隐藏层三因子）的启示。
```

- [ ] **Step 4: 提交**

```bash
git add scripts/compare_learning.py docs/superpowers/notes/
git commit -m "验证：在线学习 R0 vs R1 对照与诚实结果记录

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

## 后续计划（本次不做）

- **P2**：隐藏层三因子奖励调制（速率协方差资格迹 × 对错奖励）+ 漂移/突触缩放稳态 + `hidden_learn_every` 配置。`hidden_plasticity` 基因已就位。
- **P3**：体检"睡眠巩固"重拟合收尾、前端学习量曲线/隐藏可塑性进化轨迹/成熟期滑块。

## 自检记录

- 规格覆盖：数据模型（Task 2）、读出层 delta（Task 4）、成熟期（Task 5）、体检表型（Task 3）、前端解剖（Task 6）、诚实验证（Task 7）。P2 隐藏层三因子、P3 完整前端明确排到后续计划，非遗漏。
- 占位符：无 TBD/TODO。
- 类型一致：`_phenotype_genomes`、`forward_learn`、`_learn_readout`、`_maturate` 签名在任务间引用一致；`hidden_rates` 统一为 `List[np.ndarray]` 且与输入顺序对齐。
