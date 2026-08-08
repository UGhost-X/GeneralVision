# eco_engine.py（本任务先写：常量、Genome、normalize、random_genome、crossover）
"""LIF 生态游戏引擎（v2 回合制）：喂食-产出-淘汰-有性繁殖（纯 numpy，一生无学习）。

生物体 = LIF 网络（784 输入 → 100 隐藏神经元 → 10 产出神经元），所有权重
出生即随机、一生固定。一回合喂 1 个数字：产出错误的当回合死亡（非自然），
活到存活回合数上限自然死亡；存活者随机两两配对、按存活时长加权交叉繁殖，
密度依赖 + 承载力封顶。输出标准化事件流供前端播放。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

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
CAPACITY = 500           # 环境承载力（种群上限；全容量回合约 3-4s，网页游戏可玩）
INIT_POP = 60            # 初始/全灭重播种群数


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


def forward(genome: Genome, pixels: np.ndarray, rng: np.random.Generator
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pixels: (B,784) ∈[0,1]。返回 (produced, hidden_counts, readout_counts)。

    泊松编码 → 隐藏层 LIF+WTA（无 STDP、无 homeostasis，纯放电）→ 产出层 LIF。
    produced = 产出层累计发放最多的数字；-1 表示整场未发放。
    逐样本重置 V/refr，与 snn.py step() 语义一致。
    """
    B = pixels.shape[0]
    # 直接用 float32 生成泊松随机发放，避免 float64 中间数组的双倍内存分配
    S = (rng.random((B, 784, T), dtype=np.float32) < (pixels[:, :, None] * SPIKE_GAIN)).astype(np.float32)
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


# ---- 生态主循环（v2 回合制）----
from data_loading import load_mnist


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
