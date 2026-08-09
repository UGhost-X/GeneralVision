# eco_engine.py（本任务先写：常量、Genome、normalize、random_genome、crossover）
"""LIF 生态游戏引擎（v2 回合制）：喂食-产出-淘汰-有性繁殖（纯 numpy，一生无学习）。

生物体 = LIF 网络（784 输入 → 100 隐藏神经元 → 10 产出神经元），所有权重
出生即随机、一生固定。一回合喂 1 个数字：产出错误的当回合死亡（非自然），
活到存活回合数上限自然死亡；存活者随机两两配对、按存活时长加权交叉繁殖，
密度依赖 + 承载力封顶。输出标准化事件流供前端播放。
"""
from __future__ import annotations

import numpy as np
import numba
from dataclasses import dataclass

# ---- 游戏参数（v2 回合制） ----
T = 40                    # 仿真步数。v2 加速：原 120 时满容量(500)回合 ~4.9s，降为 40 后 ~2.4s（~2×），
                          # 正确率仍贴随机线、None 率<0.7、12 测试全绿；T 越长单回合越慢（每只 ~0.06ms/步）
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
CAPACITY = 500           # 环境承载力（种群上限；满容量实测均值约 4.5s、峰值约 5.5s）
DENSITY_FLOOR = 0.05     # 密度地板：承载力处仍有 5% 替代性繁殖，防"满→90%暴毙→回填"锯齿
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


@numba.njit(cache=True)
def _normalize_cols(W: np.ndarray, norm: float) -> np.ndarray:
    """每列归一化到指定 L2 范数（保证随机权重动力学不爆/不哑）。numba JIT 版。"""
    out = W.copy()
    nrow, ncol = out.shape
    for j in range(ncol):
        s = 0.0
        for i in range(nrow):
            s += out[i, j] * out[i, j]
        inv = norm / (np.sqrt(s) + 1e-8)
        for i in range(nrow):
            out[i, j] *= inv
    return out


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
    """有性繁殖：逐权重以 pa 取 a（存活时长加权）+ 均匀小扰动 + 千分之一大突变。

    默认 weight_a=weight_b → 50/50；weight_a 越大后代越偏 a（活越久的个体基因占比越大）。
    扰动用均匀分布（rng.uniform(-σ,σ)）代替高斯：同为"小扰动"，但均匀生成比正态快 ~4×，
    是 4500 只/回合级繁殖的主要提速点（2026-08-09 v2 性能优化）。
    """
    wsum = weight_a + weight_b
    pa = weight_a / wsum if wsum > 0 else 0.5
    hidden = np.where(rng.random(a.hidden.shape) < pa, a.hidden, b.hidden)
    hidden += rng.uniform(-CROSS_SIGMA, CROSS_SIGMA, hidden.shape)
    mh = rng.random(hidden.shape) < MUT_RATE
    hidden[mh] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(mh.sum()))
    hidden = _normalize_cols(hidden, W_NORM_HIDDEN)

    readout = np.where(rng.random(a.readout.shape) < pa, a.readout, b.readout)
    readout += rng.uniform(-CROSS_SIGMA, CROSS_SIGMA, readout.shape)
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


@numba.njit(cache=True, inline="always")
def _forward_core(S: np.ndarray, Wh: np.ndarray, Wr: np.ndarray,
                  leak: float, theta_h: float, theta_r: float,
                  ref_period: int, n_t: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """numba JIT 版 LIF 前向核心：逐 b 逐 t 放电。

    S:(B,784,T) float32 泊松脉冲；Wh:(784,n_h) float64；Wr:(n_h,n_r) float64。
    镜像 numpy forward 语义：累积→不应期清零→漏电→WTA(首个最大者)→发放→不应期递减；
    产出层随隐藏层发放同步积分+WTA。返回 (produced[B] int64, hc(B,n_h) int64, rc(B,n_r) int64)。
    """
    B = S.shape[0]
    n_in = S.shape[1]
    n_h = Wh.shape[1]
    n_r = Wr.shape[1]
    Vh = np.zeros((B, n_h), dtype=np.float32)
    refh = np.zeros((B, n_h), dtype=np.int32)
    Vr = np.zeros((B, n_r), dtype=np.float32)
    refr = np.zeros((B, n_r), dtype=np.int32)
    hc = np.zeros((B, n_h), dtype=np.int64)
    rc = np.zeros((B, n_r), dtype=np.int64)
    for t in range(n_t):
        for b in range(B):
            # 隐藏层：稀疏点积先累进 float64 行（与 numpy 完整 float64 点积一致），再一次性加进 float32 Vh
            row = np.zeros(n_h, dtype=np.float64)
            for i in range(n_in):
                if S[b, i, t] > 0.0:
                    for j in range(n_h):
                        row[j] += Wh[i, j]
            for j in range(n_h):
                Vh[b, j] += row[j]
            for j in range(n_h):
                if refh[b, j] > 0:
                    Vh[b, j] = 0.0
                Vh[b, j] *= leak
            best = -1
            best_v = -1.0e30
            for j in range(n_h):
                if refh[b, j] <= 0 and Vh[b, j] >= theta_h and Vh[b, j] > best_v:
                    best_v = Vh[b, j]
                    best = j
            if best >= 0:
                for j in range(n_h):
                    Vh[b, j] = 0.0
                    if refh[b, j] <= 0:
                        refh[b, j] = 1
                refh[b, best] = ref_period
                hc[b, best] += 1
                # 产出层：仅当隐藏发放时累加 Wr[best, :]（hspk=onehot(best) @ Wr）
                for k in range(n_r):
                    Vr[b, k] += Wr[best, k]
            # 产出层：每步都对所有行漏电 + 不应期清零 + WTA（与 numpy 一致，不依赖隐藏层是否发放）
            for k in range(n_r):
                if refr[b, k] > 0:
                    Vr[b, k] = 0.0
                Vr[b, k] *= leak
            best_r = -1
            best_vr = -1.0e30
            for k in range(n_r):
                if refr[b, k] <= 0 and Vr[b, k] >= theta_r and Vr[b, k] > best_vr:
                    best_vr = Vr[b, k]
                    best_r = k
            if best_r >= 0:
                for k in range(n_r):
                    Vr[b, k] = 0.0
                    if refr[b, k] <= 0:
                        refr[b, k] = 1
                refr[b, best_r] = ref_period
                rc[b, best_r] += 1
            # 不应期递减（每步对每行执行）
            for j in range(n_h):
                if refh[b, j] > 0:
                    refh[b, j] -= 1
            for k in range(n_r):
                if refr[b, k] > 0:
                    refr[b, k] -= 1
    produced = np.empty(B, dtype=np.int64)
    for b in range(B):
        total = 0
        bestp = -1
        bestc = -1
        for k in range(n_r):
            total += rc[b, k]
            if rc[b, k] > bestc:
                bestc = rc[b, k]
                bestp = k
        produced[b] = bestp if total > 0 else -1
    return produced, hc, rc


def forward(genome: Genome, pixels: np.ndarray, rng: np.random.Generator
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pixels: (B,784) ∈[0,1]。返回 (produced, hidden_counts, readout_counts)。

    泊松编码（numpy 生成脉冲）→ numba JIT 核心（隐藏层 LIF+WTA 无学习 → 产出层 LIF）。
    produced = 产出层累计发放最多的数字；-1 表示整场未发放。
    """
    B = pixels.shape[0]
    S = (rng.random((B, 784, T), dtype=np.float32) < (pixels[:, :, None] * SPIKE_GAIN)).astype(np.float32)
    produced, hc, rc = _forward_core(S, genome.hidden, genome.readout,
                                     LEAK, THETA_HIDDEN, THETA_READOUT, REF_PERIOD, T)
    return produced, hc, rc


def forward_from_S(genome: Genome, S: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复用已生成的泊松脉冲 S 做前向（回合内所有个体共享同一 S，省去逐只生成脉冲）。

    与 forward 等价，只是脉冲由调用方一次生成。S:(1,784,T) float32。
    """
    produced, hc, rc = _forward_core(S, genome.hidden, genome.readout,
                                     LEAK, THETA_HIDDEN, THETA_READOUT, REF_PERIOD, T)
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
        n = min(self.initial_pop, self.capacity)   # 初始种群不得超承载力（滑块互相独立）
        self.pop: list[Genome] = [
            random_genome(f"eco#{i}", self.rng, gen=0) for i in range(n)
        ]
        self.counter = n

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
        # 每回合只生成一份泊松脉冲 S，全部个体共享（省 5000×0.27ms 的逐只脉冲生成；确定性由 seed 派生保证）
        food_pix = self._img[food_idx][None]
        S = (np.random.default_rng(self.round * 1_000_003).random((1, 784, T), dtype=np.float32)
             < (food_pix[:, :, None] * SPIKE_GAIN)).astype(np.float32)
        for i, g in enumerate(self.pop):
            produced, _hc, rc = forward_from_S(g, S)
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
            density = max(DENSITY_FLOOR, 1.0 - len(self.pop) / self.capacity)
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
            n = min(self.initial_pop, self.capacity)   # 重播种群数同样受承载力封顶
            self.pop = [random_genome(f"eco#{self.counter + i}", self.rng, gen=self.round)
                        for i in range(n)]
            self.counter += n
            events.append({"type": "reseed", "count": n})

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
