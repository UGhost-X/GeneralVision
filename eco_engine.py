# eco_engine.py（本任务先写：常量、Genome、normalize、random_genome、crossover）
"""LIF 生态游戏引擎：喂食-产出-淘汰-有性繁殖（纯 numpy，一生无学习）。

生物体 = LIF 网络（784 输入 → 100 隐藏神经元 → 10 产出神经元），所有权重
出生即随机、一生固定。适应度 = 每天吃对率。有性繁殖 = 逐权重 50/50 取双亲
+ 高斯扰动 + 千分之一大突变。输出标准化事件流供前端播放。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# ---- 游戏与网络参数（阈值于 Task 2 校准并验证：默认值经 200 样本 MNIST 实测，
#       隐藏层每场总发放均值≈99、产出 None 率 13.5%、10 通道全用，无需调参） ----
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


# ---- 生态主循环（Task 3 追加）----
from data_loading import load_mnist


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
            while len(survivors) < POP_CAP:
                if len(survivors) >= 2:
                    # 每轮按当前幸存者重算 roulette 权重（新生儿无当日 acc，权重≈0）
                    w = np.array([accs.get(g.name, 0.0) + 1e-6 for g in survivors],
                                 np.float64)
                    probs = w / w.sum()
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
