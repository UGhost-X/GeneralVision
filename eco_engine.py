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
