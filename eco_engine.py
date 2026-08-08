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
