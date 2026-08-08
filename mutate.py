"""变异算子：无性分裂（克隆 + 极端变异）+ 变异强度随代数衰减。

每个子代 = 亲代克隆 + 随机施加若干变异：
- 连续超参高斯扰动（幅度随代数衰减）
- 每层神经元数增减
- 增删/复制整层（极端变异的核心理念）
- 全局超参（T、spike_gain、train_samples、seed）

变异后所有参数钳制到合法范围，保证子代可用。权重的"重新初始化"体现在
LayerConfig.seed 的变异上（一生重新学，见 evaluate.py）。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from genome import Genome
from snn import LayerConfig

# 各超参的合法范围与相对扰动幅度
PARAM_BOUNDS = {
    "n_out": (20, 200),
    "leak": (0.80, 0.995),
    "theta_init": (8.0, 60.0),
    "refr_period": (1, 8),
    "tau_plus": (1, 6),
    "a_plus": (0.05, 2.0),
    "w_norm": (4.0, 40.0),
    "rate_alpha": (0.0005, 0.02),
    "beta": (100.0, 1000.0),
    "input_gain": (0.5, 200.0),
    "w_init_mean": (0.02, 0.5),
}
SCALAR_BOUNDS = {
    "spike_gain": (0.2, 1.2),
    "T": (50, 300),
    "train_samples": (100, 1500),
}
MAX_LAYERS = 4
MIN_LAYERS = 1


@dataclass
class MutationSchedule:
    sigma0: float = 0.15          # 第 0 代连续超参相对扰动幅度
    sigma_min: float = 0.02
    g_max: int = 50               # 衰减到 sigma_min 的代数
    structural_p0: float = 0.35   # 第 0 代结构性变异（增删层）概率
    structural_min: float = 0.05

    def sigma(self, generation: int) -> float:
        frac = min(1.0, generation / max(1, self.g_max))
        return self.sigma_min + (self.sigma0 - self.sigma_min) * (1.0 - frac)

    def structural_p(self, generation: int) -> float:
        frac = min(1.0, generation / max(1, self.g_max))
        return self.structural_min + (self.structural_p0 - self.structural_min) * (1.0 - frac)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _perturb(rng: np.random.Generator, v: float, lo: float, hi: float,
             sigma: float) -> float:
    """对数空间高斯扰动，保持正值参数尺度合理。"""
    log_v = math.log(v)
    new = math.exp(log_v + rng.normal(0.0, sigma))
    return _clamp(new, lo, hi)


def _random_layer(rng: np.random.Generator, depth: int) -> LayerConfig:
    """随机生成一个新层。depth=0 输入是像素（gain=1），更深层需更高 gain。"""
    n_out = int(rng.integers(25, 120))
    input_gain = 1.0 if depth == 0 else float(10 ** rng.uniform(0.8, 2.0))
    return LayerConfig(
        n_out=n_out,
        w_norm=float(rng.uniform(8.0, 24.0)),
        theta_init=float(rng.uniform(15.0, 35.0)),
        theta_clamp=(5.0, 100.0),
        input_gain=input_gain,
        seed=int(rng.integers(0, 2**31)),
    )


def _mutate_layer(rng: np.random.Generator, layer: LayerConfig,
                  sigma: float) -> LayerConfig:
    """对单层做连续/离散超参扰动。"""
    l = copy.deepcopy(layer)
    b = PARAM_BOUNDS

    if rng.random() < 0.6:
        l.leak = _perturb(rng, l.leak, *b["leak"], sigma)
    if rng.random() < 0.6:
        l.theta_init = _perturb(rng, l.theta_init, *b["theta_init"], sigma)
    if rng.random() < 0.4:
        l.refr_period = int(_clamp(l.refr_period + int(rng.normal(0, 1)), *b["refr_period"]))
    if rng.random() < 0.4:
        l.tau_plus = int(_clamp(l.tau_plus + int(rng.normal(0, 1)), *b["tau_plus"]))
    if rng.random() < 0.5:
        l.a_plus = _perturb(rng, l.a_plus, *b["a_plus"], sigma)
    if rng.random() < 0.6:
        l.w_norm = _perturb(rng, l.w_norm, *b["w_norm"], sigma)
    if rng.random() < 0.4:
        l.rate_alpha = _perturb(rng, l.rate_alpha, *b["rate_alpha"], sigma)
    if rng.random() < 0.4:
        l.beta = _perturb(rng, l.beta, *b["beta"], sigma)
    if rng.random() < 0.4:
        l.input_gain = _perturb(rng, l.input_gain, *b["input_gain"], sigma)
    if rng.random() < 0.4:
        l.w_init_mean = _perturb(rng, l.w_init_mean, *b["w_init_mean"], sigma)
    # 神经元数增减
    if rng.random() < 0.5:
        delta = int(rng.normal(0, 15))
        l.n_out = int(_clamp(l.n_out + delta, *b["n_out"]))
    # 权重初始化 seed 变异 → 一生重新学
    l.seed = int(rng.integers(0, 2**31))
    return l


def mutate(genome: Genome, generation: int, rng: np.random.Generator,
           schedule: MutationSchedule | None = None) -> Genome:
    """无性分裂：克隆亲代并极端变异，返回子代 genome。"""
    sched = schedule or MutationSchedule()
    sigma = sched.sigma(generation)
    p_struct = sched.structural_p(generation)
    child = genome.clone()
    child.name = f"{genome.name}.c"

    # 逐层连续/离散超参扰动
    child.layers = [_mutate_layer(rng, l, sigma) for l in child.layers]

    # 结构性变异：增删/复制整层
    if rng.random() < p_struct and len(child.layers) < MAX_LAYERS:
        pos = int(rng.integers(0, len(child.layers) + 1))
        new_layer = _random_layer(rng, pos)
        child.layers.insert(pos, new_layer)
    if rng.random() < p_struct * 0.5 and len(child.layers) > MIN_LAYERS:
        del_idx = int(rng.integers(0, len(child.layers)))
        del child.layers[del_idx]
    if rng.random() < p_struct * 0.5 and len(child.layers) < MAX_LAYERS:
        src = int(rng.integers(0, len(child.layers)))
        dup = _mutate_layer(rng, child.layers[src], sigma * 2.0)
        pos = int(rng.integers(0, len(child.layers) + 1))
        child.layers.insert(pos, dup)

    # 全局超参变异
    if rng.random() < 0.5:
        child.spike_gain = _perturb(rng, child.spike_gain, *SCALAR_BOUNDS["spike_gain"], sigma)
    if rng.random() < 0.4:
        child.T = int(_clamp(child.T + int(rng.normal(0, 20)), *SCALAR_BOUNDS["T"]))
    if rng.random() < 0.4:
        child.train_samples = int(_clamp(
            child.train_samples + int(rng.normal(0, 100)), *SCALAR_BOUNDS["train_samples"]))
    child.seed = int(rng.integers(0, 2**31))

    return child


def _self_check() -> None:
    from genome import seed_genome
    rng = np.random.default_rng(0)
    parent = seed_genome(seed=0)
    print("parent:", parent.describe())
    for g in range(5):
        sched = MutationSchedule()
        child = mutate(parent, g, rng, sched)
        assert 1 <= len(child.layers) <= MAX_LAYERS
        assert all(PARAM_BOUNDS["n_out"][0] <= l.n_out <= PARAM_BOUNDS["n_out"][1] for l in child.layers)
        print(f"gen {g}: {child.describe()}")
    print("mutate self-check OK")


if __name__ == "__main__":
    _self_check()
