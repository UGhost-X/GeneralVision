"""eco2_engine.py — eco2 能量生态游戏批量仿真引擎（torch + spikingjelly 原语）。

个体用 mSTDP 奖励调制在运行中学习读出层；能量经济定生死；拉马克混合遗传。
详细设计见 docs/superpowers/specs/2026-08-09-eco2-design.md（本地，gitignored）。
"""
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from spikingjelly.activation_based import functional as sjf
except ImportError:
    _sj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spikingjelly")
    sys.path.insert(0, _sj)
    for _m in list(sys.modules):
        if _m == "spikingjelly" or _m.startswith("spikingjelly."):
            sys.modules.pop(_m, None)
    from spikingjelly.activation_based import functional as sjf


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_mnist(root: str = "data/MNIST/raw") -> Tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(root, "train-images-idx3-ubyte"), "rb") as f:
        buf = f.read()
    _, n, rows, cols = struct.unpack(">IIII", buf[:16])
    imgs = np.frombuffer(buf[16:], dtype=np.uint8).reshape(n, rows, cols)
    imgs = imgs.astype(np.float32) / 255.0
    with open(os.path.join(root, "train-labels-idx1-ubyte"), "rb") as f:
        buf = f.read()
    _, n = struct.unpack(">II", buf[:8])
    labs = np.frombuffer(buf[8:], dtype=np.uint8)
    return imgs, labs


def downsample(imgs: np.ndarray, size: int = 14) -> np.ndarray:
    """28x28 -> size×size 平均池化并 flatten。imgs: [N,28,28] -> [N, size*size]"""
    n = imgs.shape[0]
    step = 28 // size
    x = imgs.reshape(n, size, step, size, step).mean(axis=(2, 4))
    return x.reshape(n, size * size).astype(np.float32)


# --------------------------------------------------------------------------- #
# 基因
# --------------------------------------------------------------------------- #
@dataclass
class Genome:
    layer_sizes: Tuple[int, ...] = (64,)
    wta_k: int = 6
    leak: float = 0.94
    input_gain: float = 1.0
    threshold_scale: float = 1.0
    lamarckism: float = 0.5
    lr_scale: float = 1.0
    fecundity: int = 1
    mutation_rate: float = 0.1

    def to_dict(self) -> dict:
        return {
            "layer_sizes": list(self.layer_sizes), "wta_k": self.wta_k,
            "leak": self.leak, "input_gain": self.input_gain,
            "threshold_scale": self.threshold_scale, "lamarckism": self.lamarckism,
            "lr_scale": self.lr_scale, "fecundity": self.fecundity,
            "mutation_rate": self.mutation_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        d = dict(d)
        d["layer_sizes"] = tuple(d["layer_sizes"])
        return cls(**d)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def random_genome(rng: np.random.Generator) -> Genome:
    n_layers = int(rng.integers(1, 4))
    return Genome(
        layer_sizes=tuple(int(rng.integers(32, 129)) for _ in range(n_layers)),
        wta_k=int(rng.integers(2, 13)),
        leak=float(rng.uniform(0.80, 0.99)),
        input_gain=float(rng.uniform(0.5, 3.0)),
        threshold_scale=float(rng.uniform(0.5, 2.0)),
        lamarckism=float(rng.uniform(0.0, 1.0)),
        lr_scale=float(rng.uniform(0.5, 2.0)),
        fecundity=int(rng.integers(1, 4)),
        mutation_rate=float(rng.uniform(0.02, 0.2)),
    )


def mutate_genome(g: Genome, rng: np.random.Generator) -> Genome:
    """生态基因高斯扰动 + 结构基因偶发扰动，全部钳制在范围内。

    结构变异为偶发事件：默认保持亲代结构，仅 ~15-20% 概率发生增层/删层/缩放。
    """
    layer_sizes = g.layer_sizes
    r = rng.random()
    if r < 0.05 and len(layer_sizes) < 3:
        layer_sizes = tuple(sorted((*layer_sizes, int(rng.integers(32, 129)))))
    elif r < 0.10 and len(layer_sizes) > 1:
        layer_sizes = layer_sizes[:-1]
    elif r < 0.20:
        n = rng.normal(0, 1)
        layer_sizes = tuple(_clamp(int(s * (1.0 + 0.1 * n)), 32, 128) for s in g.layer_sizes)
    return Genome(
        layer_sizes=layer_sizes,
        wta_k=int(_clamp(int(g.wta_k + round(rng.normal(0, 1))), 2, 12)),
        leak=float(_clamp(g.leak + rng.normal(0, 0.01), 0.80, 0.99)),
        input_gain=float(_clamp(g.input_gain * (1 + rng.normal(0, 0.1)), 0.5, 3.0)),
        threshold_scale=float(_clamp(g.threshold_scale * (1 + rng.normal(0, 0.1)), 0.5, 2.0)),
        lamarckism=float(_clamp(g.lamarckism + rng.normal(0, 0.05), 0.0, 1.0)),
        lr_scale=float(_clamp(g.lr_scale * (1 + rng.normal(0, 0.1)), 0.5, 2.0)),
        fecundity=int(_clamp(int(g.fecundity + round(rng.normal(0, 0.3))), 1, 3)),
        mutation_rate=float(_clamp(g.mutation_rate * (1 + rng.normal(0, 0.1)), 0.02, 0.2)),
    )


def init_weights(genome: Genome, in_size: int, rng: np.random.Generator,
                 scale: float = 0.15) -> List[torch.Tensor]:
    """每层 [out,in] + 读出 [10,last]，小随机（无学习部分为‘先天连接’）。"""
    sizes = [in_size] + list(genome.layer_sizes) + [10]
    ws: List[torch.Tensor] = []
    for i in range(len(sizes) - 1):
        w = (rng.normal(size=(sizes[i + 1], sizes[i])).astype(np.float32) * scale)
        ws.append(torch.from_numpy(w))
    return ws


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class Eco2Config:
    init_pop: int = 10000
    capacity: int = 20000
    max_rounds: int = 2000
    seed: int = 0
    # 能量经济
    e0: float = 100.0
    e_gain: float = 10.0
    e_cost: float = 8.0
    metabolism: float = 1.0
    repro_threshold: float = 200.0
    repro_cost: float = 100.0
    e_birth: float = 80.0
    age_max: Optional[int] = None
    # 学习
    T: int = 16
    tau_pre: float = 20.0
    tau_post: float = 20.0
    lr_base: float = 0.01
    w_min: float = -2.0
    w_max: float = 2.0
    learn_hidden: bool = False
    hidden_learn_factor: float = 0.5
    eligibility_decay: float = 0.95
    # 输入
    downsample_size: int = 14
    w_init_scale: float = 0.15
    mutation_noise: float = 0.05
    # 结束 / 采样
    goal: Optional[dict] = None
    on_extinction: str = "reseed"
    sample_every: int = 10
    # 奠基筛选
    founder_candidates: int = 20000
    founder_screen_digits: int = 4
    # 场景事件（Task 6 用）
    events: List["Event"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["events"] = [e.to_dict() for e in d["events"]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Eco2Config":
        from eco2_engine import Event
        d = dict(d)
        d["events"] = [Event.from_dict(e) for e in d.get("events", [])]
        return cls(**d)


# --------------------------------------------------------------------------- #
# 个体
# --------------------------------------------------------------------------- #
@dataclass
class Organism:
    uid: int
    genome: Genome
    energy: float = 100.0
    age: int = 0
    alive: bool = True
    weights: Optional[List[torch.Tensor]] = None
    correct: bool = False
    prediction: int = -1
    digit_counts: np.ndarray = field(default_factory=lambda: np.zeros(10, np.int32))

    def to_dict(self) -> dict:
        return {
            "uid": self.uid, "age": self.age, "energy": round(float(self.energy), 1),
            "alive": self.alive, "correct": self.correct,
            "prediction": self.prediction,
            "genome": self.genome.to_dict(),
            "digit_counts": self.digit_counts.tolist(),
        }


# --------------------------------------------------------------------------- #
# 批量前向（LIF + WTA + 读出）与预测
# --------------------------------------------------------------------------- #
THETA_BASE = 12.0  # 隐藏层/读出层阈值基数（× threshold_scale）


def poisson_encode(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    r = torch.from_numpy(rng.random(x.shape, dtype=np.float32)).to(x.device)
    return r.le(x).to(x)


def _lif(x: torch.Tensor, v: torch.Tensor, leak: torch.Tensor, threshold: torch.Tensor):
    """批量 LIF（软重置）。leak/threshold 为 [N,1]（每行不同基因可广播）。

    说明：spikingjelly 的 functional.lif_step 要求标量 tau/阈值，无法按个体基因
    批量，故手写（与旧 eco_engine 动力学一致：v=v·leak+x，v≥thr 发放，软重置）。
    """
    v = v * leak + x
    spike = (v >= threshold).float()
    v = v - spike * threshold
    return spike, v


def _wta(spikes: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """精确每行 top-k（含并列按索引序），其余置 0。k: [N] long。"""
    N, H = spikes.shape
    _, idx = torch.sort(spikes, dim=1, descending=True)
    pos = torch.arange(H, device=spikes.device).unsqueeze(0).expand(N, H)
    keep = pos < k.unsqueeze(1)
    out = torch.zeros_like(spikes)
    # 先 gather 成排序序空间再掩码，否则 keep 的列索引与 spikes 的列索引错位
    out.scatter_(1, idx, spikes.gather(1, idx) * keep.float())
    return out


def forward_group(spike_seq: torch.Tensor, weights: List[torch.Tensor],
                  genomes: List[Genome], cfg: Eco2Config):
    """批量前向 T 步，返回 (out_sum [N,10], elig_acc 每层 [N,out,in])。

    weights 每层为 [N, out, in]（已按种群 stack）；genomes 同 layer_sizes 分组内
    每个体（提供每行不同的 leak/threshold_scale/input_gain/wta_k）。eligibility
    在 T 步内以 eligibility_decay 累计（迹衰减），迹更新由 mstdp_linear_step 内部完成。
    """
    T, N, in_size = spike_seq.shape
    n_layers = len(weights)
    sizes = [in_size] + list(genomes[0].layer_sizes) + [10]
    dev = spike_seq.device
    leak = torch.tensor([g.leak for g in genomes], dtype=torch.float32, device=dev).unsqueeze(1)
    thr = (torch.tensor([g.threshold_scale for g in genomes], dtype=torch.float32,
                        device=dev) * THETA_BASE).unsqueeze(1)
    gain = torch.tensor([g.input_gain for g in genomes], dtype=torch.float32,
                        device=dev).unsqueeze(1)
    wta_k = torch.tensor([g.wta_k for g in genomes], dtype=torch.long, device=dev)

    vs = [torch.zeros(N, sizes[i + 1], device=dev) for i in range(n_layers)]
    elig_acc = [torch.zeros(N, sizes[i + 1], sizes[i], device=dev)
                for i in range(n_layers)]
    tr_pre = [torch.zeros(N, sizes[i], device=dev) for i in range(n_layers)]
    tr_post = [torch.zeros(N, sizes[i + 1], device=dev) for i in range(n_layers)]
    out_sum = torch.zeros(N, 10, device=dev)

    for t in range(T):
        x = spike_seq[t] * gain  # [N, in]
        for L in range(n_layers):
            x_in = torch.einsum("bi,noi->no", x, weights[L])  # [N, out]
            spike, vs[L] = _lif(x_in, vs[L], leak, thr)
            if L < n_layers - 1:  # 隐层 WTA（读出层不用 WTA）
                spike = _wta(spike, wta_k)
            # mstdp_linear_step 内部更新 trace 并返回每步 eligibility
            elig_t, (tr_pre[L], tr_post[L]) = sjf.learning.mstdp_linear_step(
                x, spike, (tr_pre[L], tr_post[L]), weights[L],
                tau_pre=cfg.tau_pre, tau_post=cfg.tau_post,
            )
            elig_acc[L] = elig_acc[L] * cfg.eligibility_decay + elig_t
            x = spike  # 下一层输入
        out_sum = out_sum + x  # x = 读出层脉冲
    return out_sum, elig_acc


# --------------------------------------------------------------------------- #
# 学习 / 能量 / 死亡（Task 3）
# --------------------------------------------------------------------------- #
def apply_mstdp(weights: List[torch.Tensor], elig_acc: List[torch.Tensor],
                reward: torch.Tensor, cfg: Eco2Config) -> None:
    """原位 mSTDP 更新。reward: [N]（+1 对 / -1 错）。读出层为主，隐层可选弱更新。"""
    n = len(weights)
    for L in range(n):
        factor = 1.0
        if L < n - 1:  # 隐层
            if not cfg.learn_hidden:
                continue
            factor = cfg.hidden_learn_factor
        lr = cfg.lr_base * factor
        w = weights[L]
        w += lr * reward[:, None, None] * elig_acc[L]
        w.clamp_(cfg.w_min, cfg.w_max)
        if torch.isnan(w).any():
            w.nan_to_num_(0.0)


def settle_energy(energy: torch.Tensor, correct: torch.Tensor, cfg: Eco2Config) -> torch.Tensor:
    return energy + torch.where(correct, cfg.e_gain, -cfg.e_cost) - cfg.metabolism


def mark_deaths(energy: torch.Tensor, age: torch.Tensor, cfg: Eco2Config) -> torch.Tensor:
    starved = energy <= 0.0
    if cfg.age_max is not None:
        starved = starved | (age >= cfg.age_max)
    return starved


# --------------------------------------------------------------------------- #
# 繁殖 + 拉马克混合遗传（Task 4）
# --------------------------------------------------------------------------- #
def lamarckism_blend(parent_w: List[torch.Tensor], genome: Genome, rng,
                     cfg: Eco2Config) -> List[torch.Tensor]:
    """拉马克混合：w_child = lamarckism×w_parent + (1-lamarckism)×w_random + 噪声"""
    lam = float(genome.lamarckism)
    child: List[torch.Tensor] = []
    for w in parent_w:
        w_random = torch.from_numpy(
            rng.normal(size=w.shape).astype(np.float32) * cfg.w_init_scale
        ).to(w.device)
        noise = torch.randn_like(w) * cfg.mutation_noise
        w_child = lam * w + (1.0 - lam) * w_random + noise
        w_child.clamp_(cfg.w_min, cfg.w_max)
        child.append(w_child)
    return child


def reproduce(parents: List[Organism], rng: np.random.Generator,
              cfg: Eco2Config, uid_counter: List[int], in_size: int) -> List[Organism]:
    """亲代能量已 >= repro_threshold 才被调用。每个亲代付代价，产 fecundity 个子代。

    结构未变 → 拉马克继承（lamarckism_blend）；结构变异 → 权重重启（随机先天连接）。
    """
    children: List[Organism] = []
    for p in parents:
        p.energy -= cfg.repro_cost
        for _ in range(p.genome.fecundity):
            g_child = mutate_genome(p.genome, rng)
            if g_child.layer_sizes == p.genome.layer_sizes:
                w_child = lamarckism_blend(p.weights, p.genome, rng, cfg)
            else:
                w_child = init_weights(g_child, in_size, rng, scale=cfg.w_init_scale)
            children.append(Organism(
                uid=uid_counter[0], genome=g_child, energy=cfg.e_birth, age=0,
                alive=True, weights=w_child,
            ))
            uid_counter[0] += 1
    return children
