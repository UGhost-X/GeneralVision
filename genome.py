"""基因组：描述一个神经组（个体）的架构与超参。

基因组只含架构/超参（离散+连续），不含突触权重——权重在每个个体"一生"中
由 STDP 从初始化分布学习，生命周期结束即丢弃，不遗传。进化压力只作用在架构层。

LayerSpec 直接复用 snn.LayerConfig（含 n_out/leak/theta/w_norm/input_gain 等）。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from snn import LayerConfig, SNNParams

# 调通的单层基线（见 git 记录：JS 忠实参数退化，此为调参后的可用基线）
BASE_LAYER = dict(w_norm=16.0, theta_init=25.0, theta_clamp=(5, 100), a_plus=0.8)
DEFAULT_TRAIN_SAMPLES = 400
DEFAULT_T = 200
DEFAULT_SPIKE_GAIN = 0.6


@dataclass
class Genome:
    layers: list[LayerConfig]
    spike_gain: float = DEFAULT_SPIKE_GAIN
    T: int = DEFAULT_T
    train_samples: int = DEFAULT_TRAIN_SAMPLES
    seed: int = 0
    name: str = ""          # 日志/谱系用

    # ---------- 构建 ----------
    def build_snn_params(self) -> SNNParams:
        return SNNParams(spike_gain=self.spike_gain, T=self.T, seed=self.seed)

    def total_neurons(self) -> int:
        return sum(l.n_out for l in self.layers)

    def total_synapses(self) -> int:
        n_in = SNNParams.input_size
        total = 0
        for l in self.layers:
            total += n_in * l.n_out
            n_in = l.n_out
        return total

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = [asdict(l) for l in self.layers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(
            layers=[LayerConfig(**ld) for ld in d["layers"]],
            spike_gain=d.get("spike_gain", DEFAULT_SPIKE_GAIN),
            T=d.get("T", DEFAULT_T),
            train_samples=d.get("train_samples", DEFAULT_TRAIN_SAMPLES),
            seed=d.get("seed", 0),
            name=d.get("name", ""),
        )

    def clone(self) -> "Genome":
        return copy.deepcopy(self)

    def describe(self) -> str:
        """一行人类可读描述。"""
        ls = ",".join(f"{l.n_out}@{l.input_gain:g}" for l in self.layers)
        return f"layers=[{ls}] T={self.T} train={self.train_samples}"


def seed_genome(n_neurons: int = 100, seed: int = 0) -> Genome:
    """初始/默认种子个体：单层 100 神经元 + 调通的基线超参。"""
    return Genome(
        layers=[LayerConfig(n_out=n_neurons, **BASE_LAYER, seed=seed)],
        spike_gain=DEFAULT_SPIKE_GAIN,
        T=DEFAULT_T,
        train_samples=DEFAULT_TRAIN_SAMPLES,
        seed=seed,
        name="seed",
    )


def _self_check() -> None:
    g = seed_genome(seed=7)
    d = g.to_dict()
    g2 = Genome.from_dict(d)
    assert g2.to_dict() == d, "round-trip mismatch"
    assert g2.total_neurons() == 100
    assert g2.total_synapses() == 784 * 100
    print("genome self-check OK:", g.describe())
    print("serialized layers:", g2.layers[0])


if __name__ == "__main__":
    _self_check()
