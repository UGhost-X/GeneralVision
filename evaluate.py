"""评估器：给定一个基因组，完成其"一生"（STDP 训练 → 校准标签 → 评估）并算适应度。

适应度 = 投票读出准确率 + w_sparse*稀疏奖励 - w_compact*紧凑惩罚。
稀疏奖励惩罚"过度发放"神经元（抑制退化）；紧凑惩罚鼓励更小的网络。权重默认很小，
保证准确率主导。训练样本由 genome.seed 固定抽取，评估可复现。
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from genome import Genome
from snn import SNN, accuracy_votes, pref_diversity


def _pick_train_samples(genome: Genome, n_pool: int) -> np.ndarray:
    """按 genome.seed 固定抽取一生训练样本的索引（从 train pool 中）。"""
    rng = np.random.default_rng(genome.seed)
    idx = rng.permutation(n_pool)
    n = min(genome.train_samples, n_pool)
    return idx[:n]


def sparse_bonus(spike_acc: torch.Tensor) -> float:
    """稀疏奖励：1 - 过度发放神经元比例。

    spike_acc: [B, N] 各样本各神经元放电数。过度发放 = 总放电 > 2× 均值
    （抑制 WTA 里单个神经元霸占发放的退化模式）。
    """
    per_neuron = spike_acc.sum(dim=0)           # [N]
    mean = per_neuron.mean()
    if mean <= 0:
        return 0.0
    hyperactive = (per_neuron > 2.0 * mean).float().mean().item()
    return float(1.0 - hyperactive)


def evaluate(genome: Genome, train_img: np.ndarray, train_lbl: np.ndarray,
             val_img: np.ndarray, val_lbl: np.ndarray, device: torch.device,
             w_sparse: float = 0.05, w_compact: float = 0.01) -> tuple[float, dict]:
    """一生：训练→校准→评估。返回 (fitness, metrics)。"""
    t0 = time.time()
    net = SNN([copy.deepcopy(l) for l in genome.layers],
              genome.build_snn_params(), device)

    idx = _pick_train_samples(genome, len(train_img))
    tr_x = train_img[idx]
    tr_y = train_lbl[idx]

    # 一生训练（STDP 在线学习）
    gen = torch.Generator(device=device).manual_seed(genome.seed)
    for i in range(len(tr_x)):
        net.train_sample(torch.tensor(tr_x[i], device=device), int(tr_y[i]), gen)

    # 校准标签分配：冻结权重重跑训练样本，重算 dig_count
    last_layer = net.layers[-1]
    last_layer.dig_count.zero_()
    net.calibrate(torch.tensor(tr_x, device=device), torch.tensor(tr_y, device=device),
                  torch.Generator(device=device).manual_seed(genome.seed + 1))

    pref = last_layer.pref
    val_t = torch.tensor(val_img, device=device)
    score = net.evaluate_batch(val_t, torch.Generator(device=device).manual_seed(genome.seed))
    acc = accuracy_votes(score, pref, torch.tensor(val_lbl, device=device))

    n_neurons = genome.total_neurons()
    sparsity = sparse_bonus(score)
    fitness = acc + w_sparse * sparsity - w_compact * (n_neurons / 100.0)

    metrics = {
        "accuracy": acc,
        "sparsity": sparsity,
        "neurons": n_neurons,
        "layers": len(genome.layers),
        "pref_diversity": pref_diversity(pref),
        "time": time.time() - t0,
    }
    return float(fitness), metrics
