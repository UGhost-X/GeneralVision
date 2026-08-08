"""评估器：给定一个基因组，完成其"一生"并算适应度。

一生 = STDP 无监督训练（特征学习）→ 监督线性读出（岭回归，从脉冲计数分类）。
监督读出的选择源于实验：纯无监督标签分配（digCount→pref）不稳定（同一网络因
初始化/训练量从 20% 跳到 63%，非单调），会破坏进化信号；线性读出示稳定（69-71%，
跨种子 ±1%），保留 LIF+STDP 核心，仅把读出改为监督。

适应度 = 读出准确率 + w_sparse*稀疏奖励 - w_compact*紧凑惩罚。
稀疏奖励惩罚"过度发放"神经元（抑制退化）；紧凑惩罚鼓励更小网络。训练样本与
读取拟合样本都由 genome.seed 固定抽取，评估可复现。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from genome import Genome
from snn import SNN


def _pick_indices(n_pool: int, n_pick: int, seed: int) -> np.ndarray:
    """固定种子抽取 n_pick 个索引（用于一生训练与读出拟合）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_pool)
    return idx[: min(n_pick, n_pool)]


def sparse_bonus(spike_acc: np.ndarray) -> float:
    """稀疏奖励：1 - 过度发放神经元比例。

    spike_acc: [B, N] 各样本各神经元放电数。过度发放 = 总放电 > 2× 均值
    （抑制 WTA 里单个神经元霸占发放的退化模式）。
    """
    per_neuron = spike_acc.sum(axis=0)
    mean = per_neuron.mean()
    if mean <= 0:
        return 0.0
    hyperactive = (per_neuron > 2.0 * mean).mean()
    return float(1.0 - hyperactive)


def _ridge_readout(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
                   ridge: float = 1e-2) -> np.ndarray:
    """岭回归读出：W = (X^T X + λI)^-1 X^T Y，返回验证集预测。"""
    Y = np.eye(10)[ytr]
    XtX = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    W = np.linalg.solve(XtX, Xtr.T @ Y)
    return (Xva @ W).argmax(1)


def evaluate(genome: Genome, train_img: np.ndarray, train_lbl: np.ndarray,
             val_img: np.ndarray, val_lbl: np.ndarray, device: torch.device,
             w_sparse: float = 0.05, w_compact: float = 0.01) -> tuple[float, dict]:
    """一生：STDP 训练 → 特征提取 → 线性读出 → (fitness, metrics)。"""
    t0 = time.time()
    net = SNN([_copy_cfg(l) for l in genome.layers],
              genome.build_snn_params(), device)

    idx = _pick_indices(len(train_img), genome.train_samples, genome.seed)
    tr_x = train_img[idx]
    tr_y = train_lbl[idx]

    # 一生：STDP 无监督训练
    gen = torch.Generator(device=device).manual_seed(genome.seed)
    for i in range(len(tr_x)):
        net.train_sample(torch.tensor(tr_x[i], device=device), int(tr_y[i]), gen)

    # 特征提取：训练样本与验证样本的最后一层脉冲计数
    gen_tr = torch.Generator(device=device).manual_seed(genome.seed + 1)
    Xtr = net.evaluate_batch(torch.tensor(tr_x, device=device), gen_tr).numpy()
    gen_va = torch.Generator(device=device).manual_seed(genome.seed + 2)
    Xva = net.evaluate_batch(torch.tensor(val_img, device=device), gen_va).numpy()

    # 监督线性读出
    preds = _ridge_readout(Xtr, tr_y, Xva)
    acc = float((preds == val_lbl).mean())

    n_neurons = genome.total_neurons()
    sparsity = sparse_bonus(Xva)
    fitness = acc + w_sparse * sparsity - w_compact * (n_neurons / 100.0)

    metrics = {
        "accuracy": acc,
        "sparsity": sparsity,
        "neurons": n_neurons,
        "layers": len(genome.layers),
        "time": time.time() - t0,
    }
    return float(fitness), metrics


def _copy_cfg(layer) -> object:
    import copy
    return copy.deepcopy(layer)
