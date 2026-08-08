"""临时：训练/校准样本量 vs 准确率——找最便宜仍有效的量。"""
import time
import torch

from config import EvolutionConfig
from data_loading import load_mnist, split_pools
from evaluate import evaluate
from genome import seed_genome

cfg = EvolutionConfig()
train_img, train_lbl, _, _ = load_mnist()
pool, val = split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
device = torch.device("cpu")


def run(train_n):
    g = seed_genome(n_neurons=100, seed=0)
    g.train_samples = train_n
    t0 = time.time()
    fit, m = evaluate(g, tr_img, tr_lbl, val_img, val_lbl, device)
    dt = time.time() - t0
    print(f"train={train_n}: acc={m['accuracy']:.3f} 耗时={dt:.1f}s")
    return m['accuracy']


for n in (100, 150, 200, 300, 400):
    run(n)
