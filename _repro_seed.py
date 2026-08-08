"""临时：直接用 evaluate() 复现种子 genome 的准确率，排查冒烟中种子 acc=14% 的异常。"""
import torch

from config import EvolutionConfig
from data_loading import load_mnist, split_pools
from evaluate import evaluate
from genome import seed_genome

cfg = EvolutionConfig()
train_img, train_lbl, _, _ = load_mnist()
pool, val = split_pools(train_img, train_lbl, val_size=cfg.val_size, seed=cfg.seed)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
device = torch.device("cpu")

for ts in (150, 400):
    g = seed_genome(n_neurons=100, seed=cfg.seed)
    g.train_samples = ts
    fit, m = evaluate(g, tr_img, tr_lbl, val_img, val_lbl, device)
    print(f"seed train_samples={ts}: fit={fit:.3f} acc={m['accuracy']:.3f} "
          f"sparse={m['sparsity']:.3f}")
