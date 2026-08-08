"""临时：计时单次完整 evaluate（400 训练 + 400 校准 + 1000 验证）。"""
import time
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

g = seed_genome(n_neurons=100, seed=0)
g.train_samples = 400
t0 = time.time()
fit, m = evaluate(g, tr_img, tr_lbl, val_img, val_lbl, device)
dt = time.time() - t0
print(f"单次评估: {dt:.1f}s  fit={fit:.3f} acc={m['accuracy']:.3f}")
print(f"50×50×repeats2: {dt*50*50*2/3600:.1f}h  25×25×rep2: {dt*25*25*2/3600:.1f}h  "
      f"30×20×rep2: {dt*30*20*2/3600:.1f}h")
