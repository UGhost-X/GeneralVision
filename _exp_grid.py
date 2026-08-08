"""临时：细网格——围绕 theta0=30, w_norm=20 找甜点，并试更多训练样本。"""
import torch

import data_loading as dl
from snn import LayerConfig, SNN, SNNParams

train_img, train_lbl, _, _ = dl.load_mnist()
pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
device = torch.device("cpu")


def run(tag, n_train, cfg_kwargs):
    cfg = LayerConfig(**cfg_kwargs)
    net = SNN([cfg], SNNParams(seed=0), device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n_train):
        net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
    layer = net.layers[0]
    pref = layer.pref
    score = net.evaluate_batch(torch.tensor(val_img, device=device),
                               torch.Generator(device="cpu").manual_seed(0))
    acc1 = (pref[score.argmax(1)] == torch.tensor(val_lbl)).float().mean().item()
    votes = torch.zeros(score.shape[0], 10)
    for j in range(100):
        votes[:, pref[j]] += score[:, j]
    acc2 = (votes.argmax(1) == torch.tensor(val_lbl)).float().mean().item()
    best_digit = torch.zeros(100, dtype=torch.long)
    for j in range(100):
        per = torch.zeros(10)
        for d in range(10):
            per[d] = score[torch.tensor(val_lbl) == d, j].sum()
        best_digit[j] = per.argmax()
    agree = (best_digit == pref).float().mean().item()
    print(f"{tag:38s} theta[{layer.theta.min():.0f},{layer.theta.max():.0f}] "
          f"argmax={acc1:.3f} sumvote={acc2:.3f} agree={agree:.3f}")
    return acc2


base = dict(theta_clamp=(5, 100))
print("n_train=400:")
for wn in (12, 16, 20, 24):
    for th in (20, 25, 30):
        run(f"w_norm={wn} theta0={th}", 400, dict(base, w_norm=wn, theta_init=th))

print("\nn_train=800 (best few):")
for wn in (12, 16, 20):
    for th in (25, 30):
        run(f"w_norm={wn} theta0={th}", 800, dict(base, w_norm=wn, theta_init=th))
