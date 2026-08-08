"""临时：锁定稳健基线——beta、T、归一化响应标签。"""
import torch

import data_loading as dl
from snn import LayerConfig, SNN, SNNParams

train_img, train_lbl, _, _ = dl.load_mnist()
pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
device = torch.device("cpu")
n_train = 400
tr_img, tr_lbl = tr_img[:n_train], tr_lbl[:n_train]


def run(tag, cfg_kwargs, params_kwargs=None, n_train=n_train):
    cfg = LayerConfig(**cfg_kwargs)
    net = SNN([cfg], SNNParams(**(params_kwargs or {})), device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n_train):
        net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
    layer = net.layers[0]
    pref = layer.pref
    score = net.evaluate_batch(torch.tensor(val_img, device=device),
                               torch.Generator(device="cpu").manual_seed(0))
    lbl = torch.tensor(val_lbl)
    acc1 = (pref[score.argmax(1)] == lbl).float().mean().item()
    votes = torch.zeros(score.shape[0], 10)
    for j in range(100):
        votes[:, pref[j]] += score[:, j]
    acc2 = (votes.argmax(1) == lbl).float().mean().item()
    print(f"{tag:34s} argmax={acc1:.3f} sumvote={acc2:.3f}")
    return acc2


base = dict(w_norm=16.0, theta_init=25.0, theta_clamp=(5, 100))
print("beta (homeostasis 强度):")
for b in (400, 1000, 2000):
    run(f"beta={b}", dict(base, beta=b))
print("\nT (仿真步数):")
for T in (100, 150):
    run(f"T={T}", base, dict(T=T))
print("\n更高 theta clamp 上限:")
run("clamp[10,80]", dict(base, theta_clamp=(10, 80)))
run("clamp[15,60]", dict(base, theta_clamp=(15, 60)))
