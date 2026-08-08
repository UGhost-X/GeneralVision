"""临时：2 层——找第二层 input_gain，使深层神经元分化。"""
import torch

import data_loading as dl
from snn import (LayerConfig, SNN, SNNParams, accuracy_votes, pref_diversity)

train_img, train_lbl, _, _ = dl.load_mnist()
pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
device = torch.device("cpu")
n_train = 400
tr_img, tr_lbl = tr_img[:n_train], tr_lbl[:n_train]
BASE = dict(w_norm=16.0, theta_init=25.0, theta_clamp=(5, 100))


def run(gain2, n_train=n_train):
    net = SNN([LayerConfig(n_out=60, **BASE, seed=1),
               LayerConfig(n_out=40, **BASE, input_gain=gain2, seed=2)],
              SNNParams(seed=0), device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n_train):
        net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
    l2 = net.layers[-1]
    l2.dig_count.zero_()
    net.calibrate(torch.tensor(tr_img[:n_train]), torch.tensor(tr_lbl[:n_train]),
                  torch.Generator(device="cpu").manual_seed(1))
    pref = l2.pref
    div = pref_diversity(pref)
    score = net.evaluate_batch(torch.tensor(val_img, device=device),
                               torch.Generator(device="cpu").manual_seed(0))
    acc = accuracy_votes(score, pref, torch.tensor(val_lbl))
    print(f"gain2={gain2:6.1f}  div={div}  acc={acc:.3f}")
    return acc


for g in (1, 5, 10, 20, 40, 80):
    run(g)
