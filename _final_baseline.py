"""临时：最终基线验证 + 2 层冒烟测试。"""
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


def evaluate(net):
    layer = net.layers[-1]
    pref = layer.pref
    score = net.evaluate_batch(torch.tensor(val_img, device=device),
                               torch.Generator(device="cpu").manual_seed(0))
    return accuracy_votes(score, pref, torch.tensor(val_lbl))


def train_calibrate(net, n=n_train):
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n):
        net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
    layer = net.layers[-1]
    layer.dig_count.zero_()
    net.calibrate(torch.tensor(tr_img[:n]), torch.tensor(tr_lbl[:n]),
                  torch.Generator(device="cpu").manual_seed(1))


# 单层基线
net = SNN([LayerConfig(n_out=100, **BASE, seed=1)], SNNParams(seed=0), device)
train_calibrate(net)
print(f"单层基线 acc={evaluate(net):.3f}  pref_diversity={pref_diversity(net.layers[0].pref)}")

# 2 层冒烟：784→60→40
net2 = SNN([LayerConfig(n_out=60, **BASE, seed=1),
            LayerConfig(n_out=40, **BASE, seed=2)], SNNParams(seed=0), device)
train_calibrate(net2, n=100)
print(f"2层冒烟(100样本) acc={evaluate(net2):.3f}  pref_diversity={pref_diversity(net2.layers[-1].pref)}")
