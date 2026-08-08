"""临时：诊断单层移植——theta 分布、评估发放统计、权重-数字相关性。"""
import numpy as np
import torch

import data_loading as dl
from snn import LayerConfig, SNN, SNNParams

train_img, train_lbl, _, _ = dl.load_mnist()
pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val

n_train, n_val = 400, 1000
tr_img, tr_lbl = tr_img[:n_train], tr_lbl[:n_train]
val_img, val_lbl = val_img[:n_val], val_lbl[:n_val]

device = torch.device("cpu")
cfg = LayerConfig(n_out=100, seed=1)
net = SNN([cfg], SNNParams(seed=0), device)
layer = net.layers[0]

gen = torch.Generator(device="cpu").manual_seed(0)
for i in range(n_train):
    net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)

print("theta:  min=%.2f mean=%.2f max=%.2f" % (layer.theta.min(), layer.theta.mean(), layer.theta.max()))
print("rate:   min=%.4f mean=%.4f max=%.4f" % (layer.rate.min(), layer.rate.mean(), layer.rate.max()))
print("dig_count sum per neuron:", layer.dig_count.sum(1)[:10].tolist())
pref = layer.pref
print("pref counts:", np.bincount(pref.numpy(), minlength=10).tolist())

# 评估时发放统计
score = net.evaluate_batch(torch.tensor(val_img, device=device), gen)
sc = score.numpy()
print("eval spike_acc per sample: min=%.1f mean=%.2f max=%.1f, zero-firing samples: %d/%d"
      % (sc.sum(1).min(), sc.sum(1).mean(), sc.sum(1).max(), (sc.sum(1) == 0).sum(), n_val))
print("per-neuron total firing over eval:", (sc > 0).sum(0).tolist()[:20])

# 权重-数字相关性：每个神经元学到的权重模式 vs 各数字平均图
W = layer.W.detach().numpy()  # [784, 100]
avg_digit = np.stack([tr_img[tr_lbl == d].mean(0) for d in range(10)])  # [10, 784]
corr = np.zeros((100, 10))
for j in range(100):
    w = W[:, j]
    for d in range(10):
        ww = w - w.mean(); dd = avg_digit[d] - avg_digit[d].mean()
        denom = (np.linalg.norm(ww) * np.linalg.norm(dd))
        corr[j, d] = np.dot(ww, dd) / denom if denom > 0 else 0
best_digit = corr.argmax(1)
agree = (best_digit == pref.numpy()).mean()
print("weight-pattern best digit vs pref agreement: %.2f" % agree)
print("corr best value mean: %.3f (max=%.3f)" % (corr.max(1).mean(), corr.max()))
