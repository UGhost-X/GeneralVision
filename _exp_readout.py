"""临时：STDP 学特征 + 监督线性读出——是否稳定、准确率高。"""
import numpy as np
import torch

from snn import LayerConfig, SNN, SNNParams
import data_loading as dl

train_img, train_lbl, _, _ = dl.load_mnist()
pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
device = torch.device("cpu")
BASE = dict(w_norm=16.0, theta_init=25.0, theta_clamp=(5, 100))


def extract_features(net, px):
    """批量评估：返回 [B, n_last] 放电计数。"""
    return net.evaluate_batch(torch.tensor(px, device=device),
                              torch.Generator(device="cpu").manual_seed(0)).numpy()


def train_readout(Xtr, ytr, Xva, yva, ridge=1e-3):
    """岭回归读出：W = (X^T X + λI)^-1 X^T Y。"""
    Y = np.eye(10)[ytr]
    W = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    preds = (Xva @ W).argmax(1)
    return (preds == yva).mean()


def run(n_train):
    cfg = LayerConfig(n_out=100, **BASE, seed=0)
    net = SNN([cfg], SNNParams(seed=0), device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n_train):
        net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
    Xtr = extract_features(net, tr_img[:n_train])
    acc = train_readout(Xtr, tr_lbl[:n_train], extract_features(net, val_img), val_lbl)
    # 稳定性：换一个初始化种子再测
    cfg2 = LayerConfig(n_out=100, **BASE, seed=7)
    net2 = SNN([cfg2], SNNParams(seed=0), device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n_train):
        net2.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
    acc2 = train_readout(extract_features(net2, tr_img[:n_train]), tr_lbl[:n_train],
                         extract_features(net2, val_img), val_lbl)
    print(f"train={n_train}: seed0 acc={acc:.3f}  seed7 acc={acc2:.3f}")
    return acc, acc2


for n in (200, 400, 800):
    run(n)
