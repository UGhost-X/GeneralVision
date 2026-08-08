"""临时：神谕标签诊断 + 大训练集。"""
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
    lbl = torch.tensor(val_lbl)
    # 用 pref 读出
    acc_pref = (pref[score.argmax(1)] == lbl).float().mean().item()
    # 神谕：每神经元用验证集反推真实偏好
    oracle_pref = torch.zeros(100, dtype=torch.long)
    for j in range(100):
        per = torch.zeros(10)
        for d in range(10):
            per[d] = score[lbl == d, j].sum()
        oracle_pref[j] = per.argmax()
    acc_oracle = (oracle_pref[score.argmax(1)] == lbl).float().mean().item()
    print(f"{tag:40s} acc_pref={acc_pref:.3f}  acc_oracle={acc_oracle:.3f}")
    return acc_oracle


base = dict(w_norm=16.0, theta_init=25.0, theta_clamp=(5, 100))
for n in (400, 1500, 4000):
    run(f"n_train={n} w16/th25", n, base)
