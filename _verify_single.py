"""临时：单层 LIF+STDP 忠实移植验证与基准测速。"""
import time

import torch

import data_loading as dl
from snn import LayerConfig, SNN, SNNParams, accuracy, pref_diversity


def main():
    train_img, train_lbl, test_img, test_lbl = dl.load_mnist()
    pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
    (tr_img, tr_lbl), (val_img, val_lbl) = pool, val

    n_train = 400      # JS 演示训练规模
    n_val = 1000
    tr_img, tr_lbl = tr_img[:n_train], tr_lbl[:n_train]
    val_img, val_lbl = val_img[:n_val], val_lbl[:n_val]

    device = torch.device("cpu")
    print("device:", device)

    cfg = LayerConfig(n_out=100, seed=1)
    params = SNNParams(seed=0)
    net = SNN([cfg], params, device)

    gen = torch.Generator(device=device).manual_seed(0)
    for i in range(n_train):
        px = torch.tensor(tr_img[i], device=device)
        net.train_sample(px, int(tr_lbl[i]), gen)

    pref = net.layers[0].pref
    print("pref diversity:", pref_diversity(pref))

    val_pt = torch.tensor(val_img, device=device)
    lbl_pt = torch.tensor(val_lbl, device=device)

    # 批量评估（冻结 theta）
    gen = torch.Generator(device=device).manual_seed(0)
    t0 = time.time()
    score = net.evaluate_batch(val_pt, gen)
    t_eval = time.time() - t0
    acc = accuracy(score, pref, lbl_pt)
    print(f"batch eval {n_val} samples: {t_eval:.2f}s, acc={acc:.3f}")

    # 顺序评估（忠实 JS：推理期跑 homeostasis）
    gen = torch.Generator(device=device).manual_seed(0)
    t0 = time.time()
    score = net.evaluate_sequential(val_pt, gen)
    t_eval = time.time() - t0
    acc = accuracy(score, pref, lbl_pt)
    print(f"sequential eval {n_val} samples: {t_eval:.2f}s, acc={acc:.3f}")


if __name__ == "__main__":
    main()
