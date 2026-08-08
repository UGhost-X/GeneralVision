"""临时：网格实验——哪组配置能让单层 STDP 学到数字（目标 >50%）。"""
import torch

import data_loading as dl
from snn import LayerConfig, SNN, SNNParams

train_img, train_lbl, _, _ = dl.load_mnist()
pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
n_train = 400
tr_img, tr_lbl = tr_img[:n_train], tr_lbl[:n_train]
device = torch.device("cpu")


def run(tag, cfg_kwargs, n_train=n_train):
    cfg = LayerConfig(**cfg_kwargs)
    net = SNN([cfg], SNNParams(seed=0), device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n_train):
        net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
    layer = net.layers[0]
    pref = layer.pref
    score = net.evaluate_batch(torch.tensor(val_img, device=device),
                               torch.Generator(device="cpu").manual_seed(0))
    # argmax 读出
    acc1 = (pref[score.argmax(1)] == torch.tensor(val_lbl)).float().mean().item()
    # 投票求和读出
    votes = torch.zeros(score.shape[0], 10)
    for j in range(100):
        votes[:, pref[j]] += score[:, j]
    acc2 = (votes.argmax(1) == torch.tensor(val_lbl)).float().mean().item()
    # pref 与真实选择一致率
    best_digit = torch.zeros(100, dtype=torch.long)
    for j in range(100):
        per = torch.zeros(10)
        for d in range(10):
            per[d] = score[torch.tensor(val_lbl) == d, j].sum()
        best_digit[j] = per.argmax()
    agree = (best_digit == pref).float().mean().item()
    print(f"{tag:34s} theta[{layer.theta.min():.0f},{layer.theta.max():.0f}] "
          f"argmax={acc1:.3f} sumvote={acc2:.3f} agree={agree:.3f}")
    return acc2


print("baseline (faithful):")
run("faithful", {})
print("\ntheta clamp during training:")
run("clamp[5,100]", dict(theta_clamp=(5, 100)))
run("clamp[10,60]", dict(theta_clamp=(10, 60)))
run("clamp[15,40]", dict(theta_clamp=(15, 40)))
print("\nsparser firing (higher theta, weaker input):")
run("theta0=30", dict(theta_init=30.0, theta_clamp=(5, 100)))
run("theta0=50", dict(theta_init=50.0, theta_clamp=(5, 100)))
print("\nsmaller weights (w_norm):")
run("w_norm=20", dict(w_norm=20.0, theta_clamp=(5, 100)))
run("w_norm=8", dict(w_norm=8.0, theta_clamp=(5, 100)))
print("\ncombos:")
run("theta0=30 w_norm=20 clamp", dict(theta_init=30.0, w_norm=20.0, theta_clamp=(5, 100)))
run("theta0=50 w_norm=20 clamp", dict(theta_init=50.0, w_norm=20.0, theta_clamp=(5, 100)))
