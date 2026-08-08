"""临时：两阶段实验——训练后无学习校准标签分配。"""
import torch

import data_loading as dl
from snn import LayerConfig, SNN, SNNParams

train_img, train_lbl, _, _ = dl.load_mnist()
pool, val = dl.split_pools(train_img, train_lbl, val_size=1000, seed=0)
(tr_img, tr_lbl), (val_img, val_lbl) = pool, val
device = torch.device("cpu")
n_train = 400
tr_img, tr_lbl = tr_img[:n_train], tr_lbl[:n_train]
cfg_kwargs = dict(w_norm=16.0, theta_init=25.0, theta_clamp=(5, 100))


def eval_acc(net):
    layer = net.layers[0]
    pref = layer.pref
    score = net.evaluate_batch(torch.tensor(val_img, device=device),
                               torch.Generator(device="cpu").manual_seed(0))
    lbl = torch.tensor(val_lbl)
    votes = torch.zeros(score.shape[0], 10)
    for j in range(100):
        votes[:, pref[j]] += score[:, j]
    return (votes.argmax(1) == lbl).float().mean().item()


cfg = LayerConfig(**cfg_kwargs)
net = SNN([cfg], SNNParams(seed=0), device)
gen = torch.Generator(device="cpu").manual_seed(0)
for i in range(n_train):
    net.train_sample(torch.tensor(tr_img[i]), int(tr_lbl[i]), gen)
print(f"训练后直接读 pref: acc={eval_acc(net):.3f}")

# 无学习校准：重算 dig_count（覆盖旧的）
net.layers[0].dig_count.zero_()
gen2 = torch.Generator(device="cpu").manual_seed(1)
net.calibrate(torch.tensor(tr_img[:n_train]), torch.tensor(tr_lbl[:n_train]), gen2)
print(f"校准后 pref:     acc={eval_acc(net):.3f}")
