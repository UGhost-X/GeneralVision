"""临时：cProfile 剖析单样本训练。"""
import cProfile
import pstats
import torch

from snn import LayerConfig, SNN, SNNParams
import data_loading as dl

train_img, train_lbl, _, _ = dl.load_mnist()

cfg = LayerConfig(n_out=100, seed=1)
params = SNNParams(T=200, seed=0)
net = SNN([cfg], params, torch.device("cpu"))
gen = torch.Generator(device="cpu").manual_seed(0)

px = torch.tensor(train_img[0])
prof = cProfile.Profile()
prof.enable()
for i in range(20):
    net.train_sample(px, int(train_lbl[0]), gen)
prof.disable()
pstats.Stats(prof).sort_stats("tottime").print_stats(20)
