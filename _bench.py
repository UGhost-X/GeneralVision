"""临时：对比 CPU/GPU 训练速度，找出瓶颈。"""
import time

import torch

from snn import LayerConfig, SNN, SNNParams
import data_loading as dl

train_img, train_lbl, _, _ = dl.load_mnist()


def bench(device_name, n=50, T=200):
    device = torch.device(device_name)
    cfg = LayerConfig(n_out=100, seed=1)
    params = SNNParams(T=T, seed=0)
    net = SNN([cfg], params, device)
    gen = torch.Generator(device=device).manual_seed(0)
    t0 = time.time()
    for i in range(n):
        px = torch.tensor(train_img[i], device=device)
        net.train_sample(px, int(train_lbl[i]), gen)
    dt = time.time() - t0
    print(f"{device_name} T={T}: {n} samples in {dt:.2f}s = {dt/n*1000:.1f} ms/sample, "
          f"{dt/n/T*1000:.2f} ms/step")


for d in ["cuda", "cpu"]:
    bench(d, n=50, T=200)
