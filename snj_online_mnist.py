"""
snj_online_mnist.py — 基于 SpikingJelly 的 MNIST 在线学习原型

目标：搭建一个「在运行中就能更新权重」的类脑（脉冲神经）模型。
学习规则：OTTT（Online Training Through Time）在线梯度训练 —— 把一张图的脉冲序列一个时间步一个
时间步喂进去，每个时间步都执行 前向 → 算损失 → 反向 → 更新权重，没有独立的训练阶段；
网络一边前向推理，一边实时更新自己的突触权重。

用法示例：
    # 完整在线训练（约 1.5 万样本，逐步更新权重，每 2000 样本打印一次准确率）
    python snj_online_mnist.py --samples 15000

    # 快速验证（1000 样本，确认能跑通）
    python snj_online_mnist.py --quick

    # 自定义
    python snj_online_mnist.py --hidden 256 --T 16 --batch 16 --lr 1e-3 --eval-every 2000

环境要求：当前 venv（Python 3.10 + torch 2.6.0+cu124）；若 spikingjelly 未 pip 安装，
脚本会自动把同目录下的 spikingjelly/ 加入 sys.path（见文件底部）。
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.activation_based import encoding, functional, layer, neuron
    from spikingjelly.activation_based.layer.online_learning import OTTTSequential
except ImportError:
    # 未 pip install 时，回退到同目录克隆库。外层同名目录会被当作命名空间包缓存进
    # sys.modules（无 __init__.py），必须先移除缓存，再把「内层包的父目录」加进
    # sys.path，import spikingjelly 才能命中真正的包。
    _sj_parent = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "spikingjelly"
    )
    sys.path.insert(0, _sj_parent)
    for _m in list(sys.modules):
        if _m == "spikingjelly" or _m.startswith("spikingjelly."):
            sys.modules.pop(_m, None)
    from spikingjelly.activation_based import encoding, functional, layer, neuron
    from spikingjelly.activation_based.layer.online_learning import OTTTSequential


# --------------------------------------------------------------------------- #
# 数据：直接从 data/MNIST/raw 下的 idx3/idx1 二进制读取（不依赖 torchvision）
# --------------------------------------------------------------------------- #
def load_mnist_raw(root, split="train"):
    """读 MNIST 原始文件，返回 (imgs: float32 [N,28,28] 0~1, labs: uint8 [N])。"""
    if split == "train":
        img_path = os.path.join(root, "train-images-idx3-ubyte")
        lab_path = os.path.join(root, "train-labels-idx1-ubyte")
    else:
        img_path = os.path.join(root, "t10k-images-idx3-ubyte")
        lab_path = os.path.join(root, "t10k-labels-idx1-ubyte")

    with open(img_path, "rb") as f:
        buf = f.read()
    _, n, rows, cols = struct.unpack(">IIII", buf[:16])
    imgs = np.frombuffer(buf[16:], dtype=np.uint8).reshape(n, rows, cols)
    imgs = imgs.astype(np.float32) / 255.0

    with open(lab_path, "rb") as f:
        buf = f.read()
    _, n = struct.unpack(">II", buf[:8])
    labs = np.frombuffer(buf[8:], dtype=np.uint8)
    return imgs, labs


# --------------------------------------------------------------------------- #
# 网络：OTTTSequential（能处理 OTTTLIFNode 返回的 [spike, trace] 脉冲流）
# --------------------------------------------------------------------------- #
def build_net(hidden, seed=0):
    """两层 LIF 脉冲网络：784 → hidden → 10，全部用 OTTT 在线学习神经元。"""
    torch.manual_seed(seed)
    return OTTTSequential(
        layer.Linear(28 * 28, hidden, bias=False),
        neuron.OTTTLIFNode(tau=2.0),
        layer.Linear(hidden, 10, bias=False),
        neuron.OTTTLIFNode(tau=2.0),
    )


# --------------------------------------------------------------------------- #
# 评估：脉冲率读出（累计 T 步输出脉冲数，argmax）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_acc(net, enc, imgs, labs, T, batch, device, max_samples=1000, seed=0):
    """在样本上测准确率。返回 acc in [0,1]。"""
    net.eval()
    functional.reset_net(net)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(imgs), size=min(max_samples, len(imgs)), replace=False)
    correct = 0
    for i in range(0, len(idx), batch):
        bi = idx[i : i + batch]
        xb = torch.from_numpy(imgs[bi]).to(device)  # [B,28,28]
        spike_sum = None
        for _ in range(T):
            out = net(enc(xb).flatten(1))
            spike = out[0] if isinstance(out, list) else out
            spike = spike.detach()
            spike_sum = spike if spike_sum is None else spike_sum + spike
        pred = spike_sum.argmax(1).cpu().numpy()
        correct += int((pred == labs[bi]).sum())
    net.train()
    functional.reset_net(net)
    return correct / len(idx)


# --------------------------------------------------------------------------- #
# 主流程：在线训练（每个时间步即时更新权重）
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="SpikingJelly MNIST 在线学习原型（OTTT）")
    ap.add_argument("--data-dir", default="data/MNIST/raw",
                    help="MNIST raw 文件目录")
    ap.add_argument("--hidden", type=int, default=256, help="隐藏层神经元数")
    ap.add_argument("--T", type=int, default=16, help="每张图编码的脉冲时间步数")
    ap.add_argument("--batch", type=int, default=16, help="在线流式更新批次大小")
    ap.add_argument("--samples", type=int, default=15000, help="在线训练总样本数")
    ap.add_argument("--eval-every", type=int, default=2000, help="每多少样本测一次准确率")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    ap.add_argument("--clip", type=float, default=1.0, help="梯度裁剪范数（None=不裁剪）")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--eval-max", type=int, default=1000, help="每次评估用的样本数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="快速验证：1000 样本")
    args = ap.parse_args()
    if args.quick:
        args.samples = 1000
        args.eval_every = 500
        args.hidden = 128

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  hidden={args.hidden}  T={args.T}  "
          f"batch={args.batch}  lr={args.lr}  samples={args.samples}")

    # 数据
    tr_imgs, tr_labs = load_mnist_raw(args.data_dir, "train")
    te_imgs, te_labs = load_mnist_raw(args.data_dir, "test")
    print(f"[data] train={len(tr_imgs)}  test={len(te_imgs)}")

    # 网络 / 优化器 / 编码器
    net = build_net(args.hidden, args.seed).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    enc = encoding.PoissonEncoder()
    print(f"[net] 参数量={sum(p.numel() for p in net.parameters())}")

    # 初始准确率
    acc0 = eval_acc(net, enc, te_imgs, te_labs, args.T, args.batch, device,
                    max_samples=args.eval_max)
    print(f"[eval] initial acc = {acc0:.4f}")

    # ---- 在线训练：样本流式进入，每个时间步前向→反向→更新权重 ----
    curve = [(0, acc0)]
    t0 = time.time()
    for s in range(0, args.samples, args.batch):
        bi = np.random.choice(len(tr_imgs), size=args.batch, replace=False)
        xb = torch.from_numpy(tr_imgs[bi]).to(device)
        yb = torch.from_numpy(tr_labs[bi].astype(np.int64)).to(device)

        functional.reset_net(net)          # 只在每个样本批次前 reset
        for _ in range(args.T):            # 一个时间步一个时间步地在线学
            out = net(enc(xb).flatten(1))  # 现场泊松编码 + 前向
            loss = F.cross_entropy(out[0], yb)
            optim.zero_grad()
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optim.step()                   # ← 运行中实时更新权重

        n_done = s + args.batch
        if n_done % args.eval_every == 0 or n_done >= args.samples:
            acc = eval_acc(net, enc, te_imgs, te_labs, args.T, args.batch,
                           device, max_samples=args.eval_max)
            curve.append((n_done, acc))
            dt = time.time() - t0
            print(f"[train] {n_done:6d}/{args.samples}  loss={loss.item():.4f}  "
                  f"acc={acc:.4f}  ({args.batch / max(dt, 1e-9):.0f} 样本/秒)")

    # 学习曲线保存
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs, ys = zip(*curve)
        plt.plot(xs, ys, "o-")
        plt.xlabel("online samples")
        plt.ylabel("test accuracy")
        plt.title(f"MNIST online learning (OTTT, hidden={args.hidden}, T={args.T})")
        plt.grid(alpha=0.3)
        out_png = "snj_online_curve.png"
        plt.savefig(out_png, dpi=120)
        print(f"[done] learning curve -> {out_png}  |  final acc = {ys[-1]:.4f}")
    except Exception as e:
        print(f"[done] final acc = {curve[-1][1]:.4f}  (curve save skipped: {e})")


if __name__ == "__main__":
    main()
