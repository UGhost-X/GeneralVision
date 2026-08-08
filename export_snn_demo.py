"""导出 SNN 推理演示数据：预训练权重 + 神经元偏好 + 演示样本图。

用项目真实 snn（LIF+WTA+STDP）训练单层 784→100 网络，验证 group-by-pref
读出准确率后，把权重/阈值/偏好/样本图以 base64 写入 snn_weights.js，
供 snn_demo.html 用 <script src> 加载（file:// 双击可用）。

用法:
    python export_snn_demo.py            # 默认 baseline 参数，2000 样本
    python export_snn_demo.py --variant classic --train 4000
    python export_snn_demo.py --no-export   # 只训练+评估，不写文件
"""
from __future__ import annotations

import argparse
import base64
import os

import numpy as np
import torch

from data_loading import load_mnist
from genome import BASE_LAYER
from snn import LayerConfig, SNN, SNNParams, accuracy_votes, pref_diversity

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "snn_weights.js")
SEED = 0
N_IMGS_PER_DIGIT = 5


def _pack_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def _pack_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<u1").tobytes()).decode()


def _train_layer(tr_x: np.ndarray, tr_y: np.ndarray, variant: str,
                 device: torch.device, cal_n: int = 2000) -> SNN:
    """训练单层网络（在线 STDP + 冻结权重校准标签分配），返回已训练 SNN。"""
    layer = (LayerConfig(n_out=100, seed=SEED, w_norm=78.4, theta_init=15.0,
                         theta_clamp=(5, 100))
             if variant == "classic"
             else LayerConfig(n_out=100, seed=SEED, **BASE_LAYER))
    params = SNNParams(spike_gain=0.6, T=200, num_classes=10,
                       input_size=784, seed=SEED)
    net = SNN([layer], params, device)

    gen = torch.Generator(device=device).manual_seed(SEED)
    for i in range(len(tr_x)):                       # 一生：在线 STDP 训练
        net.train_sample(torch.tensor(tr_x[i], device=device), int(tr_y[i]), gen)
    # 训练期 dig_count 含权重演化早期噪声，清零后仅用冻结权重校准标签
    net.layers[0].dig_count.zero_()
    cal = np.random.default_rng(SEED + 1).permutation(len(tr_x))[:cal_n]
    net.calibrate(torch.tensor(tr_x[cal], device=device),
                  torch.tensor(tr_y[cal], device=device), gen)
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--calibrate", type=int, default=2000)
    ap.add_argument("--variant", choices=["baseline", "classic"], default="baseline")
    ap.add_argument("--no-export", action="store_true")   # 只评估，不写文件
    args = ap.parse_args()

    device = torch.device("cpu")
    train_img, train_lbl, test_img, test_lbl = load_mnist()

    # 训练样本：固定种子抽取
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_img))
    tr_x = train_img[idx[: args.train]]
    tr_y = train_lbl[idx[: args.train]]

    net = _train_layer(tr_x, tr_y, args.variant, device, cal_n=args.calibrate)
    layer = net.layers[0]

    # 评估 group-by-pref 读出（演示页用的就是这种读出）
    va_x, va_y = test_img[:1000], test_lbl[:1000]
    gen_va = torch.Generator(device=device).manual_seed(SEED + 3)
    score = net.evaluate_batch(torch.tensor(va_x, device=device), gen_va)
    pref = layer.pref
    acc = accuracy_votes(score, pref, torch.tensor(va_y, device=device))
    div = pref_diversity(pref, 10)
    print(f"[{args.variant}] train={len(tr_x)} group-by-pref 准确率={acc:.3f} "
          f"pref 多样性={div}/10")
    if args.no_export:
        return

    # 演示样本图：每数字取前 N_IMGS_PER_DIGIT 张官方测试图（uint8 0-255）
    imgs, lbls = [], []
    for d in range(10):
        d_idx = np.where(test_lbl == d)[0][:N_IMGS_PER_DIGIT]
        imgs.append((test_img[d_idx] * 255).round().astype(np.uint8))
        lbls.append(np.full(len(d_idx), d, dtype=np.uint8))
    imgs = np.concatenate(imgs).reshape(-1)          # [50*784]
    lbls = np.concatenate(lbls).tolist()

    W = layer.W.detach().cpu().numpy()               # [784, 100]，列主序 W[i,j]
    theta = layer.theta.detach().cpu().numpy()       # [100]
    js = (
        "// 由 export_snn_demo.py 生成，勿手改（重跑脚本再生）\n"
        f"const SNN_W_B64 = \"{_pack_f32(W)}\";\n"
        f"const SNN_THETA_B64 = \"{_pack_f32(theta)}\";\n"
        f"const SNN_PREF = {pref.cpu().tolist()};\n"
        f"const SNN_IMGS_B64 = \"{_pack_u8(imgs)}\";\n"
        f"const SNN_IMGS_LBL = {lbls};\n"
        f"const SNN_IMGS_PER_DIGIT = {N_IMGS_PER_DIGIT};\n"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(js)

    # 自检：base64 往返一致
    dec = np.frombuffer(base64.b64decode(_pack_f32(W)), dtype="<f4").reshape(784, 100)
    assert np.array_equal(dec, W), "权重往返不一致"
    dec_i = np.frombuffer(base64.b64decode(_pack_u8(imgs)), dtype="<u1")
    assert np.array_equal(dec_i, imgs), "图像往返不一致"
    print(f"已导出 {OUT} ({os.path.getsize(OUT) // 1024}KB)  准确率={acc:.3f}")


if __name__ == "__main__":
    main()
