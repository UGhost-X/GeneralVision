"""MNIST 数据加载与固定划分。

从 data/MNIST/raw/ 读取原始 ubyte 文件（已解压），不依赖 torchvision 的下载逻辑。
返回归一化到 [0,1] 的 float32 numpy 数组与 int64 标签。

划分固定、可复现：train pool 与 val pool 由 seed 确定，供进化评估器使用。
"""
from __future__ import annotations

import os
import struct

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MNIST", "raw")


def _read_idx(path: str) -> np.ndarray:
    """读取 IDX 格式文件（MNIST ubyte）。"""
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        dims = []
        ndim = magic & 0xFF
        for _ in range(ndim):
            dims.append(struct.unpack(">I", f.read(4))[0])
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
    return data


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """加载 train 或 test 的 (images, labels)。"""
    images = _read_idx(os.path.join(ROOT, f"{name}-images-idx3-ubyte"))
    labels = _read_idx(os.path.join(ROOT, f"{name}-labels-idx1-ubyte"))
    return images, labels


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (train_images, train_labels, test_images, test_labels)。

    images 形状 (N, 784) float32 ∈ [0,1]；labels 形状 (N,) int64 ∈ {0..9}。
    """
    train_images, train_labels = _load_split("train")
    test_images, test_labels = _load_split("t10k")

    # 拉平并归一化到 [0,1]
    train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0
    train_labels = train_labels.astype(np.int64)
    test_labels = test_labels.astype(np.int64)
    return train_images, train_labels, test_images, test_labels


def split_pools(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_size: int = 1000,
    seed: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """把官方训练集按固定 seed 划分为 train pool 与 val pool。

    返回 ((train_imgs, train_lbls), (val_imgs, val_lbls))。
    保证对相同 seed 划分完全一致，供进化使用（不同个体用相同 val 评价）。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_images))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return (
        (train_images[train_idx], train_labels[train_idx]),
        (train_images[val_idx], train_labels[val_idx]),
    )


def _self_check() -> None:
    """命令行自检：打印各集合形状与像素范围。"""
    ti, tl, te, tel = load_mnist()
    print(f"train images:  {ti.shape}  dtype={ti.dtype}  range=[{ti.min():.3f}, {ti.max():.3f}]")
    print(f"train labels:  {tl.shape}  unique={np.unique(tl)}")
    print(f"test images:   {te.shape}  dtype={te.dtype}")
    print(f"test labels:   {tel.shape}  unique={np.unique(tel)}")

    pool, val = split_pools(ti, tl, val_size=1000, seed=0)
    print(f"train pool:    {pool[0].shape}")
    print(f"val pool:      {val[0].shape}")
    assert ti.shape == (60000, 784), f"unexpected train shape {ti.shape}"
    assert te.shape == (10000, 784), f"unexpected test shape {te.shape}"
    assert tl.min() == 0 and tl.max() == 9
    assert pool[0].shape[0] == 59000 and val[0].shape[0] == 1000
    print("self-check OK")


if __name__ == "__main__":
    _self_check()
