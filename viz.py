"""进化结果可视化：从检查点历史绘制适应度曲线与最优架构演化。

用法：python viz.py [checkpoint_dir] [--out evolution_curves.png]
读取目录下最后一个 gen_*.json 的 history 字段绘图。
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(checkpoint_dir: str) -> tuple[list[dict], dict]:
    files = sorted(f for f in os.listdir(checkpoint_dir)
                   if f.startswith("gen_") and f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"{checkpoint_dir} 无检查点")
    with open(os.path.join(checkpoint_dir, files[-1]), encoding="utf-8") as f:
        data = json.load(f)
    return data["history"], data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="checkpoints")
    ap.add_argument("--out", default="evolution_curves.png")
    args = ap.parse_args()

    history, data = load_history(args.dir)
    gens = [r["gen"] for r in history]
    best = [r["best_fitness"] for r in history]
    med = [r["median_fitness"] for r in history]
    worst = [r["worst_fitness"] for r in history]
    acc = [r["best_accuracy"] for r in history]
    layers = [r["best_metrics"]["layers"] for r in history]
    neurons = [r["best_metrics"]["neurons"] for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("LIF 神经进化 — 数字识别（监督线性读出）", fontsize=14)

    ax = axes[0, 0]
    ax.plot(gens, best, "o-", label="best", color="#d62728")
    ax.plot(gens, med, "s-", label="median", color="#1f77b4")
    ax.plot(gens, worst, "^-", label="worst", color="#7f7f7f")
    ax.set_xlabel("generation"); ax.set_ylabel("fitness"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("适应度（准确率+稀疏-紧凑）随代数变化")

    ax = axes[0, 1]
    ax.plot(gens, acc, "o-", color="#2ca02c")
    ax.set_xlabel("generation"); ax.set_ylabel("best accuracy"); ax.grid(alpha=0.3)
    ax.set_title("当代最优个体在验证集上的读出准确率")

    ax = axes[1, 0]
    ax.plot(gens, layers, "o-", color="#9467bd")
    ax.set_xlabel("generation"); ax.set_ylabel("best layers"); ax.grid(alpha=0.3)
    ax.set_yticks(sorted(set(layers)))
    ax.set_title("当代最优个体层数")

    ax = axes[1, 1]
    ax.plot(gens, neurons, "o-", color="#ff7f0e")
    ax.set_xlabel("generation"); ax.set_ylabel("best neurons"); ax.grid(alpha=0.3)
    ax.set_title("当代最优个体神经元总数")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"已保存: {args.out}  （共 {len(gens)} 代）")

    # 终端摘要
    last = history[-1]
    g = last["best_genome"]
    ls = ",".join(str(l["n_out"]) for l in g["layers"])
    print(f"最优适应度 {last['best_fitness']:.3f} (acc={last['best_accuracy']:.3f})")
    print(f"最优架构: layers=[{ls}] T={g['T']} train={g['train_samples']}")


if __name__ == "__main__":
    main()
