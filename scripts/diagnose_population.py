"""诊断：种群萎缩 + 准确率不涨 的机制取证。

测量：
1. 种群轨迹（population_after 随回合变化）
2. 第 N 回合存活个体的 表型 vs 基因型 读出层准确率差距
   （若 表型 >> 基因型，说明"学习发生但不遗传"，后代从差起点起步）
3. 基因型读出层准确率随代际是否衰减
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eco_engine import (
    EcoConfig,
    Ecosystem,
    forward,
    _phenotype_genomes,
    _sample_spikes,
)


def eval_acc(eco, organisms):
    """对一组 organism，用固定 50 张测试图（每数字 5 张）评估准确率。"""
    if not organisms:
        return 0.0
    labels, spikes_list = [], []
    for digit in range(10):
        candidates = np.flatnonzero(eco._test_labels == digit)
        for idx in candidates[:5]:
            spikes_list.append(
                _sample_spikes(
                    np.asarray(eco._test_images[idx], np.float32), eco.rng
                )
            )
            labels.append(digit)
    correct = 0
    for spikes, label in zip(spikes_list, labels):
        preds, _ = forward(_phenotype_genomes(organisms), spikes)
        correct += int((preds == label).sum())
    return correct / (len(organisms) * len(labels))


def eval_geno_acc(eco, organisms):
    """用基因型（先天）权重评估准确率，即后代继承到的起点质量。"""
    if not organisms:
        return 0.0
    labels, spikes_list = [], []
    for digit in range(10):
        candidates = np.flatnonzero(eco._test_labels == digit)
        for idx in candidates[:5]:
            spikes_list.append(
                _sample_spikes(
                    np.asarray(eco._test_images[idx], np.float32), eco.rng
                )
            )
            labels.append(digit)
    correct = 0
    for spikes, label in zip(spikes_list, labels):
        preds, _ = forward([o.genome for o in organisms], spikes)
        correct += int((preds == label).sum())
    return correct / (len(organisms) * len(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning", action="store_true", default=True)
    parser.add_argument("--no-learning", dest="learning", action="store_false")
    args = parser.parse_args()

    eco = Ecosystem(
        config=EcoConfig(init_pop=1000, learning_on=args.learning), seed=11
    )
    pop_history = []
    for rnd in range(1, 41):
        events = eco.step()
        pop_history.append(events["population_after"])
        natural = events.get("natural_deaths", 0)
        no_repro = events.get("no_repro_deaths", 0)
        offspring = events.get("offspring", 0)
        alive_before = events.get("alive_before", 0)
        print(
            f"round={rnd} pop={events['population_after']} "
            f"acc={events['accuracy']:.3f} "
            f"before={alive_before} natural={natural} "
            f"no_repro={no_repro} born={offspring}",
            flush=True,
        )
        if rnd in (5, 10, 20, 30, 40):
            alive = [o for o in eco.population if o.alive]
            sample = alive[:50]
            phe = eval_acc(eco, sample)
            geno = eval_geno_acc(eco, sample)
            print(
                f"  [census] phenotype_acc={phe:.3f} genotype_acc={geno:.3f}",
                flush=True,
            )
    print("pop_history=", pop_history)


if __name__ == "__main__":
    main()
