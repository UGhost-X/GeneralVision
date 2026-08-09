"""P1 诚实验证：R0(纯遗传，learning_on=False) vs R1(在线学习，learning_on=True)。

固定 seed 各跑 N 回合，输出每回合 accuracy 与期末均值/最优。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eco_engine import EcoConfig, Ecosystem


def run(seed: int, rounds: int, learning_on: bool) -> list:
    config = EcoConfig(init_pop=1000, learning_on=learning_on)
    eco = Ecosystem(config=config, seed=seed)
    accs = []
    for _ in range(rounds):
        events = eco.step()
        accs.append(float(events.get("accuracy", 0.0)))
    return accs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    results = {"rounds": args.rounds, "seeds": args.seeds, "runs": {}}
    for seed in args.seeds:
        for tag, learning_on in (("R0_no_learning", False), ("R1_learning", True)):
            accs = run(seed, args.rounds, learning_on)
            results["runs"][f"{tag}_seed{seed}"] = {
                "trajectory": accs,
                "mean": float(np.mean(accs)),
                "last10_mean": float(np.mean(accs[-10:])),
                "max": float(np.max(accs)),
            }
            print(
                f"seed={seed} {tag}: mean={np.mean(accs):.3f} "
                f"last10={np.mean(accs[-10:]):.3f} max={np.max(accs):.3f}",
                flush=True,
            )
    with open(
        "docs/superpowers/notes/2026-08-09-online-learning-p1-results.json",
        "w",
    ) as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
