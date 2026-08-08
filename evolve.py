"""进化主循环：稳态选择（锦标赛+轮盘赌）、精英保留、分裂繁殖、日志、检查点。

用法：
    python evolve.py            # 正式运行（config.py 默认 50 个体 × 50 代）
    python evolve.py --smoke    # 冒烟测试（6 个体 × 2 代）
    python evolve.py --resume <检查点目录>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from config import EvolutionConfig
from data_loading import load_mnist, split_pools
from evaluate import evaluate
from genome import Genome, seed_genome
from mutate import MutationSchedule, mutate

LOG_FILE = "evolution_log.jsonl"


def init_population(cfg: EvolutionConfig, rng: np.random.Generator) -> list[Genome]:
    """第 0 代：1 个调通基线种子 + 其余为其小幅变异（保证初始多样性）。"""
    seed = seed_genome(n_neurons=100, seed=cfg.seed)
    pop = [seed]
    for i in range(cfg.population_size - 1):
        child = mutate(seed, 0, rng)
        child.name = f"seed#{i}"
        pop.append(child)
    if cfg.is_smoke:
        for g in pop:
            g.train_samples = 150
    return pop


def tournament_select(fitness: list[float], survivors: list[int], n_breeders: int,
                      k: int, roulette_frac: float, rng: np.random.Generator) -> list[int]:
    """从幸存者中选 n_breeders 个繁殖者：锦标赛为主 + 少量轮盘赌补充多样性。"""
    breeders: list[int] = []
    for _ in range(n_breeders):
        if rng.random() < roulette_frac:
            # 轮盘赌：按适应度在幸存者中的排名加权
            ranks = np.argsort(np.argsort([fitness[i] for i in survivors]))  # 0..M-1 排名
            probs = (ranks + 1) / ranks.sum()
            idx = rng.choice(len(survivors), p=probs)
            breeders.append(survivors[int(idx)])
        else:
            candidates = rng.choice(survivors, size=k, replace=True)
            breeder = int(max(candidates, key=lambda i: fitness[int(i)]))
            breeders.append(breeder)
    return breeders


def evolve(cfg: EvolutionConfig) -> dict:
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    train_img, train_lbl, _, _ = load_mnist()
    pool, val = split_pools(train_img, train_lbl, val_size=cfg.val_size, seed=cfg.seed)
    (tr_img, tr_lbl), (val_img, val_lbl) = pool, val

    pop = init_population(cfg, rng)
    sched = MutationSchedule(g_max=cfg.g_max)
    history: list[dict] = []
    best_fitness = float("-inf")
    no_improve = 0
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_f = open(os.path.join(cfg.checkpoint_dir, LOG_FILE), "a") if cfg.checkpoint_dir else None

    for gen in range(cfg.g_max):
        t_gen = time.time()
        results = []
        for i, g in enumerate(pop):
            # 多次不同初始化种子评估取平均，抑制权重初始化的随机噪声
            best_fit_rep, best_metrics_rep = None, None
            fit_scores = []
            for rep in range(cfg.eval_repeats):
                g_rep = g.clone()
                g_rep.seed = g.seed + rep * 7919
                fit, metrics = evaluate(g_rep, tr_img, tr_lbl, val_img, val_lbl, device,
                                        w_sparse=cfg.w_sparse, w_compact=cfg.w_compact)
                fit_scores.append(fit)
                if best_fit_rep is None or fit > best_fit_rep:
                    best_fit_rep, best_metrics_rep = fit, metrics
            results.append((float(np.mean(fit_scores)), best_metrics_rep, g))
            if cfg.log_every and gen % cfg.log_every == 0:
                print(f"  ind {i}: fit={np.mean(fit_scores):.3f} "
                      f"acc={best_metrics_rep['accuracy']:.3f} {g.describe()}")

        results.sort(key=lambda r: r[0], reverse=True)
        fits = [r[0] for r in results]
        best_f, best_m, best_g = results[0]
        med_f = float(np.median(fits))
        worst_f = fits[-1]
        # 架构多样性：不同的（层数, 各层神经元数）签名数
        archs = {tuple(g.layers[i].n_out for i in range(len(g.layers))) for _, _, g in results}
        div = len(archs)

        # 提前终止判断
        if best_f > best_fitness + 1e-6:
            best_fitness = best_f
            no_improve = 0
        else:
            no_improve += 1

        rec = {
            "gen": gen,
            "best_fitness": best_f,
            "median_fitness": med_f,
            "worst_fitness": worst_f,
            "best_accuracy": best_m["accuracy"],
            "best_genome": best_g.to_dict(),
            "best_metrics": best_m,
            "time_gen": time.time() - t_gen,
            "time_total": time.time() - t_start,
        }
        history.append(rec)
        if log_f:
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()

        print(f"[gen {gen}] best={best_f:.3f} (acc={best_m['accuracy']:.3f}, "
              f"sparse={best_m['sparsity']:.3f}, neurons={best_m['neurons']}, "
              f"layers={best_m['layers']}) | med={med_f:.3f} worst={worst_f:.3f} "
              f"| {best_g.describe()}")

        # ---- 选择与繁殖 ----
        N = len(pop)
        n_elim = max(1, round(N * cfg.bottom_frac))
        n_children = n_elim
        survivors = list(range(N - n_elim))          # 顶部幸存（含精英）
        breeders = tournament_select(fits, survivors, n_children,
                                     cfg.tournament_size, cfg.roulette_frac, rng)

        next_pop: list[Genome] = []
        for idx in survivors:                        # 幸存者原样进入下一代
            next_pop.append(results[idx][2])
        for b in breeders:                           # 繁殖者各分裂 1 子代填补淘汰
            child = mutate(results[b][2], gen, rng, sched)
            child.name = f"{results[b][2].name}#g{gen}"
            next_pop.append(child)
        pop = next_pop[:N]

        # 检查点
        if cfg.checkpoint_every and (gen + 1) % cfg.checkpoint_every == 0:
            _save_checkpoint(cfg, gen, pop, history)

        if no_improve >= cfg.patience:
            print(f"提前终止：最优适应度 {cfg.patience} 代无提升（gen={gen}）")
            break

    if log_f:
        log_f.close()
    _save_checkpoint(cfg, gen, pop, history)
    summary = {
        "best_fitness": best_fitness,
        "final_history": history,
        "best_genome": best_g.to_dict(),
        "best_metrics": best_m,
    }
    print(f"\n=== 进化完成 === 最优适应度 {best_fitness:.3f} "
          f"(acc={best_m['accuracy']:.3f}) 耗时 {time.time()-t_start:.0f}s")
    print("最优架构:", best_g.describe())
    return summary


def _save_checkpoint(cfg: EvolutionConfig, gen: int, pop: list[Genome],
                     history: list[dict]) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"gen_{gen:03d}.json")
    data = {
        "gen": gen,
        "population": [g.to_dict() for g in pop],
        "history": history,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  检查点已保存: {path}")


def _resume(cfg: EvolutionConfig, checkpoint_dir: str) -> dict:
    """从检查点目录的最后一代恢复。"""
    files = sorted(f for f in os.listdir(checkpoint_dir) if f.startswith("gen_") and f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"检查点目录 {checkpoint_dir} 无 gen_*.json")
    last = os.path.join(checkpoint_dir, files[-1])
    with open(last, encoding="utf-8") as f:
        data = json.load(f)
    pop = [Genome.from_dict(d) for d in data["population"]]
    print(f"从 {last} 恢复，gen={data['gen']}，种群 {len(pop)} 个体")
    # TODO: 恢复后续续跑（当前先实现保存，恢复续跑留作后续）
    return {"resumed_from": last, "gen": data["gen"], "population": pop}


def main() -> None:
    ap = argparse.ArgumentParser(description="LIF 神经进化数字识别")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试（小种群少代数）")
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None, metavar="DIR")
    args = ap.parse_args()

    cfg = EvolutionConfig()
    if args.population:
        cfg.population_size = args.population
    if args.generations:
        cfg.g_max = args.generations
    if args.device:
        cfg.device = args.device
    if args.smoke:
        cfg = cfg.smoke()

    if args.resume:
        _resume(cfg, args.resume)
        return
    evolve(cfg)


if __name__ == "__main__":
    main()
