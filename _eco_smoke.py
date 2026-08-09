# _eco_smoke.py
"""合并迭代冒烟：跑 30 回合，记录每回合种群/正确率/存活率/alpha/多层占比/单回合耗时 + 诚实基线。

多层占比 = 隐藏层数 > 1 的个体比例；maxAge = 当前存活个体最大年龄（对照存活回合数上限）。
"""
import time
import eco_engine as eco


def main():
    e = eco.Ecosystem(seed=0)
    t0 = time.time()
    prev_best = 0.0
    multi_peak = 0
    for _ in range(30):
        t = time.time()
        events, stats = e.step_round()
        dt = time.time() - t
        prev_best = max(prev_best, stats["avg_acc"])
        n_multi = sum(1 for g in e.pop if len(g.arch()) > 1)
        multi_peak = max(multi_peak, n_multi)
        multi_frac = n_multi / max(1, len(e.pop))
        max_age = max((g.age for g in e.pop), default=0)
        print(f"round {stats['round']:>3} | alive={stats['alive']:>5} "
              f"correct={stats['avg_acc']:.3f} sr={stats['survival_rate']:.3f} "
              f"alpha={stats['alpha']:>6.1f} multi={n_multi:>4}({multi_frac:.2f}) "
              f"maxAge={max_age:>3} newborns={stats['newborns']:>3} | {dt:.2f}s")
    print(f"\n=== 冒烟完成 === 最优正确率={prev_best:.3f}（随机基线≈0.10） "
          f"多层峰值={multi_peak} 只 总耗时 {time.time()-t0:.0f}s "
          f"最终 natural_rate={stats['natural_rate']:.4f}（目标 0.95）")


if __name__ == "__main__":
    main()
