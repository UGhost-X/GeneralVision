# _eco_smoke.py
"""引擎冒烟跑：20 天，打印种群统计与耗时，验证生态动力学并记录诚实基线。"""
import time
import numpy as np
import eco_engine as eco

def main():
    eco_ = eco.Ecosystem(seed=0)
    t0 = time.time()
    prev_best = -1
    for d in range(20):
        t = time.time()
        events, stats = eco_.step_day()
        dt = time.time() - t
        names = sorted(set(g.name for g in eco_.pop))
        ages = np.array([g.age for g in eco_.pop])
        print(f"day {stats['day']:>3} | alive={stats['alive']:>3} "
              f"avg={stats['avg_acc']:.3f} best={stats['best_acc']:.3f} "
              f"med={stats['median_acc']:.3f} worst={stats['worst_acc']:.3f} "
              f"| 唯一名={len(names)} 平均年龄={ages.mean():.1f} "
              f"最老={ages.max()} | {dt:.2f}s")
        if stats["best_acc"] > prev_best:
            prev_best = stats["best_acc"]
    print(f"\n=== 冒烟完成 === 最优吃对率={prev_best:.3f} "
          f"(随机基线≈0.10) 总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
