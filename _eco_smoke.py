# _eco_smoke.py
"""回合制引擎冒烟：跑 30 回合，记录每回合种群/正确率/自然死亡率/耗时与诚实基线。"""
import time
import numpy as np
import eco_engine as eco

def main():
    e = eco.Ecosystem(seed=0)
    t0 = time.time()
    prev_best = 0.0
    for r in range(30):
        t = time.time()
        events, stats = e.step_round()
        dt = time.time() - t
        prev_best = max(prev_best, stats["avg_acc"])
        print(f"round {stats['round']:>3} | alive={stats['alive']:>5} "
              f"correct={stats['avg_acc']:.3f} natural_rate={stats['natural_rate']:.4f} "
              f"n_deaths={stats['total_deaths']:>5} | {dt:.2f}s")
    print(f"\n=== 冒烟完成 === 最优单回合正确率={prev_best:.3f} "
          f"(随机基线≈0.10) 总耗时 {time.time()-t0:.0f}s "
          f"最终 natural_rate={stats['natural_rate']:.4f}（目标 0.95）")

if __name__ == "__main__":
    main()
