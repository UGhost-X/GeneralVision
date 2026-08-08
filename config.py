"""进化超参数集中配置。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvolutionConfig:
    # 种群与代数
    population_size: int = 50
    g_max: int = 50
    patience: int = 8            # 最优适应度连续无提升的代数，超过则提前终止

    # 选择
    elite: int = 2               # 每代免变异直达下一代的精英数
    bottom_frac: float = 0.3     # 淘汰底部比例
    top_frac: float = 0.3        # 顶部繁殖比例
    tournament_size: int = 3     # 锦标赛选择大小
    roulette_frac: float = 0.2   # 繁殖者中轮盘赌（多样性补充）比例

    # 适应度
    w_sparse: float = 0.05       # 稀疏奖励权重
    w_compact: float = 0.01      # 紧凑惩罚权重
    eval_repeats: int = 1        # 监督读出已稳定；如需更强抑噪可设 2

    # 数据/运行
    val_size: int = 1000
    device: str = "cpu"
    seed: int = 0
    checkpoint_every: int = 2    # 每 N 代存检查点（防崩溃丢进度）
    checkpoint_dir: str = "checkpoints"
    log_every: int = 1
    is_smoke: bool = False       # 冒烟模式：个体一生训练样本缩减

    def smoke(self, override: bool = True) -> "EvolutionConfig":
        """冒烟测试：小种群、少代数、小验证集、CPU。"""
        c = EvolutionConfig(**self.__dict__)
        if override:
            c.population_size = 6
            c.g_max = 2
            c.val_size = 200
            c.device = "cpu"
            c.patience = 100
            c.is_smoke = True
            c.eval_repeats = 1
        return c
