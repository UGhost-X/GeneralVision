"""LIF 生态游戏在线学习 P1 测试。"""

import numpy as np
import pytest

from eco_engine import EcoConfig, Ecosystem


@pytest.fixture(scope="module")
def eco() -> Ecosystem:
    """共享一个最小生态实例（首次构造含奠基筛选，之后 reset 廉价）。"""
    return Ecosystem(config=EcoConfig(init_pop=100), seed=7)


def test_smoke_step(eco):
    eco.reset()
    events = eco.step()
    assert events["round"] == 1
    assert events["population_after"] > 0
    assert "accuracy" in events
    assert eco.state()["population_size"] > 0
