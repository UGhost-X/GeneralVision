"""LIF 生态游戏在线学习 P1 测试。"""

import numpy as np
import pytest

from eco_engine import EcoConfig, Ecosystem, crossover, random_genome


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


def test_genome_has_learning_genes():
    g = random_genome(np.random.default_rng(0))
    assert 0.0 <= g.readout_lr <= 1.0
    assert 0.0 <= g.hidden_plasticity <= 1.0
    assert 0.0 <= g.plasticity_drift <= 0.5
    d = g.to_dict()
    assert d["readout_lr"] == g.readout_lr


def test_crossover_keeps_learning_genes():
    a = random_genome(np.random.default_rng(1))
    b = random_genome(np.random.default_rng(2))
    c = crossover(a, b, 1.0, 1.0, np.random.default_rng(3), 0.5, 0.5)
    assert 0.0 <= c.readout_lr <= 1.0
    assert 0.0 <= c.hidden_plasticity <= 1.0
    assert 0.0 <= c.plasticity_drift <= 0.5


def test_organism_has_learned_weights(eco):
    eco.reset()
    alive = [o for o in eco.population if o.alive]
    assert alive
    o = alive[0]
    assert o.learned_weights is not None
    assert o.learned_weights.shape == o.genome.weights.shape
    assert np.allclose(o.learned_weights, o.genome.weights)
    d = o.to_dict()
    assert "readout_lr" in d and "learning_amount" in d
    assert d["learning_amount"] == 0.0
