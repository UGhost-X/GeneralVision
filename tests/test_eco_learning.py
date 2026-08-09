"""LIF 生态游戏在线学习 P1 测试。"""

import numpy as np
import pytest

from eco_engine import (
    EcoConfig,
    Ecosystem,
    crossover,
    forward,
    forward_learn,
    random_genome,
    _phenotype_genomes,
    _sample_spikes,
)


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


def test_forward_learn_shapes(eco):
    eco.reset()
    alive = [o for o in eco.population if o.alive][:10]
    spikes = _sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    shims = _phenotype_genomes(alive)
    preds, rates, hidden_rates = forward_learn(shims, spikes)
    assert len(preds) == len(alive)
    assert rates.shape == (len(alive), 10)
    for i, o in enumerate(alive):
        assert hidden_rates[i].shape == (o.genome.layer_sizes[-1],)


def test_shims_match_genotype_when_unlearned(eco):
    eco.reset()
    alive = [o for o in eco.population if o.alive][:10]
    spikes = _sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    genomes = [o.genome for o in alive]
    shims = _phenotype_genomes(alive)
    p1, r1 = forward(genomes, spikes)
    p2, r2, _ = forward_learn(shims, spikes)
    assert np.array_equal(p1, p2)
    assert np.allclose(r1, r2)
