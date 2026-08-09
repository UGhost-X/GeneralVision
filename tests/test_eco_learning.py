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
    _readout_source,
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


def test_online_learning_changes_readout(eco):
    eco.reset()
    for o in eco.population:
        o.genome.readout_lr = 0.5
    alive = [o for o in eco.population if o.alive][:20]
    before = [o.learned_weights.copy() for o in alive]
    spikes = _sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    shims = _phenotype_genomes(alive)
    _, rates, hidden_rates = forward_learn(shims, spikes)
    label = int(eco._test_labels[0])
    eco._learn_readout(alive, hidden_rates, rates, label)
    for i, o in enumerate(alive):
        assert not np.allclose(before[i], o.learned_weights)
        assert o.samples_learned == 1


def test_zero_readout_lr_skips_learning(eco):
    eco.reset()
    for o in eco.population:
        o.genome.readout_lr = 0.0
    alive = [o for o in eco.population if o.alive][:20]
    before = [o.learned_weights.copy() for o in alive]
    spikes = _sample_spikes(
        np.asarray(eco._test_images[0], np.float32), eco.rng
    )
    shims = _phenotype_genomes(alive)
    _, rates, hidden_rates = forward_learn(shims, spikes)
    label = int(eco._test_labels[0])
    eco._learn_readout(alive, hidden_rates, rates, label)
    for i, o in enumerate(alive):
        assert np.allclose(before[i], o.learned_weights)
        assert o.samples_learned == 0


def test_maturate_updates_readout(eco):
    eco.reset()
    eco.config.maturity_samples = 3
    orgs = [o for o in eco.population if o.alive][:5]
    for o in orgs:
        o.genome.readout_lr = 0.5
    before = [o.learned_weights.copy() for o in orgs]
    eco._maturate(orgs)
    for i, o in enumerate(orgs):
        assert o.samples_learned == 3
        assert not np.allclose(before[i], o.learned_weights)


def test_step_respects_learning_switch(eco):
    eco.reset()
    eco.config.learning_on = False
    for o in eco.population:
        o.genome.readout_lr = 0.5
    before = {
        o.uid: o.learned_weights.copy()
        for o in eco.population
        if o.alive
    }
    eco.step()
    for o in eco.population:
        if o.alive and o.uid in before:
            assert np.allclose(before[o.uid], o.learned_weights)


def test_readout_source_prefers_learned():
    wa = np.zeros(100)
    wb = np.zeros(100)
    la = np.ones(100)
    lb = np.ones(100) * 2
    # 高 fitness 亲本的已学读出层优先
    assert np.array_equal(_readout_source(1.0, 0.0, wa, wb, la, lb), la)
    assert np.array_equal(_readout_source(0.0, 1.0, wa, wb, la, lb), lb)
    # 缺已学时回退到该亲本的基因型权重
    assert np.array_equal(_readout_source(1.0, 0.0, wa, wb, None, None), wa)
    assert np.array_equal(_readout_source(0.0, 1.0, wa, wb, None, None), wb)


def test_repro_success_uses_smoothed_accuracy(eco):
    """产错惩罚改用平滑准确率 acc_ema，而非单次对错（消除单数字噪声误杀）。"""
    eco.reset()
    alive = [o for o in eco.population if o.alive][:2]
    a, b = alive
    a.correct = False
    b.correct = False
    a.genome.wrong_tolerance = 1.0
    b.genome.wrong_tolerance = 1.0
    a.acc_ema = 0.9
    b.acc_ema = 0.9
    p_high = eco._pair_repro_success(a, b)
    a.acc_ema = 0.1
    b.acc_ema = 0.1
    p_low = eco._pair_repro_success(a, b)
    # 即便本次都产错，平滑准确率高者惩罚更轻
    assert p_high > p_low


def test_repro_success_wrong_tolerance_still_works(eco):
    eco.reset()
    alive = [o for o in eco.population if o.alive][:2]
    a, b = alive
    a.correct = b.correct = False
    a.acc_ema = b.acc_ema = 0.1
    a.genome.wrong_tolerance = 3.0
    b.genome.wrong_tolerance = 3.0
    p_tolerant = eco._pair_repro_success(a, b)
    a.genome.wrong_tolerance = 1.0
    b.genome.wrong_tolerance = 1.0
    p_intolerant = eco._pair_repro_success(a, b)
    # 高错误耐受者繁殖成功率更高
    assert p_tolerant > p_intolerant
