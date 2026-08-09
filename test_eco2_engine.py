# test_eco2_engine.py
import numpy as np, torch, pytest
from eco2_engine import (Genome, Eco2Config, random_genome, mutate_genome,
                         init_weights, load_mnist, downsample)

def test_random_genome_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = random_genome(rng)
        assert 1 <= len(g.layer_sizes) <= 3
        assert all(32 <= s <= 128 for s in g.layer_sizes)
        assert 2 <= g.wta_k <= 12
        assert 0.80 <= g.leak <= 0.99
        assert 0.5 <= g.input_gain <= 3.0
        assert 0.5 <= g.threshold_scale <= 2.0
        assert 0.0 <= g.lamarckism <= 1.0
        assert 0.5 <= g.lr_scale <= 2.0
        assert 1 <= g.fecundity <= 3
        assert 0.02 <= g.mutation_rate <= 0.2

def test_genome_roundtrip():
    g = random_genome(np.random.default_rng(1))
    g2 = Genome.from_dict(g.to_dict())
    assert g == g2

def test_init_weights_shapes():
    g = Genome(layer_sizes=(64,), wta_k=6, leak=0.94, input_gain=1.0,
               threshold_scale=1.0, lamarckism=0.5, lr_scale=1.0,
               fecundity=1, mutation_rate=0.1)
    ws = init_weights(g, 196, np.random.default_rng(0))
    assert len(ws) == 2
    assert ws[0].shape == (64, 196)
    assert ws[1].shape == (10, 64)

def test_load_mnist_shapes():
    imgs, labs = load_mnist()
    assert imgs.shape == (60000, 28, 28)
    assert labs.shape == (60000,)

def test_downsample_shape():
    imgs = np.zeros((4, 28, 28), np.float32)
    d = downsample(imgs, 14)
    assert d.shape == (4, 196)
