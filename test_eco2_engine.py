# test_eco2_engine.py
import numpy as np, torch, pytest
from eco2_engine import (Genome, Eco2Config, random_genome, mutate_genome,
                         init_weights, load_mnist, downsample, forward_group)

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


# --------------------------------------------------------------------------- #
# Task 2: 批量前向（LIF + WTA + 读出）
# --------------------------------------------------------------------------- #
def _mk_inputs(n=4, in_s=196, T=8, hidden=(64,)):
    g = Genome(layer_sizes=hidden, wta_k=6, leak=0.94, input_gain=1.0,
               threshold_scale=1.0, lamarckism=0.5, lr_scale=1.0, fecundity=1,
               mutation_rate=0.1)
    ws0 = init_weights(g, in_s, np.random.default_rng(0))  # [out,in]
    weights = [w.unsqueeze(0).repeat(n, 1, 1) for w in ws0]  # [N,out,in]
    seq = (torch.rand(T, n, in_s) > 0.7).float()
    return seq, weights, [g] * n


def test_forward_shapes():
    seq, weights, genomes = _mk_inputs()
    cfg = Eco2Config(T=8)
    out_sum, elig = forward_group(seq, weights, genomes, cfg)
    n = seq.shape[1]
    assert out_sum.shape == (n, 10)
    assert len(elig) == len(weights)
    assert elig[0].shape == (n, 64, 196)   # 隐层 eligibility [N,out,in]
    assert elig[-1].shape == (n, 10, 64)   # 读出 eligibility


def test_prediction_is_argmax_of_counts():
    seq, weights, genomes = _mk_inputs()
    cfg = Eco2Config(T=8)
    out_sum, _ = forward_group(seq, weights, genomes, cfg)
    pred = out_sum.argmax(1)
    assert (out_sum.max(1).values >= 0).all()   # 累计脉冲数非负


def test_batch_equals_individual():
    """批量前向 == 逐个体前向（种群一致性关键测试）"""
    n, in_s, T = 5, 196, 6
    g = Genome(layer_sizes=(48,), wta_k=4, leak=0.94, input_gain=1.0,
               threshold_scale=1.0, lamarckism=0.5, lr_scale=1.0, fecundity=1,
               mutation_rate=0.1)
    cfg = Eco2Config(T=T)
    rng = np.random.default_rng(0)
    seq = (torch.rand(T, n, in_s) > 0.7).float()
    # 每个体不同权重
    weights_3d = []
    for _ in range(n):
        ws = init_weights(g, in_s, rng)
        weights_3d.append([w.unsqueeze(0) for w in ws])
    stacked = [torch.cat([ws[i] for ws in weights_3d], 0) for i in range(len(weights_3d[0]))]
    out_batch, _ = forward_group(seq, stacked, [g] * n, cfg)
    # 逐个体
    preds = []
    for k in range(n):
        w_k = [w[k:k+1] for w in stacked]
        out_k, _ = forward_group(seq[:, k:k+1], w_k, [g], cfg)
        preds.append(out_k.argmax(1).item())
    assert out_batch.argmax(1).tolist() == preds


def test_wta_topk():
    """精确 top-k（含并列按索引序），k 可每行不同。"""
    from eco2_engine import _wta
    spikes = torch.tensor([[3., 1., 2., 0., 0.],
                           [0., 5., 1., 4., 2.]])
    k = torch.tensor([2, 3])
    out = _wta(spikes, k)
    assert (out[0] == torch.tensor([3., 0., 2., 0., 0.])).all()   # top2: 3,2
    assert (out[1] == torch.tensor([0., 5., 0., 4., 2.])).all()   # top3: 5,4,2
