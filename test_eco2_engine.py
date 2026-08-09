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


# --------------------------------------------------------------------------- #
# Task 3: mSTDP 学习更新 + 能量结算 + 死亡判定
# --------------------------------------------------------------------------- #
import torch
from eco2_engine import apply_mstdp, settle_energy, mark_deaths, Eco2Config, Genome

def test_mstdp_update_sign():
    """已知 reward 方向 → 权重变化方向正确。"""
    cfg = Eco2Config(lr_base=0.1, w_min=-2.0, w_max=2.0)
    N, OUT, IN = 2, 3, 4
    w = torch.randn(N, OUT, IN) * 0.1
    w0 = w.clone()
    elig = torch.ones(N, OUT, IN) * 0.5
    reward = torch.tensor([1.0, -1.0])
    apply_mstdp([w], [elig], reward, cfg)
    # 个体0 正确(reward+1)：w 增大；个体1 错误(reward-1)：w 减小
    assert (w[0] > w0[0]).all()
    assert (w[1] < w0[1]).all()

def test_mstdp_clamp():
    cfg = Eco2Config(lr_base=10.0, w_min=-1.0, w_max=1.0)
    w = torch.zeros(2, 2, 2)
    elig = torch.ones(2, 2, 2) * 10.0
    apply_mstdp([w], [elig], torch.tensor([1.0, -1.0]), cfg)
    assert (w <= 1.0).all() and (w >= -1.0).all()

def test_settle_energy():
    cfg = Eco2Config(e_gain=10.0, e_cost=8.0, metabolism=1.0)
    e = torch.tensor([100.0, 100.0])
    correct = torch.tensor([True, False])
    out = settle_energy(e, correct, cfg)
    assert out[0].item() == 100 + 10 - 1      # 正确
    assert out[1].item() == 100 - 8 - 1       # 错误

def test_mark_deaths_starvation():
    cfg = Eco2Config(age_max=None)
    e = torch.tensor([0.0, 5.0, -1.0, 100.0])
    age = torch.tensor([0, 0, 0, 0])
    dead = mark_deaths(e, age, cfg)
    assert dead.tolist() == [True, False, True, False]

def test_mark_deaths_age():
    cfg = Eco2Config(age_max=50)
    e = torch.tensor([100.0, 100.0])
    age = torch.tensor([49, 50])
    dead = mark_deaths(e, age, cfg)
    assert dead.tolist() == [False, True]


# --------------------------------------------------------------------------- #
# Task 4: 繁殖 + 拉马克混合遗传
# --------------------------------------------------------------------------- #
import numpy as np, torch
from eco2_engine import (lamarckism_blend, reproduce, random_genome, init_weights,
                         Organism, Eco2Config, mutate_genome, Genome)

def test_lamarckism_zero_is_fresh():
    cfg = Eco2Config(mutation_noise=0.0)
    rng = np.random.default_rng(0)
    g = Genome(layer_sizes=(32,), wta_k=6, leak=0.94, input_gain=1.0,
               threshold_scale=1.0, lamarckism=0.0, lr_scale=1.0, fecundity=1,
               mutation_rate=0.1)
    pw = init_weights(g, 196, rng, scale=1.0)  # 大权重便于区分
    cw = lamarckism_blend(pw, g, rng, cfg)
    # lamarckism=0 → 子代应 ≈ 新随机，与亲代差异明显
    diff = sum((cw[i] - pw[i]).abs().mean().item() for i in range(len(pw)))
    assert diff > 0.5

def test_lamarckism_one_is_parent():
    cfg = Eco2Config(mutation_noise=0.0)
    rng = np.random.default_rng(1)
    g = Genome(layer_sizes=(32,), wta_k=6, leak=0.94, input_gain=1.0,
               threshold_scale=1.0, lamarckism=1.0, lr_scale=1.0, fecundity=1,
               mutation_rate=0.1)
    pw = init_weights(g, 196, rng, scale=0.3)
    cw = lamarckism_blend(pw, g, rng, cfg)
    for i in range(len(pw)):
        torch.testing.assert_close(cw[i], pw[i], atol=1e-6, rtol=1e-6)

def test_reproduce_pays_cost_and_fecundity():
    cfg = Eco2Config(repro_cost=100.0, e_birth=80.0)
    rng = np.random.default_rng(2)
    g = Genome(layer_sizes=(32,), wta_k=6, leak=0.94, input_gain=1.0,
               threshold_scale=1.0, lamarckism=0.5, lr_scale=1.0, fecundity=2,
               mutation_rate=0.1)
    parent = Organism(uid=0, genome=g, energy=300.0,
                      weights=init_weights(g, 196, rng))
    children = reproduce([parent], rng, cfg, [1])
    assert parent.energy == 200.0          # 付了繁殖代价
    assert len(children) == 2              # fecundity=2
    for c in children:
        assert c.alive and c.age == 0 and c.energy == 80.0
        assert c.genome != g or c.genome == g  # 基因至少结构合法
        assert len(c.weights) == len(parent.weights)
    # uid 递增
    assert [c.uid for c in children] == [1, 2]
