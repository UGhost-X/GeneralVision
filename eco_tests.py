# eco_tests.py
"""LIF 生态引擎测试。运行：python eco_tests.py"""
import numpy as np
import eco_engine as eco

def test_crossover_mixes_both_parents():
    rng = np.random.default_rng(7)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child = eco.crossover(a, b, rng)
    assert child.layers[0].shape == (784, 100), child.layers[0].shape
    assert child.readout.shape == (100, 10), child.readout.shape
    # 逐权重取父/母：每个权重应"更接近"其中一方（噪声 σ=0.01 远小于双亲差距）
    for Wc, Wa, Wb in [(child.layers[0], a.layers[0], b.layers[0]),
                       (child.readout, a.readout, b.readout)]:
        closer_a = (np.abs(Wc - Wa) < np.abs(Wc - Wb)).mean()
        assert 0.35 < closer_a < 0.65, f"closer_a={closer_a}"

def test_columns_normalized():
    g = eco.random_genome("n", np.random.default_rng(0))
    norms = np.linalg.norm(g.layers[0], axis=0)
    assert np.allclose(norms, eco.W_NORM_HIDDEN, atol=1e-4), norms[:5]
    rnorms = np.linalg.norm(g.readout, axis=0)
    assert np.allclose(rnorms, eco.W_NORM_READOUT, atol=1e-4), rnorms[:5]

def test_mutation_is_rare():
    rng = np.random.default_rng(11)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child = eco.crossover(a, b, rng)
    d_a = np.abs(child.layers[0] - a.layers[0])
    d_b = np.abs(child.layers[0] - b.layers[0])
    from_a = (d_a < d_b).mean()                 # 一半来自 a，一半来自 b
    both_far = ((d_a > 0.3) & (d_b > 0.3)).mean()  # 大突变应极罕见
    assert 0.35 < from_a < 0.65, from_a
    assert both_far < 0.05, both_far

def test_genome_serialize():
    import json
    g = eco.random_genome("s", np.random.default_rng(3))
    d = {"name": g.name, "layers": [W.tolist() for W in g.layers],
         "readout": g.readout.tolist(),
         "born_gen": g.born_gen, "age": g.age}
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["name"] == g.name
    assert d2["born_gen"] == g.born_gen and d2["age"] == g.age
    assert len(d2["layers"]) == len(g.layers)
    for W2, W in zip(d2["layers"], g.layers):
        assert np.array_equal(np.array(W2, g.layers[0].dtype), W), "layer 数组应往返一致"
    assert np.array_equal(np.array(d2["readout"], g.readout.dtype), g.readout), "readout 数组应往返一致"

def test_forward_shapes_and_deterministic():
    rng = np.random.default_rng(5)
    g = eco.random_genome("f", rng)
    pix = np.zeros((3, 784), np.float32); pix[0, 200:250] = 1.0
    produced, lc, rc = eco.forward(g, pix, np.random.default_rng(5))
    assert produced.shape == (3,) and len(lc) == 1 and lc[0].shape == (3, 100) and rc.shape == (3, 10)
    assert produced.dtype == np.int64 and lc[0].dtype == np.int64
    produced2, _, _ = eco.forward(g, pix, np.random.default_rng(5))
    assert np.array_equal(produced, produced2), "同 seed 应可复现"

def test_forward_firing_sane():
    """对随机 digit 批次：隐藏层应发放（非全零）、产出不应过于集中/未发放过多。"""
    from data_loading import load_mnist
    ti, tl, _, _ = load_mnist()
    rng = np.random.default_rng(1)
    g = eco.random_genome("f2", rng)
    idx = rng.integers(0, len(ti), 40)
    produced, lc, rc = eco.forward(g, ti[idx], np.random.default_rng(2))
    assert sum(int(c.sum()) for c in lc) > 0, "隐藏层整场无发放——动力学哑了"
    none_frac = float((produced == -1).mean())
    assert none_frac < 0.7, f"产出层未发放比例过高 {none_frac:.2f}"
    real = produced[produced != -1]
    assert len(np.unique(real)) >= 3, f"产出数字过于集中: {np.unique(real)}"
    assert (rc.sum(axis=0) > 0).sum() >= 2, f"产出层几乎只有单一通道发放: {rc.sum(axis=0)}"


def test_round_loop_invariants():
    eco_ = eco.Ecosystem(seed=0)
    assert len(eco_.pop) == eco.INIT_POP
    for _round in range(4):
        events, stats = eco_.step_round()
        types = [e["type"] for e in events]
        assert types[0] == "round_begin" and types[-1] == "round_end"
        assert len(eco_.pop) <= eco_.capacity, f"round{_round} 种群 {len(eco_.pop)} 超承载力"
        assert 0.0 <= stats["avg_acc"] <= 1.0
        assert 0.0 <= stats["natural_rate"] <= 1.0
        names = {g.name for g in eco_.pop}
        assert len(names) == len(eco_.pop), "重名"
        for e in events:
            if e["type"] == "org_round":
                assert "produced" in e and "correct" in e and "age" in e
            if e["type"] == "death":
                assert e["cause"] in ("natural", "unnatural")
            if e["type"] == "birth":
                assert len(e["parents"]) == 2
    # 手动喂食
    g0 = eco_.pop[0]
    r = eco_.manual_feed(g0.name, 3)
    assert r["label"] == 3 and r["produced"] in list(range(-1, 10))
    assert len(r["food_pixels"]) == 784 and len(r["readout_counts"]) == 10
    assert len(r["layer_counts"]) == 1 and len(r["layer_counts"][0]) == eco.HIDDEN_SIZE
    # 可复现
    def _run2(seed):
        e = eco.Ecosystem(seed=seed)
        out = []
        for _ in range(2):
            e.step_round()
            out.append(e._round_fingerprint())
        return out
    assert _run2(9) == _run2(9), "同 seed 应可复现"

def test_round_never_exceeds_capacity():
    """多回合后种群应 ≤ 承载力（密度依赖 + 硬上限）。"""
    e = eco.Ecosystem(seed=1)
    for _ in range(8):
        e.step_round()
        assert len(e.pop) <= e.capacity, f"种群 {len(e.pop)} > 承载力 {e.capacity}"
    assert e.total_deaths > 0, "应有死亡记录"
    assert len(e.pop) >= 1, "不应灭绝到 0（全灭重播应恢复）"

def test_reseed_respects_capacity():
    """极端合法配置下，全灭重播种群数不得超承载力。"""
    e = eco.Ecosystem(seed=3)
    e.set_config(capacity=100, initial_pop=1000)   # 两者都是合法滑块值
    e.pop = []                                     # 强制触发全灭重播
    events, stats = e.step_round()
    assert len(e.pop) <= e.capacity, f"重播 {len(e.pop)} > 承载力 {e.capacity}"

def test_weighted_crossover():
    """存活加权交叉：weight_a 越大，后代越接近 a。"""
    rng = np.random.default_rng(7)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child_50 = eco.crossover(a, b, np.random.default_rng(8), weight_a=1, weight_b=1)
    close50 = (np.abs(child_50.layers[0] - a.layers[0]) < np.abs(child_50.layers[0] - b.layers[0])).mean()
    assert 0.35 < close50 < 0.65, close50
    child_90 = eco.crossover(a, b, np.random.default_rng(8), weight_a=9, weight_b=1)
    close90 = (np.abs(child_90.layers[0] - a.layers[0]) < np.abs(child_90.layers[0] - b.layers[0])).mean()
    assert close90 > 0.75, f"weight_a=9 应显著偏向 a: {close90}"

def test_death_cause():
    """死亡分类：错→unnatural；对但超龄→natural；对且未超龄→存活。"""
    g = eco.random_genome("d", np.random.default_rng(0))
    assert eco.death_cause(g, False, 20) == "unnatural"
    g2 = eco.random_genome("d2", np.random.default_rng(1)); g2.age = 20
    assert eco.death_cause(g2, True, 20) == "natural"
    g3 = eco.random_genome("d3", np.random.default_rng(2)); g3.age = 1
    assert eco.death_cause(g3, True, 20) is None


def test_server_endpoints():
    """HTTP 服务端到端：状态/推演/数字图像/手动喂食/配置透传/首页 HTML。

    配置经 POST /api/config 修改后返回完整 config 供前端使用；
    首页由 eco_game.html（Task 6 交付）托管，断言其包含 id="dish" 培养皿网格
    与 <canvas> 统计曲线。
    注：run_server_in_thread 返回 (port, server)（ThreadingHTTPServer），其关闭方式
    为 server.shutdown() + server.server_close()（无 join 方法）。
    """
    import json, urllib.request
    from eco_server import run_server_in_thread, PORT_DEFAULT
    port, server = run_server_in_thread(seed=0)
    base = f"http://127.0.0.1:{port}"
    try:
        s = json.load(urllib.request.urlopen(base + "/api/state"))
        assert s["config"]["survival_rounds"] == eco.SURVIVAL_ROUNDS
        assert len(s["population"]) == eco.INIT_POP
        req = urllib.request.Request(base + "/api/step", method="POST")
        r = json.load(urllib.request.urlopen(req))
        assert r["stats"]["natural_rate"] >= 0.0
        assert r["events"][0]["type"] == "round_begin"
        img = json.load(urllib.request.urlopen(base + "/api/digit_image/0"))
        assert len(img["pixels"]) == 784
        # 回合制下大部分个体当回合死亡，喂食目标须取推演后的存活者
        s2 = json.load(urllib.request.urlopen(base + "/api/state"))
        body = json.dumps({"digit": 4, "name": s2["population"][0]["name"]}).encode()
        rq = urllib.request.Request(base + "/api/manual_feed", data=body, method="POST",
                                    headers={"Content-Type": "application/json"})
        mf = json.load(urllib.request.urlopen(rq))
        assert mf["label"] == 4 and len(mf["readout_counts"]) == 10
        cfg = json.dumps({"n_repro": 60}).encode()
        cr = urllib.request.Request(base + "/api/config", data=cfg, method="POST",
                                    headers={"Content-Type": "application/json"})
        c2 = json.load(urllib.request.urlopen(cr))
        assert c2["config"]["n_repro"] == 60
        html = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert 'id="dish"' in html and "<canvas" in html
    finally:
        server.shutdown()
        server.server_close()


def test_survival_alpha():
    """存活奖励 alpha = 1/每回合存活率：全员存活→1.0；半存活→2.0；0 存活→1.0（防御）。"""
    assert eco.survival_alpha(10, 10) == 1.0
    assert abs(eco.survival_alpha(5, 10) - 2.0) < 1e-9
    assert eco.survival_alpha(0, 10) == 1.0     # 0 存活防御
    assert eco.survival_alpha(3, 0) == 1.0      # 空回合防御


def test_assortative_pairing_tends_similar_age():
    """选型交配：s=1.0 时年龄两极分化的存活者各自聚类配对，平均 |Δage| 显著小于 s=0（随机）。"""
    rng = np.random.default_rng(42)
    survivors = [eco.random_genome(f"lo{i}", rng) for i in range(20)]
    for g in survivors:
        g.age = 1
    survivors += [eco.random_genome(f"hi{i}", rng) for i in range(20)]
    for g in survivors[20:]:
        g.age = 20

    def mean_abs_diff(strength):
        diffs = []
        for s in range(30):
            pairs = eco.assortative_pairs(survivors, strength,
                                          np.random.default_rng(1000 + s))
            diffs += [abs(a.age - b.age) for a, b in pairs]
        return float(np.mean(diffs))

    d_random = mean_abs_diff(0.0)
    d_assort = mean_abs_diff(1.0)
    assert d_assort < d_random * 0.6, f"选型配对平均年龄差 {d_assort:.2f} 应显著小于随机 {d_random:.2f}"


def test_stats_has_survival_rate_alpha():
    """round_end stats 与 /api/state stats 均含 survival_rate/alpha；未取整的 last_* 满足恒等。"""
    e = eco.Ecosystem(seed=2)
    for _ in range(3):
        _, stats = e.step_round()
        assert "survival_rate" in stats and "alpha" in stats
        assert stats["survival_rate"] >= 0.0 and stats["alpha"] >= 1.0
    # 核心恒等：alpha == 1/存活率（用未取整值，取整会引入 ~0.005 误差，故不在 stats 上断言精确恒等）
    if e.last_survival_rate > 0:
        assert abs(e.last_alpha - 1.0 / e.last_survival_rate) < 1e-9
    st = e.get_state()["stats"]
    assert "survival_rate" in st and "alpha" in st


def test_config_assort_strength():
    """assort_strength 可设/可回读；非法值（<0 或 >1）被忽略。"""
    e = eco.Ecosystem(seed=4)
    e.set_config(assort_strength=0.8)
    assert abs(e.assort_strength - 0.8) < 1e-9
    assert abs(e._config()["assort_strength"] - 0.8) < 1e-9
    e.set_config(assort_strength=-0.1)
    assert abs(e.assort_strength - 0.8) < 1e-9   # 非法值忽略
    e.set_config(assort_strength=1.5)
    assert abs(e.assort_strength - 0.8) < 1e-9


def _forward_numpy_reference_multi(genome, pixels, rng):
    """纯 numpy 多层参考实现（对照 numba _forward_core_multi，验证语义一致；仅测试用）。

    语义镜像：累积→不应期清零→漏电→WTA(首个最大者)→发放→不应期递减；逐层 one-hot 前馈。
    """
    B = pixels.shape[0]; T = eco.T
    S = (rng.random((B, 784, T), dtype=np.float32) < (pixels[:, :, None] * eco.SPIKE_GAIN)).astype(np.float32)
    K = len(genome.layers)
    max_n = max(W.shape[1] for W in genome.layers)
    V = np.zeros((K, B, max_n), np.float32)
    ref = np.zeros((K, B, max_n), np.int32)
    cnt = np.zeros((K, B, max_n), np.int64)
    Vr = np.zeros((B, eco.READOUT_SIZE), np.float32)
    refr = np.zeros((B, eco.READOUT_SIZE), np.int32)
    rc = np.zeros((B, eco.READOUT_SIZE), np.int64)
    prev = np.zeros((B, max_n), np.float32)
    for t in range(T):
        for l in range(K):
            W = genome.layers[l]; n_in = W.shape[0]; n_out = W.shape[1]
            inp = S[:, :, t] if l == 0 else prev[:, :n_in]
            Vl = V[l][:, :n_out]
            refl = ref[l][:, :n_out]
            cntl = cnt[l][:, :n_out]
            Vl += inp @ W
            Vl[refl > 0] = 0.0
            Vl *= eco.LEAK
            elig = (refl <= 0) & (Vl >= eco.THETA_HIDDEN)
            fire_rows = np.nonzero(elig.any(axis=1))[0]
            out = np.zeros((B, n_out), np.float32)
            if fire_rows.size:
                win = np.where(elig, Vl, -np.inf).argmax(axis=1)[fire_rows]
                Vl[fire_rows] = 0.0
                was_idle = refl[fire_rows] <= 0
                refl[fire_rows] = np.where(was_idle, 1, refl[fire_rows])
                refl[fire_rows, win] = eco.REF_PERIOD
                out[fire_rows, win] = 1.0
                cntl[fire_rows, win] += 1
            ref[l][:, :n_out] = np.maximum(refl - 1, 0)
            prev[:, :n_out] = out
        # 产出层：末层 one-hot @ readout
        n_k = genome.readout.shape[0]
        Vr += prev[:, :n_k] @ genome.readout
        Vr[refr > 0] = 0.0
        Vr *= eco.LEAK
        eligr = (refr <= 0) & (Vr >= eco.THETA_READOUT)
        fire_r = np.nonzero(eligr.any(axis=1))[0]
        if fire_r.size:
            winr = np.where(eligr, Vr, -np.inf).argmax(axis=1)[fire_r]
            Vr[fire_r] = 0.0
            was_idle_r = refr[fire_r] <= 0
            refr[fire_r] = np.where(was_idle_r, 1, refr[fire_r])
            refr[fire_r, winr] = eco.REF_PERIOD
            rc[fire_r, winr] += 1
        refr = np.maximum(refr - 1, 0)
    produced = np.where(rc.sum(axis=1) > 0, rc.argmax(axis=1), -1)
    layer_counts = [cnt[l][:, :genome.layers[l].shape[1]] for l in range(K)]
    return produced, layer_counts, rc


def test_numba_matches_reference():
    """numba 多层 forward 与纯 numpy 参考在相同泊松脉冲上应给出几乎一致的 produced。

    用同一 rng seed 两次调用保证 S 相同；浮点累加顺序不同允许个别近平局样本翻转，
    但整体一致性须 ≥0.9（抓语义级 bug）。单层基线。
    """
    from data_loading import load_mnist
    ti, tl, _, _ = load_mnist()
    rng = np.random.default_rng(1)
    g = eco.random_genome("f2", rng)
    idx = rng.integers(0, len(ti), 60)
    pix = ti[idx]
    p_num, lc_num, rc_num = eco.forward(g, pix, np.random.default_rng(2))
    p_ref, lc_ref, rc_ref = _forward_numpy_reference_multi(g, pix, np.random.default_rng(2))
    agree = float((p_num == p_ref).mean())
    assert agree >= 0.9, f"numba 与 numpy produced 一致性仅 {agree:.2f}（语义疑似偏离）"
    # 每层发放计数也应一致
    for a, b in zip(lc_num, lc_ref):
        assert int(a.sum()) == int(b.sum()), "隐藏层总发放数不一致"
    assert int(rc_num.sum()) == int(rc_ref.sum()), "产出层总发放数不一致"


def test_multilayer_forward_matches_reference():
    """2/3 层个体：numba _forward_core_multi 与纯 numpy 参考在相同 S 上 produced/发放数一致。"""
    from data_loading import load_mnist
    ti, tl, _, _ = load_mnist()
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(ti), 40)
    pix = ti[idx]
    for arch in ([100], [80, 50], [60, 40, 30]):
        g = eco.Genome(name="m", layers=[
            eco._random_weights(784 if i == 0 else arch[i - 1], n, eco.W_NORM_HIDDEN, rng)
            for i, n in enumerate(arch)],
            readout=eco._random_weights(arch[-1], eco.READOUT_SIZE, eco.W_NORM_READOUT, rng))
        p_num, lc_num, rc_num = eco.forward(g, pix, np.random.default_rng(2))
        p_ref, lc_ref, rc_ref = _forward_numpy_reference_multi(g, pix, np.random.default_rng(2))
        agree = float((p_num == p_ref).mean())
        assert agree >= 0.9, f"arch={arch} 一致性仅 {agree:.2f}（语义疑似偏离）"
        for a, b in zip(lc_num, lc_ref):
            assert int(a.sum()) == int(b.sum()), f"arch={arch} 层发放数不一致"


def test_architecture_inheritance():
    """双亲架构不同：子代架构 = 存活更长亲代的架构（crossover 纯重组，不施加结构突变）。"""
    rng = np.random.default_rng(4)
    a = eco.Genome(name="a", layers=[
        eco._random_weights(784, 100, eco.W_NORM_HIDDEN, rng),
        eco._random_weights(100, 50, eco.W_NORM_HIDDEN, rng)],
        readout=eco._random_weights(50, eco.READOUT_SIZE, eco.W_NORM_READOUT, rng), age=5)
    b = eco.Genome(name="b", layers=[eco._random_weights(784, 80, eco.W_NORM_HIDDEN, rng)],
                   readout=eco._random_weights(80, eco.READOUT_SIZE, eco.W_NORM_READOUT, rng), age=3)
    child = eco.crossover(a, b, np.random.default_rng(9), weight_a=5, weight_b=3)
    assert child.arch() == a.arch(), "存活更长的 a 应提供架构"
    child2 = eco.crossover(a, b, np.random.default_rng(9), weight_a=3, weight_b=5)
    assert child2.arch() == b.arch(), "存活更长的 b 应提供架构"


def test_structural_mutations_bounded():
    """任意结构突变后：1≤层数≤4、每层 20≤n≤200、总隐藏神经元≤400、维度链闭合。"""
    rng = np.random.default_rng(0)
    g = eco.random_genome("s", rng)
    for _ in range(200):
        c = eco._apply_structure(eco.crossover(g, g, rng), rng)
        assert eco.MIN_LAYERS <= len(c.layers) <= eco.MAX_LAYERS, c.arch()
        assert all(eco.MIN_NEURONS <= n <= eco.MAX_NEURONS for n in c.arch()), c.arch()
        assert sum(c.arch()) <= eco.MAX_HIDDEN, c.arch()
        assert c.layers[0].shape[0] == 784, "输入维应 784"
        for W, nxt in zip(c.layers, c.layers[1:] + [c.readout]):
            assert W.shape[1] == nxt.shape[0], f"维度链断裂: {W.shape} → {nxt.shape}"
        assert c.readout.shape[1] == eco.READOUT_SIZE


def test_silent_birth_preserves_output():
    """静默神经元诞生（force="grow"）前后对同 S 的输出分布一致——行为保持是稳定性核心。"""
    import copy
    from data_loading import load_mnist
    ti, _, _, _ = load_mnist()
    rng = np.random.default_rng(1)
    g = eco.random_genome("g", rng)
    S = (np.random.default_rng(2).random((1, 784, eco.T), dtype=np.float32)
         < (ti[5][None][:, :, None] * eco.SPIKE_GAIN)).astype(np.float32)
    p0, lc0, rc0 = eco.forward_from_S(g, S)
    g2 = eco._apply_structure(copy.deepcopy(g), np.random.default_rng(6), force="grow")
    p1, lc1, rc1 = eco.forward_from_S(g2, S)
    assert sum(g2.arch()) > sum(g.arch()), "grow 应增加神经元数"
    # 静默神经元近零 → 不放电 → 输出分布一致
    assert int(rc0.sum()) == int(rc1.sum()), "静默诞生不应改变产出发放"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("ALL TESTS PASSED")
