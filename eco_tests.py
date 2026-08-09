# eco_tests.py
"""LIF 生态引擎测试。运行：python eco_tests.py"""
import numpy as np
import eco_engine as eco

def test_crossover_mixes_both_parents():
    rng = np.random.default_rng(7)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child = eco.crossover(a, b, rng)
    assert child.hidden.shape == (784, 100), child.hidden.shape
    assert child.readout.shape == (100, 10), child.readout.shape
    # 逐权重取父/母：每个权重应"更接近"其中一方（噪声 σ=0.01 远小于双亲差距）
    for Wc, Wa, Wb in [(child.hidden, a.hidden, b.hidden),
                       (child.readout, a.readout, b.readout)]:
        closer_a = (np.abs(Wc - Wa) < np.abs(Wc - Wb)).mean()
        assert 0.35 < closer_a < 0.65, f"closer_a={closer_a}"

def test_columns_normalized():
    g = eco.random_genome("n", np.random.default_rng(0))
    norms = np.linalg.norm(g.hidden, axis=0)
    assert np.allclose(norms, eco.W_NORM_HIDDEN, atol=1e-4), norms[:5]
    rnorms = np.linalg.norm(g.readout, axis=0)
    assert np.allclose(rnorms, eco.W_NORM_READOUT, atol=1e-4), rnorms[:5]

def test_mutation_is_rare():
    rng = np.random.default_rng(11)
    a = eco.random_genome("a", rng)
    b = eco.random_genome("b", rng)
    child = eco.crossover(a, b, rng)
    d_a = np.abs(child.hidden - a.hidden)
    d_b = np.abs(child.hidden - b.hidden)
    from_a = (d_a < d_b).mean()                 # 一半来自 a，一半来自 b
    both_far = ((d_a > 0.3) & (d_b > 0.3)).mean()  # 大突变应极罕见
    assert 0.35 < from_a < 0.65, from_a
    assert both_far < 0.05, both_far

def test_genome_serialize():
    import json
    g = eco.random_genome("s", np.random.default_rng(3))
    d = {"name": g.name, "hidden": g.hidden.tolist(), "readout": g.readout.tolist(),
         "born_gen": g.born_gen, "age": g.age}
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["name"] == g.name
    assert d2["born_gen"] == g.born_gen and d2["age"] == g.age
    assert np.array_equal(np.array(d2["hidden"], g.hidden.dtype), g.hidden), "hidden 数组应往返一致"
    assert np.array_equal(np.array(d2["readout"], g.readout.dtype), g.readout), "readout 数组应往返一致"

def test_forward_shapes_and_deterministic():
    rng = np.random.default_rng(5)
    g = eco.random_genome("f", rng)
    pix = np.zeros((3, 784), np.float32); pix[0, 200:250] = 1.0
    produced, hc, rc = eco.forward(g, pix, np.random.default_rng(5))
    assert produced.shape == (3,) and hc.shape == (3, 100) and rc.shape == (3, 10)
    assert produced.dtype == np.int64 and hc.dtype == np.int64
    produced2, _, _ = eco.forward(g, pix, np.random.default_rng(5))
    assert np.array_equal(produced, produced2), "同 seed 应可复现"

def test_forward_firing_sane():
    """对随机 digit 批次：隐藏层应发放（非全零）、产出不应过于集中/未发放过多。"""
    from data_loading import load_mnist
    ti, tl, _, _ = load_mnist()
    rng = np.random.default_rng(1)
    g = eco.random_genome("f2", rng)
    idx = rng.integers(0, len(ti), 40)
    produced, hc, rc = eco.forward(g, ti[idx], np.random.default_rng(2))
    assert hc.sum() > 0, "隐藏层整场无发放——动力学哑了"
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
    close50 = (np.abs(child_50.hidden - a.hidden) < np.abs(child_50.hidden - b.hidden)).mean()
    assert 0.35 < close50 < 0.65, close50
    child_90 = eco.crossover(a, b, np.random.default_rng(8), weight_a=9, weight_b=1)
    close90 = (np.abs(child_90.hidden - a.hidden) < np.abs(child_90.hidden - b.hidden)).mean()
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


def _forward_numpy_reference(genome, pixels, rng):
    """纯 numpy 参考实现（对照 numba JIT 核心，验证语义一致；仅测试用）。"""
    B = pixels.shape[0]; T = eco.T
    S = (rng.random((B, 784, T), dtype=np.float32) < (pixels[:, :, None] * eco.SPIKE_GAIN)).astype(np.float32)
    Vh = np.zeros((B, eco.HIDDEN_SIZE), np.float32)
    refh = np.zeros((B, eco.HIDDEN_SIZE), np.int32)
    Vr = np.zeros((B, eco.READOUT_SIZE), np.float32)
    refr = np.zeros((B, eco.READOUT_SIZE), np.int32)
    hc = np.zeros((B, eco.HIDDEN_SIZE), np.int64)
    rc = np.zeros((B, eco.READOUT_SIZE), np.int64)
    Wh, Wr = genome.hidden, genome.readout
    for t in range(T):
        Vh += S[:, :, t] @ Wh
        Vh[refh > 0] = 0.0
        Vh *= eco.LEAK
        elig = (refh <= 0) & (Vh >= eco.THETA_HIDDEN)
        fire_rows = np.nonzero(elig.any(axis=1))[0]
        hspk = np.zeros((B, eco.HIDDEN_SIZE), np.float32)
        if fire_rows.size:
            win = np.where(elig, Vh, -np.inf).argmax(axis=1)[fire_rows]
            Vh[fire_rows] = 0.0
            was_idle = refh[fire_rows] <= 0
            refh[fire_rows] = np.where(was_idle, 1, refh[fire_rows])
            refh[fire_rows, win] = eco.REF_PERIOD
            hspk[fire_rows, win] = 1.0
            hc[fire_rows, win] += 1
        refh = np.maximum(refh - 1, 0)
        Vr += hspk @ Wr
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
    return produced, hc, rc


def test_numba_matches_reference():
    """numba forward 与纯 numpy 参考在相同泊松脉冲上应给出几乎一致的 produced。

    用同一 rng seed 两次调用保证 S 相同；浮点累加顺序不同允许个别近平局样本翻转，
    但整体一致性须 ≥0.9（抓语义级 bug）。
    """
    from data_loading import load_mnist
    ti, tl, _, _ = load_mnist()
    rng = np.random.default_rng(1)
    g = eco.random_genome("f2", rng)
    idx = rng.integers(0, len(ti), 60)
    pix = ti[idx]
    p_num, hc_num, rc_num = eco.forward(g, pix, np.random.default_rng(2))
    p_ref, hc_ref, rc_ref = _forward_numpy_reference(g, pix, np.random.default_rng(2))
    agree = float((p_num == p_ref).mean())
    assert agree >= 0.9, f"numba 与 numpy produced 一致性仅 {agree:.2f}（语义疑似偏离）"
    # 发放计数也应一致（hidden/readout 总发放数相同）
    assert int(hc_num.sum()) == int(hc_ref.sum()), "隐藏层总发放数不一致"
    assert int(rc_num.sum()) == int(rc_ref.sum()), "产出层总发放数不一致"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("ALL TESTS PASSED")
