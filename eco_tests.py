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
    g = eco.random_genome("s", np.random.default_rng(3))
    d = {"name": g.name, "hidden": g.hidden.tolist(), "readout": g.readout.tolist(),
         "born_gen": g.born_gen, "age": g.age}
    import json
    s = json.dumps(d)
    assert "hidden" in s and "readout" in s

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


def test_day_loop_invariants():
    eco_ = eco.Ecosystem(seed=0)
    assert len(eco_.pop) == eco.INIT_POP
    for day in range(3):
        events, stats = eco_.step_day()
        types = [e["type"] for e in events]
        assert types[0] == "day_begin" and types[-1] == "day_end"
        assert len(eco_.pop) == eco.POP_CAP, f"day{day} 种群 {len(eco_.pop)}"
        assert 0.0 <= stats["avg_acc"] <= 1.0
        assert stats["alive"] == eco.POP_CAP
        names = {g.name for g in eco_.pop}
        assert len(names) == eco.POP_CAP, "重名"
        for e in events:
            if e["type"] == "org_day":
                assert len(e["produced"]) == eco.FOOD_COUNT
                assert len(e["readout_profile"]) == eco.READOUT_SIZE
            if e["type"] == "birth":
                assert len(e["parents"]) == 2
    # 手动喂食
    best = max(eco_.pop, key=lambda g: eco_.stats_cache.get(g.name, 0))
    r = eco_.manual_feed(best.name, 3)
    assert r["label"] == 3 and r["produced"] in list(range(-1, 10))
    assert len(r["food_pixels"]) == 784 and len(r["readout_counts"]) == 10
    # 可复现：同 seed 重建，前 2 天轨迹应一致（_day_fingerprint 记录 avg 与种群名）
    def _run2(seed):
        e = eco.Ecosystem(seed=seed)
        out = []
        for _ in range(2):
            e.step_day()
            out.append(e._day_fingerprint())
        return out
    assert _run2(9) == _run2(9), "同 seed 应可复现"


def test_server_endpoints():
    """HTTP 服务端到端：状态/推演/数字图像/手动喂食/首页 500 兜底。

    注：eco_game.html 由 Task 6 交付，本任务中 GET / 应返回 500 JSON 错误；
    eco_game.html 就位后，此断言应改回校验 HTML 内容（含 <canvas> 或 id="dish"）。
    注：run_server_in_thread 返回 (port, server)（ThreadingHTTPServer），其关闭方式
    为 server.shutdown() + server.server_close()（无 join 方法）。
    """
    import threading, json, urllib.request, urllib.error
    from eco_server import run_server_in_thread, PORT_DEFAULT
    port, server = run_server_in_thread(seed=0)
    base = f"http://127.0.0.1:{port}"
    try:
        s = json.load(urllib.request.urlopen(base + "/api/state"))
        assert s["config"]["pop_cap"] == eco.POP_CAP
        assert len(s["population"]) == eco.INIT_POP
        req = urllib.request.Request(base + "/api/step", method="POST")
        r = json.load(urllib.request.urlopen(req))
        assert r["stats"]["alive"] == eco.POP_CAP
        assert r["events"][0]["type"] == "day_begin"
        img = json.load(urllib.request.urlopen(base + "/api/digit_image/0"))
        assert len(img["pixels"]) == 784
        body = json.dumps({"digit": 4, "name": s["population"][0]["name"]}).encode()
        rq = urllib.request.Request(base + "/api/manual_feed", data=body, method="POST",
                                    headers={"Content-Type": "application/json"})
        mf = json.load(urllib.request.urlopen(rq))
        assert mf["label"] == 4 and len(mf["readout_counts"]) == 10
        try:  # eco_game.html 尚不存在（Task 6 交付）→ 应返回 500 JSON 错误
            urllib.request.urlopen(base + "/")
            raise AssertionError("eco_game.html 缺失时 GET / 应返回 500")
        except urllib.error.HTTPError as e:
            assert e.code == 500, e.code
            assert "error" in json.loads(e.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("ALL TESTS PASSED")
