# eco_engine.py（本任务先写：常量、Genome、normalize、random_genome、crossover）
"""LIF 生态游戏引擎（v2 回合制）：喂食-产出-淘汰-有性繁殖（纯 numpy，一生无学习）。

生物体 = LIF 网络（784 输入 → 100 隐藏神经元 → 10 产出神经元），所有权重
出生即随机、一生固定。一回合喂 1 个数字：产出错误的当回合死亡（非自然），
活到存活回合数上限自然死亡；存活者按年龄相似度选型配对（可调强度 s）、
按存活时长加权交叉繁殖，产仔数 × 存活奖励 alpha（1/每回合存活率），
密度依赖 + 承载力封顶。输出标准化事件流供前端播放。
"""
from __future__ import annotations

import numpy as np
import numba
from dataclasses import dataclass

# ---- 游戏参数（v2 回合制） ----
T = 40                    # 仿真步数。v2 加速：原 120 时满容量(500)回合 ~4.9s，降为 40 后 ~2.4s（~2×），
                          # 正确率仍贴随机线、None 率<0.7、12 测试全绿；T 越长单回合越慢（每只 ~0.06ms/步）
SPIKE_GAIN = 0.6
LEAK = 0.94
HIDDEN_SIZE = 100
READOUT_SIZE = 10
REF_PERIOD = 4
THETA_HIDDEN = 12.0
THETA_READOUT = 1.5
W_NORM_HIDDEN = 16.0
W_NORM_READOUT = 3.0
W_INIT_RANGE = 0.2
CROSS_SIGMA = 0.01       # 有性繁殖高斯扰动 σ
MUT_RATE = 0.001         # 千分之一大突变
SURVIVAL_ROUNDS = 20     # 自然寿命上限（回合）
N_REPRO = 50             # 每对每次繁殖数量 = 存活回合数 × N
ASSORT_TAU = 2.0         # 选型交配：年龄相似度核宽（回合），固定不可调
ASSORT_STRENGTH = 0.5    # 默认选型强度 s（0-1）
CAPACITY = 500           # 环境承载力（种群上限；满容量实测均值约 4.5s、峰值约 5.5s）
DENSITY_FLOOR = 0.05     # 密度地板：承载力处仍有 5% 替代性繁殖，防"满→90%暴毙→回填"锯齿
INIT_POP = 60            # 初始/全灭重播种群数

# ---- 多层架构边界（v3 结构突变） ----
MIN_LAYERS = 1           # 最小隐藏层数
MAX_LAYERS = 4           # 最大隐藏层数（保回合耗时）
MIN_NEURONS = 20         # 每层最小神经元数
MAX_NEURONS = 200        # 每层最大神经元数
MAX_HIDDEN = 400         # 全部隐藏神经元硬上限（保 forward 耗时）
NORM_ACTIVE_EPS = 1.0    # 列 L2 范数 ≥1.0 才归一化；以下视为静默（保持近零不放电）


@dataclass
class Genome:
    """一个生物体：变深度架构（1-4 隐藏层，初始单层），权重即基因。"""
    name: str
    layers: list[np.ndarray]   # 隐藏层权重 [(784,n1),(n1,n2),…]，每层 (n_in,n_out)
    readout: np.ndarray        # (n_k, 10)  末隐藏层→产出
    born_gen: int = 0
    age: int = 0
    parents: tuple | None = None

    def arch(self) -> list[int]:
        """各隐藏层神经元数（架构指纹，前端解剖用）。"""
        return [int(W.shape[1]) for W in self.layers]


@numba.njit(cache=True)
def _normalize_cols(W: np.ndarray, norm: float) -> np.ndarray:
    """每列归一化到指定 L2 范数；列范数 < NORM_ACTIVE_EPS 的静默列跳过（保持近零）。

    静默列（结构突变新生的近零权重）不归一化 → 不放大 → 不放电，行为保持；
    活跃列归一化保证动力学不爆/不哑。
    """
    out = W.copy()
    nrow, ncol = out.shape
    eps2 = NORM_ACTIVE_EPS * NORM_ACTIVE_EPS
    for j in range(ncol):
        s = 0.0
        for i in range(nrow):
            s += out[i, j] * out[i, j]
        if s < eps2:
            continue
        inv = norm / (np.sqrt(s) + 1e-8)
        for i in range(nrow):
            out[i, j] *= inv
    return out


def _random_weights(n_in: int, n_out: int, norm: float, rng: np.random.Generator) -> np.ndarray:
    W = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, (n_in, n_out))
    return _normalize_cols(W, norm)


def random_genome(name: str, rng: np.random.Generator, gen: int = 0) -> Genome:
    """初始/全灭重播个体：单层 784→100→10（多层只由结构突变产生）。"""
    return Genome(
        name=name,
        layers=[_random_weights(784, HIDDEN_SIZE, W_NORM_HIDDEN, rng)],
        readout=_random_weights(HIDDEN_SIZE, READOUT_SIZE, W_NORM_READOUT, rng),
        born_gen=gen,
    )


def crossover(a: Genome, b: Genome, rng: np.random.Generator,
              weight_a: float = 1.0, weight_b: float = 1.0) -> Genome:
    """有性繁殖（纯重组，结构突变由 _apply_structure 单独施加）。

    架构继承：存活更长者（weight 更大）提供子代架构；同寿掷硬币。
    权重混合：仅当另一亲代同位置层形状兼容才做逐权重 pa 取 a + 均匀小扰动 + 千分之一大突变；
    不兼容层取架构父原层（形状确定，绝无维度错配）。readout 同理。
    """
    if weight_a > weight_b:
        arch_parent, other = a, b
    elif weight_b > weight_a:
        arch_parent, other = b, a
    else:
        arch_parent, other = (a, b) if rng.random() < 0.5 else (b, a)
    pa = weight_a / (weight_a + weight_b) if (weight_a + weight_b) > 0 else 0.5

    layers = []
    for i, W in enumerate(arch_parent.layers):
        oW = other.layers[i] if i < len(other.layers) else None
        if oW is not None and oW.shape == W.shape:
            Wc = np.where(rng.random(W.shape) < pa, W, oW)
            Wc += rng.uniform(-CROSS_SIGMA, CROSS_SIGMA, Wc.shape)
            m = rng.random(Wc.shape) < MUT_RATE
            Wc[m] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(m.sum()))
            layers.append(_normalize_cols(Wc, W_NORM_HIDDEN))
        else:
            layers.append(W.copy())              # 形状不兼容 → 架构父原层

    if other.readout.shape == arch_parent.readout.shape:
        readout = np.where(rng.random(arch_parent.readout.shape) < pa,
                           arch_parent.readout, other.readout)
        readout += rng.uniform(-CROSS_SIGMA, CROSS_SIGMA, readout.shape)
        mr = rng.random(readout.shape) < MUT_RATE
        readout[mr] = rng.uniform(-W_INIT_RANGE, W_INIT_RANGE, int(mr.sum()))
        readout = _normalize_cols(readout, W_NORM_READOUT)
    else:
        readout = arch_parent.readout.copy()

    return Genome(name="child", layers=layers, readout=readout,
                  parents=(a.name, b.name))


# ---- 结构突变（多层架构：五类保函数突变 + 边界钳制） ----
P_GROW = 0.40            # ① 静默神经元诞生（近零起始，行为不变）
P_SPLIT = 0.15           # ② 层复制/恒等中继插入（透明）
P_MERGE = 0.10           # ③ 层合并（权重复合）
P_PRUNE = 0.10           # ④ 神经元剪枝（删不发火/幅值最小者）
P_ADDRANDOM = 0.03       # ⑤ 随机整层插入（大跳变，最稀有）
_SILENT_RANGE = 0.02     # 静默神经元初始权重幅度（近零，不归一化）


def _total_hidden(g: Genome) -> int:
    return sum(g.arch())


def _grow(g: Genome, rng: np.random.Generator) -> Genome:
    """① 静默神经元诞生：某层扩 +d 列（近零），下一层/readout 扩 +d 行。行为近似不变。"""
    if _total_hidden(g) >= MAX_HIDDEN:
        return g
    idx = int(rng.integers(0, len(g.layers)))
    n_out = g.layers[idx].shape[1]
    d = int(rng.integers(1, min(20, MAX_NEURONS - n_out) + 1))
    if d <= 0:
        return g
    w_new = rng.uniform(-_SILENT_RANGE, _SILENT_RANGE, (g.layers[idx].shape[0], d))
    g.layers[idx] = np.concatenate([g.layers[idx], w_new], axis=1)
    if idx + 1 < len(g.layers):
        nxt = g.layers[idx + 1]
        r_new = rng.uniform(-_SILENT_RANGE, _SILENT_RANGE, (d, nxt.shape[1]))
        g.layers[idx + 1] = np.concatenate([nxt, r_new], axis=0)
    else:
        g.readout = np.concatenate([g.readout, rng.uniform(-_SILENT_RANGE, _SILENT_RANGE, (d, g.readout.shape[1]))], axis=0)
    return g


def _split(g: Genome, rng: np.random.Generator) -> Genome:
    """② 层复制/恒等中继：某层后插入 (n_out→n_out) 恒等层（对角归一化到 W_NORM_HIDDEN → 透明中继）。"""
    if len(g.layers) >= MAX_LAYERS or _total_hidden(g) >= MAX_HIDDEN:
        return g
    idx = int(rng.integers(0, len(g.layers)))
    n_out = g.layers[idx].shape[1]
    W2 = np.eye(n_out, dtype=np.float64) + rng.uniform(-0.05, 0.05, (n_out, n_out))
    W2 = _normalize_cols(W2, W_NORM_HIDDEN)      # 对角≈16 → 单脉冲可达阈值 → 中继透明
    g.layers.insert(idx + 1, W2)
    return g


def _merge(g: Genome, rng: np.random.Generator) -> Genome:
    """③ 层合并：相邻 W1(a→b)、W2(b→c) 合成 W=W2@W1(a→c)，删中间 LIF。"""
    if len(g.layers) < 2:
        return g
    idx = int(rng.integers(0, len(g.layers) - 1))
    W1, W2 = g.layers[idx], g.layers[idx + 1]
    W = W1 @ W2          # 组合：(a→b) 后接 (b→c) ⇒ a→c（x@W1 再 x@W2 = x@(W1@W2)）
    if W.shape[1] > MAX_NEURONS or _total_hidden(g) - W1.shape[1] > MAX_HIDDEN:
        return g                              # 合并可能增大末层宽/总神经元 → 钳制
    g.layers[idx:idx + 2] = [_normalize_cols(W, W_NORM_HIDDEN)]
    return g


def _prune(g: Genome, rng: np.random.Generator) -> Genome:
    """④ 神经元剪枝：删某层 L2 范数最小的 d 列 + 下一层对应行（压缩，常有益）。"""
    idx = int(rng.integers(0, len(g.layers)))
    n_out = g.layers[idx].shape[1]
    if n_out <= MIN_NEURONS:
        return g
    d = int(rng.integers(1, min(5, n_out - MIN_NEURONS) + 1))
    col_norms = np.linalg.norm(g.layers[idx], axis=0)
    drop = np.argsort(col_norms)[:d]
    keep = np.setdiff1d(np.arange(n_out), drop)
    g.layers[idx] = g.layers[idx][:, keep]
    if idx + 1 < len(g.layers):
        # [keep, :] 可能产生非 C 连续数组（numba typed.List 元素必须 C 布局）
        g.layers[idx + 1] = np.ascontiguousarray(g.layers[idx + 1][keep, :])
    else:
        g.readout = np.ascontiguousarray(g.readout[keep, :])
    return g


def _add_random(g: Genome, rng: np.random.Generator) -> Genome:
    """⑤ 随机整层插入：尾部新增随机隐藏层 (n_k→n_out) + 重初始化 readout（大跳变）。"""
    if len(g.layers) >= MAX_LAYERS or _total_hidden(g) >= MAX_HIDDEN:
        return g
    n_in = g.layers[-1].shape[1]
    n_out = int(rng.integers(MIN_NEURONS, MAX_NEURONS + 1))
    if _total_hidden(g) + n_out > MAX_HIDDEN:
        n_out = MAX_HIDDEN - _total_hidden(g)
    if n_out < MIN_NEURONS:
        return g
    new_layer = _random_weights(n_in, n_out, W_NORM_HIDDEN, rng)
    g.layers.append(new_layer)
    g.readout = _random_weights(n_out, READOUT_SIZE, W_NORM_READOUT, rng)
    return g


def _apply_structure(g: Genome, rng: np.random.Generator, force: str | None = None) -> Genome:
    """施加结构突变（五类，按概率独立判定；force 用于测试强制指定类型）。"""
    if force is not None:
        return {"grow": _grow, "split": _split, "merge": _merge,
                "prune": _prune, "add_random": _add_random}[force](g, rng)
    if rng.random() < P_GROW:
        g = _grow(g, rng)
    if rng.random() < P_SPLIT:
        g = _split(g, rng)
    if rng.random() < P_MERGE:
        g = _merge(g, rng)
    if rng.random() < P_PRUNE:
        g = _prune(g, rng)
    if rng.random() < P_ADDRANDOM:
        g = _add_random(g, rng)
    return g


def survival_alpha(n_survivors: int, n_start: int) -> float:
    """存活奖励 alpha = 1 / 每回合存活率（种群级）。

    存活率 = 本回合存活者 / 回合开始种群数。全员存活→1.0；0 存活→1.0（防御）。
    """
    if n_start <= 0:
        return 1.0
    sr = n_survivors / n_start
    return 1.0 / sr if sr > 0 else 1.0


def assortative_pairs(survivors: list[Genome], strength: float,
                      rng: np.random.Generator, tau: float = ASSORT_TAU
                      ) -> list[tuple[Genome, Genome]]:
    """选型交配：年龄相似度加权配对。

    随机序取第一只 a；第二只从剩余存活者中按 w=(1-strength)+strength·exp(-|Δage|/tau)
    加权采样。strength=0 → 权重恒等 → 均匀随机配对（统计等价原 shuffle+相邻）；
    strength=1 → 纯按年龄相似度（高龄与高龄、低龄与低龄聚类）。奇数存活者最后一只不配对。
    """
    pool = list(survivors)
    rng.shuffle(pool)
    pairs = []
    while len(pool) >= 2:
        a = pool[0]
        rest = pool[1:]                                # 切片即排除 a
        d = np.abs(np.array([g.age for g in rest], dtype=float) - a.age)
        w = (1.0 - strength) + strength * np.exp(-d / tau)
        b_idx = int(rng.choice(len(rest), p=w / w.sum()))
        b = rest.pop(b_idx)                            # 从 rest 移除 b
        pairs.append((a, b))
        pool = rest
    return pairs


def death_cause(g: Genome, correct: bool, survival_rounds: int) -> str | None:
    """返回本回合的死亡原因：'natural' | 'unnatural' | None（存活）。"""
    g.age += 1
    if not correct:
        return "unnatural"
    if g.age > survival_rounds:
        return "natural"
    return None


@numba.njit(cache=True)
def _forward_core_multi(S: np.ndarray, layers: numba.typed.List, Wr: np.ndarray,
                        max_n: int, leak: float, theta_h: float, theta_r: float,
                        ref_period: int, n_t: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """numba JIT 版多层 LIF 前向核心：逐 b 逐 t 逐层放电。

    S:(B,784,T) float32 泊松脉冲；layers: 变长 typed.List[(n_in,n_out) float64]；
    Wr:(n_k,10) float64；max_n = max(layers[l].shape[1]) 供缓冲区分配。
    每层：稀疏累加 → 不应期清零 → 漏电 → WTA(首个最大者) → 发放 → 不应期递减；
    层 l 的 one-hot 作为层 l+1 输入；末层 one-hot 进产出层（同 θ_r）。
    返回 (produced[B] int64, cnt(K,B,max_n) int64, rc(B,10) int64)。
    """
    B = S.shape[0]
    K = len(layers)
    n_in0 = S.shape[1]
    n_r = Wr.shape[1]
    V = np.zeros((K, B, max_n), dtype=np.float32)
    ref = np.zeros((K, B, max_n), dtype=np.int32)
    cnt = np.zeros((K, B, max_n), dtype=np.int64)
    Vr = np.zeros((B, n_r), dtype=np.float32)
    refr = np.zeros((B, n_r), dtype=np.int32)
    rc = np.zeros((B, n_r), dtype=np.int64)
    prev = np.zeros((B, max_n), dtype=np.float32)   # 上一层 one-hot 缓冲
    for t in range(n_t):
        for l in range(K):
            W = layers[l]
            n_out = W.shape[1]
            for b in range(B):
                row = np.zeros(n_out, dtype=np.float64)
                if l == 0:
                    for i in range(n_in0):
                        if S[b, i, t] > 0.0:
                            for j in range(n_out):
                                row[j] += W[i, j]
                else:
                    for i in range(W.shape[0]):      # 本层输入维 = 上一层 n_out
                        if prev[b, i] > 0.0:
                            for j in range(n_out):
                                row[j] += W[i, j]
                for j in range(n_out):
                    V[l, b, j] += row[j]
                    if ref[l, b, j] > 0:
                        V[l, b, j] = 0.0
                    V[l, b, j] *= leak
                best = -1
                best_v = -1.0e30
                for j in range(n_out):
                    if ref[l, b, j] <= 0 and V[l, b, j] >= theta_h and V[l, b, j] > best_v:
                        best_v = V[l, b, j]
                        best = j
                for j in range(max_n):
                    prev[b, j] = 0.0
                if best >= 0:
                    for j in range(n_out):
                        V[l, b, j] = 0.0
                        if ref[l, b, j] <= 0:
                            ref[l, b, j] = 1
                    ref[l, b, best] = ref_period
                    cnt[l, b, best] += 1
                    prev[b, best] = 1.0
                for j in range(n_out):
                    if ref[l, b, j] > 0:
                        ref[l, b, j] -= 1
        # 产出层：末层 one-hot（prev）→ 积分 + WTA
        for b in range(B):
            for k in range(n_r):
                for i in range(Wr.shape[0]):          # 末层输出维 = readout 行数
                    if prev[b, i] > 0.0:
                        Vr[b, k] += Wr[i, k]
            for k in range(n_r):
                if refr[b, k] > 0:
                    Vr[b, k] = 0.0
                Vr[b, k] *= leak
            best_r = -1
            best_vr = -1.0e30
            for k in range(n_r):
                if refr[b, k] <= 0 and Vr[b, k] >= theta_r and Vr[b, k] > best_vr:
                    best_vr = Vr[b, k]
                    best_r = k
            if best_r >= 0:
                for k in range(n_r):
                    Vr[b, k] = 0.0
                    if refr[b, k] <= 0:
                        refr[b, k] = 1
                refr[b, best_r] = ref_period
                rc[b, best_r] += 1
            for k in range(n_r):
                if refr[b, k] > 0:
                    refr[b, k] -= 1
    produced = np.empty(B, dtype=np.int64)
    for b in range(B):
        total = 0
        bestp = -1
        bestc = -1
        for k in range(n_r):
            total += rc[b, k]
            if rc[b, k] > bestc:
                bestc = rc[b, k]
                bestp = k
        produced[b] = bestp if total > 0 else -1
    return produced, cnt, rc


def forward(genome: Genome, pixels: np.ndarray, rng: np.random.Generator
            ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """pixels: (B,784) ∈[0,1]。返回 (produced, layer_counts, readout_counts)。

    泊松编码（numpy 生成脉冲）→ numba JIT 多层核心（逐层 LIF+WTA 无学习 → 产出层 LIF）。
    produced = 产出层累计发放最多的数字；-1 表示整场未发放。
    """
    B = pixels.shape[0]
    S = (rng.random((B, 784, T), dtype=np.float32) < (pixels[:, :, None] * SPIKE_GAIN)).astype(np.float32)
    return forward_from_S(genome, S)


def forward_from_S(genome: Genome, S: np.ndarray
                   ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """复用已生成的泊松脉冲 S 做前向（回合内所有个体共享同一 S，省去逐只生成脉冲）。

    与 forward 等价，只是脉冲由调用方一次生成。S:(B,784,T) float32。
    """
    layers = numba.typed.List([np.ascontiguousarray(W) for W in genome.layers])
    max_n = max(W.shape[1] for W in genome.layers)
    produced, cnt, rc = _forward_core_multi(S, layers, genome.readout, max_n,
                                            LEAK, THETA_HIDDEN, THETA_READOUT, REF_PERIOD, T)
    layer_counts = [cnt[l][:, :genome.layers[l].shape[1]] for l in range(len(genome.layers))]
    return produced, layer_counts, rc


# ---- 生态主循环（v2 回合制）----
from data_loading import load_mnist


class Ecosystem:
    """回合制生态主循环。同 seed 全程可复现。"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.counter = 0
        self.round = 0
        # 可调参数（前端经 /api/config 修改）
        self.survival_rounds = SURVIVAL_ROUNDS
        self.n_repro = N_REPRO
        self.capacity = CAPACITY
        self.initial_pop = INIT_POP
        self.assort_strength = ASSORT_STRENGTH
        # 上一回合观测（前端 /api/state 展示）
        self.last_survival_rate = 0.0
        self.last_alpha = 1.0
        # 停止条件计数
        self.natural_deaths = 0
        self.total_deaths = 0
        self.stopped = False
        self._img, self._lbl, _, _ = load_mnist()
        n = min(self.initial_pop, self.capacity)   # 初始种群不得超承载力（滑块互相独立）
        self.pop: list[Genome] = [
            random_genome(f"eco#{i}", self.rng, gen=0) for i in range(n)
        ]
        self.counter = n

    def set_config(self, **kw) -> dict:
        """更新可调参数（前端调用）。合法值直接写入；非法忽略。"""
        if "survival_rounds" in kw and 10 <= kw["survival_rounds"] <= 30:
            self.survival_rounds = int(kw["survival_rounds"])
        if "n_repro" in kw and 10 <= kw["n_repro"] <= 100:
            self.n_repro = int(kw["n_repro"])
        if "capacity" in kw and 100 <= kw["capacity"] <= 5000:
            self.capacity = int(kw["capacity"])
        if "initial_pop" in kw and 60 <= kw["initial_pop"] <= 1000:
            self.initial_pop = int(kw["initial_pop"])
            # v3：无强制回填下种群 ≈ 初始种群规模，改动直接增补/截断当前种群
            cur = len(self.pop)
            if cur < self.initial_pop:
                for _ in range(self.initial_pop - cur):
                    self.pop.append(random_genome(self._next_name(), self.rng, gen=self.round))
            elif cur > self.initial_pop:
                self.pop = self.pop[: self.initial_pop]
        if "assort_strength" in kw and 0.0 <= kw["assort_strength"] <= 1.0:
            self.assort_strength = float(kw["assort_strength"])
        return self._config()

    def _config(self) -> dict:
        return {"survival_rounds": self.survival_rounds, "n_repro": self.n_repro,
                "assort_strength": self.assort_strength,
                "capacity": self.capacity, "initial_pop": self.initial_pop}

    def _next_name(self) -> str:
        n = f"eco#{self.counter}"
        self.counter += 1
        return n

    def _round_fingerprint(self) -> tuple:
        return (round(float(self.natural_deaths) / max(1, self.total_deaths), 6),
                len(self.pop), tuple(sorted(g.name for g in self.pop)))

    def step_round(self) -> tuple[list[dict], dict]:
        events: list[dict] = []
        food_idx = int(self.rng.integers(0, len(self._img)))
        food_lbl = int(self._lbl[food_idx])
        events.append({"type": "round_begin", "round": self.round,
                       "food_idx": food_idx, "food_label": food_lbl})

        survivors: list[Genome] = []
        avg_correct = 0.0
        # 每回合只生成一份泊松脉冲 S，全部个体共享（省 5000×0.27ms 的逐只脉冲生成；确定性由 seed 派生保证）
        food_pix = self._img[food_idx][None]
        S = (np.random.default_rng(self.round * 1_000_003).random((1, 784, T), dtype=np.float32)
             < (food_pix[:, :, None] * SPIKE_GAIN)).astype(np.float32)
        accs: dict[str, float] = {}
        for i, g in enumerate(self.pop):
            produced, _hc, rc = forward_from_S(g, S)
            produced = int(produced[0])
            correct = (produced == food_lbl)
            g.age += 1
            events.append({"type": "org_round", "name": g.name, "produced": produced,
                           "correct": bool(correct), "age": g.age,
                           "readout_profile": rc.mean(axis=0).round(2).tolist()})
            avg_correct += float(correct)
            accs[g.name] = 1.0 if correct else 0.0
        avg_correct /= max(1, len(self.pop))

        # ---- 淘汰规则（v3）：按正确率排淘汰最差 30%（非自然，不计停止指标）+ 顶部 70% 中 age>存活回合数自然死亡 ----
        order = sorted(self.pop, key=lambda g: -accs[g.name])
        n_keep = max(1, round(len(order) * 0.70))                 # 存活顶部 70%
        bottom_ids = {id(g) for g in order[n_keep:]}              # 最差 30% 非自然死亡
        aged_ids = {id(g) for g in order[:n_keep] if g.age > self.survival_rounds}  # 顶部高龄自然死亡
        survivors = [g for g in self.pop if id(g) not in bottom_ids and id(g) not in aged_ids]
        for g in self.pop:
            if id(g) in bottom_ids:
                self.total_deaths += 1
                events.append({"type": "death", "name": g.name, "cause": "unnatural"})
            elif id(g) in aged_ids:
                self.natural_deaths += 1
                self.total_deaths += 1
                events.append({"type": "death", "name": g.name, "cause": "natural"})

        # ---- 存活奖励 alpha：存活率 = 本回合存活者 / 回合开始种群 ----
        start_pop = len(self.pop)
        alpha = survival_alpha(len(survivors), start_pop)
        self.last_survival_rate = (len(survivors) / start_pop) if start_pop > 0 else 0.0
        self.last_alpha = alpha

        # ---- 选型交配繁殖（随机两两 + 存活加权交叉 + 密度）----
        pairs = assortative_pairs(survivors, self.assort_strength, self.rng)
        births: list[Genome] = []
        if pairs:
            density = max(DENSITY_FLOOR, 1.0 - start_pop / self.capacity)
            brood = int(round(self.survival_rounds * self.n_repro * alpha * density))
            # v3 选项 2：不强制回填到满容量——只生到能覆盖本回合死亡的量（种群保持当前规模），
            # 不再 min(承载力-存活者) 让种群一回合跳到承载力（那是 5000 大种群回合变慢的根源）。
            room = max(0, start_pop - len(survivors))
            target = min(room, len(pairs) * brood)
            for k in range(target):
                a, b = pairs[k % len(pairs)]
                child = _apply_structure(crossover(a, b, self.rng,
                                                   weight_a=float(a.age), weight_b=float(b.age)),
                                         self.rng)
                child.name = self._next_name()
                child.born_gen = self.round
                births.append(child)
        survivors.extend(births)
        self.pop = survivors
        for c in births:
            events.append({"type": "birth", "name": c.name,
                           "parents": [c.parents[0], c.parents[1]], "gen": c.born_gen})

        # ---- 全灭重播 ----
        if not self.pop:
            n = min(self.initial_pop, self.capacity)   # 重播种群数同样受承载力封顶
            self.pop = [random_genome(f"eco#{self.counter + i}", self.rng, gen=self.round)
                        for i in range(n)]
            self.counter += n
            events.append({"type": "reseed", "count": n})

        # ---- 停止条件：累计自然死亡 / 累计总死亡 ≥ 95% ----
        natural_rate = (self.natural_deaths / self.total_deaths) if self.total_deaths > 0 else 0.0
        self.stopped = self.stopped or (natural_rate >= 0.95)
        stats = {"round": self.round, "alive": len(self.pop),
                 "avg_acc": round(avg_correct, 4),
                 "survival_rate": round(self.last_survival_rate, 4),
                 "alpha": round(self.last_alpha, 3),
                 "natural_deaths": self.natural_deaths,
                 "total_deaths": self.total_deaths,
                 "natural_rate": round(natural_rate, 4),
                 "stopped": self.stopped}
        events.append({"type": "round_end", "stats": stats})
        self.round += 1
        return events, stats

    def get_state(self) -> dict:
        return {"round": self.round, "config": self._config(),
                "population": [{"name": g.name, "age": g.age, "born_gen": g.born_gen,
                                "parents": list(g.parents) if g.parents else None,
                                "alive": True} for g in self.pop],
                "stats": self._last_stats()}

    def _last_stats(self) -> dict:
        natural_rate = (self.natural_deaths / self.total_deaths) if self.total_deaths > 0 else 0.0
        return {"round": self.round, "alive": len(self.pop),
                "survival_rate": self.last_survival_rate,
                "alpha": self.last_alpha,
                "natural_deaths": self.natural_deaths,
                "total_deaths": self.total_deaths,
                "natural_rate": round(natural_rate, 4),
                "stopped": self.stopped}

    def get_digit_image(self, idx: int) -> dict:
        return {"pixels": self._img[idx].tolist(), "label": int(self._lbl[idx])}

    def manual_feed(self, name: str, digit: int) -> dict:
        g = next(x for x in self.pop if x.name == name)
        cand = np.nonzero(self._lbl == digit)[0]
        idx = int(self.rng.choice(cand))
        produced, layer_counts, rc = forward(g, self._img[idx][None],
                                             np.random.default_rng(int(self.rng.integers(0, 2**31))))
        return {"food_pixels": self._img[idx].tolist(), "label": digit,
                "produced": int(produced[0]),
                "correct": bool(produced[0] == digit),
                "layer_counts": [c[0].tolist() for c in layer_counts],
                "readout_counts": rc[0].tolist()}
