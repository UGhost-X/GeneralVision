"""LIF 脉冲神经网络 + STDP 学习（PyTorch 忠实移植自 snn_demo.html）。

参考 Diehl & Cook 2015 的单层无监督 STDP 架构，并扩展为可堆叠多层（前馈，
同一步 t 内逐层前向：层 l 的输入 = 层 l-1 在同一步 t 的输出脉冲）。

默认超参移植自 snn_demo.html（LEAK=0.94, THETA0=15, REFR=4, TAU_PLUS=3,
A_PLUS=0.8, W_NORM=78.4, SPIKE_GAIN=0.6, T=200, RATE_ALPHA=0.002, BETA=400）。

训练（逐样本在线 STDP）：权重共享、逐样本更新 → 只能一个样本一个样本跑。
评估（无学习）：可把一批样本向量化并行前向，只累计最后一层放电计数。
标签分配：仅最后一层在训练时累计 digCount[神经元, 标签]，pref = argmax。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerConfig:
    """单层 LIF 神经元的超参与结构参数。"""
    n_out: int = 100
    leak: float = 0.94
    theta_init: float = 15.0
    refr_period: int = 4
    tau_plus: int = 3
    a_plus: float = 0.8
    w_norm: float = 78.4
    rate_alpha: float = 0.002   # 放电率 EMA 系数
    beta: float = 400.0         # homeostasis 强度
    theta_clamp: tuple | None = None   # 每次阈值更新后限幅，如 (5, 100)；None=不限（忠实 JS）
    wta: bool = True            # 胜者全取侧抑制
    homeostasis: bool = True    # 阈值自适应
    input_gain: float = 1.0     # 输入脉冲乘数（补偿稀疏脉冲层→深层的驱动不足）
    w_init_mean: float = 0.15   # 均匀初始化均值（用于 [0, 2*mean]）
    seed: int = 0


class LIFLayer:
    """单层 LIF + WTA + STDP。权重 [n_in, n_out]，输入为 0/1 脉冲向量。"""

    def __init__(self, n_in: int, cfg: LayerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_in = n_in
        self.n_out = cfg.n_out
        rng = torch.Generator(device="cpu").manual_seed(cfg.seed)

        # 初始权重：均匀 [0, 2*mean]，再按列归一化到 w_norm
        W = torch.rand(n_in, cfg.n_out, generator=rng) * (2.0 * cfg.w_init_mean)
        col_sum = W.sum(dim=0, keepdim=True)
        W = W * (cfg.w_norm / col_sum)
        self.W = W.to(device)
        self.theta = torch.full((cfg.n_out,), float(cfg.theta_init), device=device)
        self.rate = torch.zeros(cfg.n_out, device=device)

        # 每样本运行态（start_sample 重置）
        self.V = torch.zeros(cfg.n_out, device=device)
        self.refr = torch.zeros(cfg.n_out, dtype=torch.long, device=device)
        self.last_pre = torch.full((n_in,), -10_000, dtype=torch.long, device=device)
        self.spike_out = torch.zeros(cfg.n_out, device=device)
        self.fire_count = torch.zeros(cfg.n_out, dtype=torch.long, device=device)

        # 训练累计（仅输出层用于标签分配）
        self.dig_count = torch.zeros(cfg.n_out, 10, dtype=torch.long, device=device)

    def reset_sample(self) -> None:
        self.V.zero_()
        self.refr.zero_()
        self.last_pre.fill_(-10_000)
        self.spike_out.zero_()
        self.fire_count.zero_()

    def reset_learned(self) -> None:
        """把一生学到的阈值/放电率复位到初始（供拉马克等复用）。"""
        self.theta.fill_(self.cfg.theta_init)
        self.rate.zero_()
        self.dig_count.zero_()

    @property
    def pref(self) -> torch.Tensor:
        """每神经元偏好的标签（训练累计后 argmax）。"""
        return self.dig_count.argmax(dim=1)

    def step(self, pre_spikes: torch.Tensor, t: int, learn: bool = False,
             label: int | None = None, apply_stdp: bool = True) -> torch.Tensor:
        """前向一步。pre_spikes: [n_in] 0/1 脉冲。返回 [n_out] 输出脉冲。

        时序忠实移植 snn_demo.html simStep()：编码→累积→漏电→WTA→fire→
        不应期递减→homeostasis（rate 衰减全体 + winner 增量 + theta 更新）。

        learn=True 累计 dig_count（标签分配）；apply_stdp=False 冻结权重（校准用）。
        """
        cfg = self.cfg
        self.spike_out.zero_()

        # 记录本步输入脉冲时刻（供 STDP 窗口判断）
        spiking = pre_spikes > 0.5
        self.last_pre[spiking] = t

        # 输入累积（不应期神经元不接收，等价于累加后再清零）
        V = self.V
        V.addmv_(self.W.t(), pre_spikes * cfg.input_gain)
        V[self.refr > 0] = 0.0
        V.mul_(cfg.leak)

        # WTA：refr<=0 且 V>=theta 中取最大者
        winner = -1
        eligible = (self.refr <= 0) & (V >= self.theta)
        if bool(eligible.any()):
            winner = int(torch.nonzero(eligible, as_tuple=True)[0][V[eligible].argmax()])
            self._fire(winner, pre_spikes, learn, label, apply_stdp)

        # 不应期递减（fire 之后）
        self.refr.sub_(1).clamp_(min=0)

        # Homeostasis：每步执行，rate 先全体衰减、winner 增量，再更新 theta
        if cfg.homeostasis:
            self.rate.mul_(1.0 - cfg.rate_alpha)
            if winner >= 0:
                self.rate[winner] += cfg.rate_alpha
            self.theta.add_(self.rate - self.rate.mean(),
                            alpha=cfg.beta * cfg.rate_alpha)
            if cfg.theta_clamp is not None:
                self.theta.clamp_(cfg.theta_clamp[0], cfg.theta_clamp[1])

        return self.spike_out

    def _fire(self, winner: int, pre_spikes: torch.Tensor,
              learn: bool, label: int | None, apply_stdp: bool) -> None:
        cfg = self.cfg
        self.V.zero_()
        # 非 winner 且未处于不应期的神经元 refr=1；winner 设 refr_period
        was_idle = self.refr <= 0
        self.refr[was_idle] = 1
        self.refr[winner] = cfg.refr_period

        # STDP LTP：本步发放的输入脉冲都给 winner 增权重（无 LTD）
        if apply_stdp:
            col = self.W[:, winner]
            col.add_(pre_spikes, alpha=cfg.a_plus)
            # 列归一化
            col_sum = col.sum()
            if col_sum > 0:
                col.mul_(cfg.w_norm / col_sum)

        if learn and label is not None:
            self.dig_count[winner, label] += 1

        self.spike_out[winner] = 1.0
        self.fire_count[winner] += 1

    # ---------- 批量评估（无学习） ----------
    def step_batch(self, pre_spikes: torch.Tensor, t: int) -> torch.Tensor:
        """批量前向一步（无 STDP/无 homeostasis/无 digCount）。
        pre_spikes: [B, n_in] 0/1。返回 [B, n_out] 输出脉冲。"""
        B = pre_spikes.shape[0]
        cfg = self.cfg
        V = self.V_buf
        V += (pre_spikes * cfg.input_gain) @ self.W          # [B, n_out]
        V[self.refr_buf > 0] = 0.0
        V *= cfg.leak

        eligible = (self.refr_buf <= 0) & (V >= self.theta)
        out = torch.zeros(B, self.n_out, device=self.device)
        winner_idx = torch.where(eligible, V, torch.full_like(V, float("-inf"))).argmax(dim=1)
        fire_b = eligible.any(dim=1).nonzero(as_tuple=False).squeeze(1)
        if fire_b.numel():
            w = winner_idx[fire_b]
            out[fire_b, w] = 1.0
            self.V_buf[fire_b] = 0.0                      # 发放样本全部神经元 V 清零
            self.refr_buf[fire_b, w] = cfg.refr_period    # winner 不应期
            was_idle = (self.refr_buf[fire_b] <= 0)       # [K, n_out] 非不应期神经元
            reset = torch.zeros(B, self.n_out, dtype=torch.long, device=self.device)
            reset[fire_b] = was_idle.to(torch.long)
            reset[fire_b, w] = 0                          # 排除 winner（已被设 refr_period）
            self.refr_buf[reset.bool()] = 1
            self.spike_acc[fire_b, w] += 1
        self.refr_buf = torch.clamp(self.refr_buf - 1, min=0)
        return out

    def reset_batch(self, B: int) -> None:
        self.V_buf = torch.zeros(B, self.n_out, device=self.device)
        self.refr_buf = torch.zeros(B, self.n_out, dtype=torch.long, device=self.device)
        self.spike_acc = torch.zeros(B, self.n_out, device=self.device)


@dataclass
class SNNParams:
    """整个 SNN（多层栈）的全局超参。"""
    spike_gain: float = 0.6
    T: int = 200
    num_classes: int = 10
    input_size: int = 784
    seed: int = 0


class SNN:
    """多层前馈 LIF 脉冲网络。最后层负责标签分配，全部层做 STDP。"""

    def __init__(self, layer_cfgs: list[LayerConfig], params: SNNParams,
                 device: torch.device):
        assert layer_cfgs, "至少一层"
        self.params = params
        self.device = device
        self.layers: list[LIFLayer] = []
        n_in = params.input_size
        for i, cfg in enumerate(layer_cfgs):
            if cfg.seed == 0:
                cfg.seed = params.seed + i * 1000
            layer = LIFLayer(n_in, cfg, device)
            self.layers.append(layer)
            n_in = cfg.n_out

    # ---------- 泊松编码 ----------
    def _poisson(self, px: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """px: [B, input_size] ∈ [0,1] → [B, input_size, T] 0/1 脉冲。"""
        p = px.unsqueeze(-1) * self.params.spike_gain          # [B, N, 1]
        shape = (px.shape[0], px.shape[1], self.params.T)
        rand = torch.rand(shape, device=self.device, generator=gen)
        return (rand < p).float()

    # ---------- 训练（逐样本在线 STDP） ----------
    def train_sample(self, px: torch.Tensor, label: int, gen: torch.Generator) -> None:
        """px: [input_size] ∈ [0,1]；label: int。在线 STDP 训练一个样本。"""
        for layer in self.layers:
            layer.reset_sample()
        S = self._poisson(px.unsqueeze(0), gen)[0]             # [input_size, T]
        last_idx = len(self.layers) - 1
        for t in range(self.params.T):
            pre = S[:, t]
            for l, layer in enumerate(self.layers):
                pre = layer.step(pre, t, learn=(l == last_idx),
                                 label=label if l == last_idx else None)

    def calibrate(self, px: torch.Tensor, labels: torch.Tensor,
                  gen: torch.Generator) -> None:
        """校准标签分配：冻结权重跑训练样本，累计 dig_count 后重算 pref。

        px: [B, input_size]；labels: [B]。不清零现有 dig_count（可增量校准）。
        """
        last_idx = len(self.layers) - 1
        for b in range(px.shape[0]):
            for layer in self.layers:
                layer.reset_sample()
            S = self._poisson(px[b].unsqueeze(0), gen)[0]
            for t in range(self.params.T):
                pre = S[:, t]
                for l, layer in enumerate(self.layers):
                    pre = layer.step(pre, t, learn=(l == last_idx),
                                     label=int(labels[b]) if l == last_idx else None,
                                     apply_stdp=False)

    # ---------- 批量评估（无学习） ----------
    def evaluate_batch(self, px: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """px: [B, input_size] ∈ [0,1] → [B, n_last] 最后一层累计放电数。

        冻结 theta，不更新 homeostasis（适合进化中快速评估）。
        """
        B = px.shape[0]
        for layer in self.layers:
            layer.reset_batch(B)
        S = self._poisson(px, gen)                             # [B, input_size, T]
        last_idx = len(self.layers) - 1
        for t in range(self.params.T):
            pre = S[:, :, t]
            for layer in self.layers:
                pre = layer.step_batch(pre, t)
        return self.layers[last_idx].spike_acc

    # ---------- 顺序评估（忠实 JS：推理期也跑 homeostasis） ----------
    def evaluate_sequential(self, px: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """逐样本推理，每样本重置 V/refr，但 rate/theta 持续、每步跑 homeostasis。

        忠实移植 JS 的 infer 模式。px: [B, input_size] → [B, n_last] 累计放电数。
        """
        B = px.shape[0]
        S = self._poisson(px, gen)                             # [B, input_size, T]
        last_idx = len(self.layers) - 1
        acc = torch.zeros(B, self.layers[last_idx].n_out, device=self.device)
        for b in range(B):
            for layer in self.layers:
                layer.reset_sample()
            for t in range(self.params.T):
                pre = S[b, :, t]
                for layer in self.layers:
                    pre = layer.step(pre, t, learn=False)
            acc[b] = self.layers[last_idx].fire_count.float()
        return acc


def accuracy(score: torch.Tensor, pref: torch.Tensor, labels: torch.Tensor) -> float:
    """score: [B, N] 各样本各神经元放电数；pref: [N] 每神经元偏好标签。"""
    preds = pref[score.argmax(dim=1)]
    return float((preds == labels).float().mean())


def accuracy_votes(score: torch.Tensor, pref: torch.Tensor, labels: torch.Tensor,
                   num_classes: int = 10) -> float:
    """投票求和读出：每个神经元按其偏好标签投票，票数=放电数，取票数最高的类。

    比单神经元 argmax 更稳健（见实验：退化发放会霸占 argmax）。
    """
    onehot = torch.nn.functional.one_hot(pref, num_classes).float()   # [N, 10]
    votes = score @ onehot                                            # [B, 10]
    preds = votes.argmax(dim=1)
    return float((preds == labels).float().mean())


def pref_diversity(pref: torch.Tensor, n_classes: int = 10) -> int:
    """偏好标签的种类数（衡量 STDP 是否让神经元分化）。"""
    return int(pref.unique().numel())
