# GeneralVision

用**神经进化 + 脉冲神经网络（SNN）**做 MNIST 手写数字识别的实验项目，附带一个自包含的 **CNN 手写数字识别可视化演示页**。

核心思路：**进化只搜索网络的架构与超参，每个个体"一生"中由 STDP（脉冲时序依赖可塑性）无监督地自己学习突触权重**；权重生命周期结束即丢弃、不遗传——避开"直接进化 1000 个连续权重"的不可行路线（neuroevolution + SNN 领域的主流做法）。

---

## 项目简介

项目分两大部分：

| 部分 | 内容 | 关键文件 |
|------|------|----------|
| **LIF 神经进化系统** | 进化算法（选择/变异/繁殖）驱动 LIF 脉冲网络做 MNIST 识别，STDP 学权重 | `evolve.py` `snn.py` `genome.py` `mutate.py` `evaluate.py` |
| **CNN 可视化演示页** | 自包含 HTML 页面，逐层可视化 CNN 如何识别手写数字 | `index.html` `train_mnist.py` `mnist_weights.json` |

> 本项目脱胎于早期的 PaDiM 风格工业视觉异常检测，现已完全转型为 SNN 进化方向（历史遗留描述见 `CLAUDE.md`，不代表当前状态）。

---

## 目录结构

```
.
├── config.py            # 进化超参数集中配置（种群/代数/选择/适应度权重）
├── genome.py            # 基因组：个体架构+超参（无权重），可序列化 round-trip
├── snn.py               # LIF 脉冲网络：泊松编码、WTA、STDP、批量/顺序评估
├── mutate.py            # 变异算子：无性分裂（连续扰动+增删层），强度随代数衰减
├── evaluate.py          # 评估器：一生 = STDP 训练 → 特征提取 → 监督线性读出
├── evolve.py            # 进化主循环：稳态选择、精英保留、日志、检查点、断点续跑
├── data_loading.py      # MNIST 原始 ubyte 加载 + 固定 train/val 划分
├── viz.py               # 从检查点历史绘制适应度/准确率/架构演化曲线
│
├── index.html           # CNN 手写数字识别可视化演示页（单文件自含，权重内联）
├── train_mnist.py       # 训练微型 CNN 并导出权重注入 index.html
├── mnist_weights.json   # CNN 权重导出（train_mnist.py 产物）
├── gen_snn_samples.py   # [已失效] 为已删除的 snn_demo.html 生成 MNIST 样本
│
├── requirements.txt     # 依赖（torch / torchvision / numpy / matplotlib）
├── evolution_curves.png # viz.py 生成的运行结果曲线
├── docs/superpowers/    # 早期 SNN 演示页（snn_demo.html）的规格与实施计划
├── _*.py                # 临时调查/实验脚本（见下文）
│
├── data/MNIST/raw/      # MNIST 原始文件（gitignored，需自行准备）
└── checkpoints/         # 进化运行产物 gen_*.json + evolution_log.jsonl（gitignored）
```

---

## 环境准备

- Python 3.10（用 `uv` 管理虚拟环境 `.venv`）
- 激活虚拟环境后安装依赖：

```bash
# Windows
.venv\Scripts\activate
uv pip install -r requirements.txt   # 或 pip install -r requirements.txt
```

- **MNIST 数据**：进化系统从 `data/MNIST/raw/` 读取已解压的原始 ubyte 文件
  （`train-images-idx3-ubyte` / `train-labels-idx1-ubyte` / `t10k-*-idx*-ubyte`）。
  该目录已被 gitignore，全新克隆需自行下载解压；`train_mnist.py` 会通过 torchvision 自动下载到 `data/MNIST/`。

---

## 快速开始

```bash
# 1. 数据自检（确认 MNIST 原始文件就位）
python data_loading.py

# 2. 冒烟测试：6 个体 × 2 代，验证流程可跑通
python evolve.py --smoke

# 3. 正式进化运行（config.py 默认 50 个体 × 50 代，CPU）
python evolve.py

# 4. 断点续跑（从 checkpoints/ 最后一个检查点恢复）
python evolve.py --resume checkpoints

# 5. 可视化运行结果（读最后一个 gen_*.json 的 history）
python viz.py                # 输出 evolution_curves.png
python viz.py checkpoints --out curves.png
```

常用命令行参数（`evolve.py`）：`--population N`、`--generations N`、`--device cpu/cuda`、`--smoke`、`--resume DIR`。

### HTML 演示页

```bash
# 直接双击/浏览器打开 index.html 即可（权重已内联，无需网络）
# 重训 CNN 并重新注入权重：
python train_mnist.py                  # 训练 + 写入 mnist_weights.json
python train_mnist.py --inject index.html            # 训练后同时注入页面
python train_mnist.py --inject-only index.html       # 仅用现有权重注入，不重训
```

---

## 系统设计

### 进化范式

- **基因组只含架构/超参**（每层神经元数、leak、阈值、w_norm、input_gain 等连续+离散量），**不含突触权重**。
- 每个个体在"一生"中：用 STDP 从随机初始化权重开始无监督学习，一生结束权重即丢弃。
- 进化压力只作用在**架构层**；权重初始化 seed 也参与变异（下一代重新学）。

### 各模块职责

| 模块 | 职责 |
|------|------|
| `genome.py` | `Genome` = 层列表（`LayerConfig`）+ `spike_gain`/`T`/`train_samples`/`seed`；`seed_genome()` 生成单层 100 神经元基线 |
| `snn.py` | `LIFLayer`：漏电积分、胜者全取（WTA）、不应期、homeostasis 阈值自适应、STDP（仅 LTP + 列归一化）；`SNN`：多层前馈栈、泊松速率编码、逐样本在线训练 / 批量评估 / 顺序评估；`accuracy`/`accuracy_votes` 读出 |
| `mutate.py` | 无性分裂 = 克隆 + 随机变异：连续超参对数空间高斯扰动、神经元数增减、增删/复制整层（结构变异）、全局超参变异；变异幅度与结构变异概率随代数衰减（先广搜后细调） |
| `evaluate.py` | 一生流程：STDP 训练 `train_samples` 个样本 → 提取末层脉冲计数作特征 → **岭回归监督线性读出** → 适应度 |
| `evolve.py` | 稳态选择：淘汰底部 `bottom_frac`，幸存者原样进入下一代；繁殖者由锦标赛 + 少量轮盘赌（多样性）选出，各分裂 1 子代；每 `checkpoint_every` 代存检查点、写 JSONL 日志，支持确定性断点续跑 |

### 适应度

```
fitness = 读出准确率 + w_sparse × 稀疏奖励 − w_compact × (神经元数 / 100)
```

- **稀疏奖励**：1 − 过度发放神经元比例（总放电 > 2× 均值），抑制 WTA 中单个神经元霸占发放的退化模式。
- **紧凑惩罚**：鼓励更小网络。
- 多次不同初始化种子评估取平均（`eval_repeats`），抑制权重初始化的随机噪声。

### 读出方式（关键）

早期纯无监督标签分配（`digCount` → 每神经元偏好 `pref`）不稳定——同一网络因初始化/训练量在 20%~63% 间非单调跳变，破坏进化信号。现改为：**STDP 学特征，监督线性读出**（岭回归，`evaluate._ridge_readout`），准确率 69–71%、跨种子 ±1%，保留了 LIF+STDP 核心，进化信号稳定。

---

## 关键设计决策（实验结论）

| 问题 | 结论 |
|------|------|
| JS 演示页忠实参数（`w_norm=78.4, θ=15`）退化 | homeostasis 使阈值发散 → 调参为 `w_norm=16, θ₀=25, θ_clamp=(5,100)`（见 `genome.BASE_LAYER`） |
| 纯无监督标签分配不稳定 | 改用监督线性读出（岭回归），准确率稳定 69–71% |
| 多层网络不达预期 | 深层稀疏脉冲驱动不足，需 `input_gain`（10–80）补偿；即使调好也 ≤ 单层 |
| 直接进化连续权重不可行 | 进化只搜架构/超参，STDP 学权重，权重不遗传 |

---

## 运行结果

**首次正式运行**（16 个体 × 8 代，监督线性读出，CPU，约 43 分钟；gen 8 因 `patience=8` 提前终止）：

- 最优适应度 **0.727**（acc **69.4%**），gen 0 即达并稳定 8 代——单层 STDP 特征 + 线性读出的上限
- 中位适应度 0.627 → 0.693 持续上升——**选择机制有效淘汰退化个体**
- 最终种群 16 个中 14 个收敛为单层（n_out 78–135，T 143–247），仅 2 个 2 层个体存活且未超越单层
- 结论：进化机制本身可行（种群质量提升），但性能上限由 **STDP 学习规则** 决定而非进化；2 层网络在现有学习规则下大多退化（acc≈0.09）

产物：`evolution_curves.png`（viz.py 绘制）、`checkpoints/`（`gen_*.json` + `evolution_log.jsonl`，gitignored）。

---

## 实验脚本（`_*.py`，临时调查记录）

以下脚本是开发过程中的一次性实验，非正式组件，供复现调查：

| 脚本 | 调查内容 |
|------|----------|
| `_exp_readout.py` | STDP 学特征 + 监督线性读出是否稳定、准确率多高 |
| `_exp_2layer.py` | 2 层网络第二层 `input_gain` 扫描，能否使深层神经元分化 |
| `_exp_theta.py` / `_exp_tune.py` / `_exp_grid.py` | 阈值与超参调参、网格搜索 |
| `_exp_twophase.py` / `_exp_oracle.py` | 两阶段训练 / 理想读出上界 |
| `_final_baseline.py` | 最终基线验证 + 2 层冒烟测试 |
| `_repro_seed.py` / `_verify_single.py` | 种子可复现性验证、单个体核对 |
| `_time_eval.py` / `_time_size.py` | 单次评估计时（估算总耗时）、网络规模对耗时的影响 |
| `_bench.py` / `_prof.py` / `_diag.py` | 基准 / 性能剖析 / 诊断 |

---

## 参数速查

### `snn.LayerConfig`（单层，默认值）

`n_out=100` · `leak=0.94` · `theta_init=15` · `refr_period=4` · `tau_plus=3` · `a_plus=0.8` · `w_norm=78.4` · `rate_alpha=0.002` · `beta=400` · `theta_clamp=None` · `wta=True` · `homeostasis=True` · `input_gain=1.0` · `w_init_mean=0.15`

> 进化基线使用调参后的 `genome.BASE_LAYER`：`w_norm=16, theta_init=25, theta_clamp=(5,100), a_plus=0.8`

### `snn.SNNParams`

`spike_gain=0.6`（泊松放电系数）· `T=200`（每样本仿真步数）· `num_classes=10` · `input_size=784`

### `config.EvolutionConfig`（进化默认值）

`population_size=50` · `g_max=50` · `patience=8` · `elite=2` · `bottom_frac=0.3` · `top_frac=0.3` · `tournament_size=3` · `roulette_frac=0.2` · `w_sparse=0.05` · `w_compact=0.01` · `eval_repeats=1` · `val_size=1000` · `device=cpu` · `checkpoint_every=2`

---

## 已知限制与后续方向

- **性能瓶颈在学习规则**：进化搜索架构已到单层 STDP + 线性读出上限（~70%），突破需改进学习规则（半监督/监督 STDP、更强的深层训练）或扩大规模，而非继续只搜架构。
- `gen_snn_samples.py` 的目标文件 `snn_demo.html` 已移除，脚本当前不可用（可作参考）。
- 当前单次评估为 CPU 逐样本 STDP 训练，正式运行耗时较长；增大种群/代数前建议先用 `_time_eval.py` 估时。

---

## 相关文档

- `docs/superpowers/specs/` 与 `plans/`：早期 `snn_demo.html`（LIF+STDP 单层演示页）的整页精简重做设计与实施计划（历史参考）。
- `CLAUDE.md`：环境/工作流说明（其中 PaDiM 描述为历史遗留，已过时）。

---

## LIF 生态游戏（回合制：喂食-产出-淘汰-有性繁殖）

游戏式神经进化：生物体（LIF 网络 784→100→10，纯 numpy、一生无学习、权重出生即随机）
住在培养皿里吃数字、产出数字。**回合制**规则（一回合 = 喂 1 个数字 → 全部存活个体各自产出 → 结算）：

- **一回合喂 1 个随机数字**：所有存活个体同时看到同一个数字并各自产出。
- **产出错误 → 当回合死亡（非自然死亡）**：不累计进自然死亡计数。
- **对且活到存活回合数上限 → 自然死亡**：只有"对但超龄"才计自然死亡。
- **存活者随机两两配对有性繁殖**：逐权重 50/50 取双亲 + **存活时长加权**（活得越久的个体基因占比越大）+ 高斯扰动 + 千分之一大突变。
- **密度依赖 + 承载力封顶**：繁殖数量随当前密度衰减（承载力处仍保留 5% 替代性繁殖，防"满→暴毙→回填"锯齿），种群硬上限 = 承载力。
- **全灭重播**：某回合全员死亡则按初始种群数重新播种。
- **停止条件**：累计自然死亡 / 累计总死亡 ≥ 95% 停止推演。

可调参数（前端滑块或 `/api/config`，改动即时生效）：

| 参数 | 范围 | 默认 | 含义 |
|------|------|------|------|
| 投喂速度 | 1–10 秒/回合 | 5 s | 自动模式下每回合的间隔 |
| 存活回合数 | 10–30 | 20 | 自然寿命上限（回合） |
| 繁殖倍数 N | 10–100 | 50 | 每对每回合繁殖量 = 存活回合数 × N（再乘密度系数） |
| 承载力 | 100–5000 | 500 | 种群硬上限 |
| 初始种群 | 60–1000 | 60 | 初始 / 全灭重播种群数 |

运行：
    python eco_server.py --port 8765
浏览器打开 http://127.0.0.1:8765 即可观看培养皿动画（只显示存活个体，死亡当回合即移除）、
手动喂食、个体解剖与统计曲线。

与 STDP 进化系统（evolve.py）的区别：本游戏是纯权重遗传进化（无 STDP/无学习），
STDP 系统是架构进化 + 一生 STDP 自学。两者独立共存。
