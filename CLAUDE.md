# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

- Git remote: `origin` → `https://github.com/UGhost-X/GeneralVision.git`
- 默认主分支为 `main`（原 `master` 已合并进 `main` 并删除，远程/本地均不再有 `master`）。所有推送目标均为 `origin main`。
- **Before modifying any code** (editing, creating, or deleting code files), first commit and push all pending changes to the remote:
  ```bash
  git add -A
  git commit -m "<描述本次改动>"
  git push origin main
  ```
- 每次修改代码之前，必须先将当前已有的改动提交并推送到远程仓库 `origin` 的 `main` 分支，然后再开始本次修改。
- 代码验证方式：**不需要运行冒烟测试或其他额外测试**。改动后通过严谨的代码审查核对逻辑正确性即可（必要时做小型直接运行验证），不引入、不维护测试套件。

## Environment

- Python 3.10 managed via `uv` (virtualenv in `.venv`)
- To activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux)
- Key packages: `torch` (2.6.0+cu124), `torchvision`, `numpy`, `PIL` (pillow)

No build system or test runner is configured (no `pyproject.toml`, `setup.py`, or test files).

## Project overview

Anomaly detection / industrial visual inspection system using a patch-distribution model (PaDiM-style) with a Wide ResNet-50-2 backbone. Each "product" is a distinct defect type or part that gets its own training run.

## Directory structure

```
product_1/          # Training data for a single product/model
  config.json       # Model hyperparams + training history (backbone, losses, etc.)
  weights.txt       # TSV: <image_filename>\t<weight> (1.0 = normal, 2.0 = anomalous)
  {class}_{idx}.jpg # Training images, class prefix groups samples by defect type
calibration/        # FP/FN calibration samples organized by date
  YYYY/MM/DD/
    calib_fp_{class}_{timestamp}.jpg  # False positives for threshold tuning
    calib_fn_{class}_{timestamp}.jpg  # False negatives for threshold tuning
compress_images.py  # Standalone PIL-based image compression script (targets a hardcoded directory)
```

## `product_1/config.json`

Training config for the anomaly detection model. Key fields:
- `backbone`: CNN backbone (`"wide_resnet50_2"`)
- `layers`: which ResNet layers to extract features from (`["layer2", "layer3"]`)
- `patch_size`, `feature_dim`, `input_size`, `crop_size`: model architecture params
- `noise_sigma`, `gaussian_sigma`: data augmentation params
- `meta_epochs`, `disc_epochs`, `batch_size`: training params
- `train_losses`, `val_losses`, `best_val_loss`: training history arrays
- `num_train_samples`, `num_val_samples`: dataset split sizes

## `product_1/weights.txt`

Tab-separated file mapping training images to anomaly weights. Weight 1.0 = normal sample, weight 2.0 = anomalous (defective) sample. The higher weight gives anomalous samples more influence during training.

## `calibration/`

Calibration images used for threshold tuning after training. Naming convention:
- `calib_fp_*` — false positive samples (normal, but model flagged them)
- `calib_fn_*` — false negative samples (defective, but model missed them)
- The `{class}` segment (e.g., `螺丝` = screw, `孔洞` = hole) indicates the defect category
- Images are organized in date-based subdirectories

## `compress_images.py`

Standalone script that batch-compresses images > 2MB in a hardcoded target directory. Uses PIL with format-specific optimization (PNG quantization, JPEG quality reduction). Run directly with `python compress_images.py`. Edit `TARGET_DIR` in the script to change the target directory.

## LIF 生态游戏（回合制：喂食-产出-淘汰-有性繁殖）

游戏式神经进化的可视化演示：生物体 = LIF 脉冲网络（纯 numpy、权重出生即随机、**一生无学习**），在培养皿里吃 MNIST 数字、产出数字、按正确率淘汰与有性繁殖。与 STDP 进化系统（evolve.py）并存独立。

### 文件与职责

- `eco_engine.py` — 引擎（纯 numpy + numba JIT）：
  - `Genome` / `random_genome` / `crossover`：基因与繁殖（存活加权交叉 + 五类结构突变 → 变深度 1-4 隐藏层）
  - `forward`：按架构分组向量化 LIF 前向（大分组用 numpy 向量化，小分组用 `_forward_core_multi` numba 回退；泊松编码 → 逐层 LIF+WTA 无学习 → 产出层）
  - `Ecosystem`：回合主循环（喂食 → 全体产出 → 淘汰 → 繁殖 → 全灭重播 → 停止判定），每回合输出事件流
- `eco_server.py` — 本地 `http.server` 服务（`python eco_server.py --port 8765`），托管 `eco_game.html` + API：
  `/api/state`（种群/配置/历史）、`/api/step`（推演一回合，返回 events+stats）、`/api/digit_image`、`/api/manual_feed`、`/api/config`
- `eco_game.html` — 单文件前端：canvas 培养皿（点击个体解剖）、食物数字、统计曲线/存活直方图、参数滑块、手动喂食


### 回合制规则（v3）

一回合 = 喂 1 个随机数字给全部存活者 → 各自产出 → **产错不立即死亡**，而是降低本回合繁殖成功概率；**连续 3 回合未繁殖成功则立即死亡** + 未死但 `age > 存活回合数上限` **自然死亡**（计停止指标）→ 存活者**选型交配**（年龄、跨数字偏好、fitness 综合加权）+ 存活加权交叉繁殖 → 密度依赖 + **承载力封顶**（繁殖目标使用 `N_REPRO × alpha` 计算，默认可超过 1000，但受 `REPRO_GROWTH_DIVISOR=30` 和承载力约束）→ **每 5 回合执行全数字体检**：喂 0-9 更新 fitness，并对 top 个体重拟合读出层 → 结算。**停止条件**：累计自然死亡/总死亡 ≥ 95%（随机权重下基本不可达）；某回合全员死亡则复用筛选出的初始奠基个体全灭重播。

### 关键常量（eco_engine.py 顶部）

`T=40`（仿真步数）、`INPUT_SAMPLES_PER_STEP=12`（每步采样的像素数）、`HIDDEN_SIZE=100`、`LEAK=0.94`、`THETA_HIDDEN=12.0`、`WTA_K=6`（每层 top-k 脉冲数）、`SURVIVAL_ROUNDS=20`（自然寿命）、`N_REPRO=50`（繁殖倍数）、`ASSORT_STRENGTH=0.5`（选型强度默认）、`CAPACITY=10000`（承载力）、`INIT_POP=1000`（初始/重播种群）、`SELECT_PER_DIGIT=100`（每个数字筛出的初始个体数）、`SCREEN_POOL_SIZE=2000`（全数字筛候选池）、`SCREEN_STEPS=8`、`SCREEN_SAMPLES=4`、`SCREEN_READOUT_SAMPLES_PER_DIGIT=1`、`CENSUS_EVERY_ROUNDS=5`、`CENSUS_REFIT_TOP=120`、`CENSUS_ELITE=60`、`CENSUS_ELITE_ROUNDS=2`、`CENSUS_DIGIT_SAMPLES=3`、`CENSUS_STEPS=24`、`CENSUS_READOUT_SAMPLES_PER_DIGIT=5`、`CENSUS_WEAK_BOOST=4`、`READOUT_LAMBDA=0.1`、`FITNESS_LAMBDA=0.5`、`COVERAGE_LAMBDA=0.35`、`P_READOUT_MUTATION=0.30`、`P_READOUT_SPARSE_RESET=0.05`、`TRAIT_MUTATION_RATE=0.25`、`P_STRUCT_MUTATION=0.45`、`WTA_K_MIN=1`、`WTA_K_MAX=12`、`LEAK_MIN=0.80`、`LEAK_MAX=0.99`、`INPUT_GAIN_MIN=0.5`、`INPUT_GAIN_MAX=3.0`、`THETA_SCALE_MIN=0.5`、`THETA_SCALE_MAX=2.0`、`REPRO_SUCCESS_BASE=0.9`、`REPRO_WRONG_PENALTY=0.5`（产错繁殖惩罚，基于平滑准确率 acc_ema 而非单次对错）、`NO_REPRO_DEATH_ROUNDS=3`、`REPRO_GROWTH_DIVISOR=30`、`DENSITY_FLOOR=0.05`、`POP_GROWTH=0.10`（每回合净增长率，密度加权）、`DIGIT_FREQUENCIES=[1.0]*10`（投喂频率：周期块中每数字 0-9 出现次数权重，默认等频，块用完按权重重新生成并洗牌，避免纯随机扎堆）；产错不立即死亡，结构突变概率 `P_GROW=0.40 / P_SPLIT=0.15 / P_MERGE=0.10 / P_PRUNE=0.10 / P_ADDRANDOM=0.03`，层数 1-4、每层 20-200、总隐藏 ≤400。

### 最近更新（2026-08-09）

- 已从零重建 LIF 生态游戏三件套：`eco_engine.py`、`eco_server.py`、`eco_game.html`，未从历史改动中复用代码。
- `eco_engine.py` 已实现 Genome/随机权重、存活加权交叉、五类结构突变、向量化分组 LIF 前向 + numba 小分组回退、v3 回合生态主循环。
- `eco_server.py` 已实现本地 `http.server` 与 `/api/state`、`/api/step`、`/api/digit_image`、`/api/manual_feed`、`/api/config`、`/api/reset`。
- `eco_game.html` 已实现培养皿动画、食物数字、参数滑块、手动喂食、统计曲线/直方图和点击个体解剖。
- 性能优化：默认 1000 个体的 HTTP `/api/step` 约 0.56 秒；20 回合压测最差约 0.96 秒，已控制在 1 秒内。
- 初始种群已改为全数字评分筛选：先生成 2000 个候选，对 0-9 全部喂食并按 `fitness = 平均数字正确率 + 0.5 × 最差数字正确率` 排序，再为每个数字挑 100 个低精度正确的候选，并用完整 `T=40` 终筛确认；最终仍组成 0-9 各 100 的 1000 个体初始种群，奠基个体会被缓存，重置/全灭重播直接复用。
- 繁殖规则更新：产错不再立即死亡，改为降低配对繁殖成功概率；连续 3 回合未繁殖成功死亡；配对更倾向跨数字偏好；繁殖目标使用 N 与 alpha，默认种群可超过 1000 且仍保持单回合约 1 秒内。
- 线性读出层已接入：LIF 隐藏层脉冲率作为特征，输出改为 `hidden_rate @ readout_weights + readout_bias`；初始筛选会为奠基个体拟合轻量岭回归读出权重。
- 新增遗传性状与变异：`longevity_bonus`（寿命）、`fecundity`（繁殖力）、`wrong_tolerance`（错误耐受）、`mutation_rate`（变异率）；同架构交叉时可整段交换读出层，读出走小扰动/稀疏重置变异，前端解剖面板已展示这些性状。
- 已加入全数字定期体检与读出层重拟合：每 5 回合喂 0-9 更新每个体的 `fitness`，对 top 200 个体用每数字 5 个训练样本重拟合读出层；fitness 会进入配对、繁殖成功率和产仔权重，同架构交叉默认继承高 fitness 亲本的读出层。
- 已加入可遗传、可变异的神经元结构基因：`wta_k`（每层 top-k 脉冲数）、`leak`（膜电位漏电率）、`input_gain`（输入增益）、`threshold_scale`（发放阈值缩放）；这些基因会参与前向计算，随交叉遗传，并由 `P_STRUCT_MUTATION` 驱动结构变异，前端解剖面板可查看。
- 泛化繁殖已增强：体检改为每个数字多个样本，fitness 使用硬准确率 + softmax 置信度 + 数字覆盖度；对种群最弱的数字定向增加读出训练样本；配对奖励高 fitness 和互补数字覆盖；体检 top 个体获得精英保护，短回合内不会被未繁殖死亡淘汰。



## 其他
 - 使用中文回复，英文思考
