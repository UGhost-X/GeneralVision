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
  - `_forward_core_multi` + `forward`：numba 多层 LIF 前向（泊松编码 → 逐层 LIF+WTA 无学习 → 产出层）
  - `Ecosystem`：回合主循环（喂食 → 全体产出 → 淘汰 → 繁殖 → 全灭重播 → 停止判定），每回合输出事件流
- `eco_server.py` — 本地 `http.server` 服务（`python eco_server.py --port 8765`），托管 `eco_game.html` + API：
  `/api/state`（种群/配置/历史）、`/api/step`（推演一回合，返回 events+stats）、`/api/digit_image`、`/api/manual_feed`、`/api/config`
- `eco_game.html` — 单文件前端：canvas 培养皿（点击个体解剖）、食物数字、统计曲线/存活直方图、参数滑块、手动喂食


### 回合制规则（v3）

一回合 = 喂 1 个随机数字给全部存活者 → 各自产出 → **产错按已存活回合数的概率死亡**（首回合 100% 即死，存活 1 回合 → 50%，每多存活一回合死亡概率 ×1.5、封顶 100%；存活 2 回合 → 75%）+ 未死但 `age > 存活回合数上限` **自然死亡**（计停止指标）→ 存活者**选型交配**（按年龄相似度，强度 s 可调）+ 存活加权交叉繁殖 → 密度依赖 + **承载力封顶**（繁殖 = 覆盖当回合死亡 + 密度加权净增长，种群向承载力 logistic 增长）→ 结算。**停止条件**：累计自然死亡/总死亡 ≥ 95%（随机权重下基本不可达）；某回合全员死亡则按初始种群数全灭重播。

### 关键常量（eco_engine.py 顶部）

`T=40`（仿真步数）、`HIDDEN_SIZE=100`、`LEAK=0.94`、`THETA_HIDDEN=12.0`、`SURVIVAL_ROUNDS=20`（自然寿命）、`N_REPRO=50`（繁殖倍数）、`ASSORT_STRENGTH=0.5`（选型强度默认）、`CAPACITY=10000`（承载力）、`INIT_POP=1000`（初始/重播种群）、`DENSITY_FLOOR=0.05`、`POP_GROWTH=0.10`（每回合净增长率，密度加权）；产错死亡概率 `wrong_death_prob(survived)`：存活 0 → 100%、存活 1 → 50%、每多存活一回合 ×1.5、封顶 99%结构突变概率 `P_GROW=0.40 / P_SPLIT=0.15 / P_MERGE=0.10 / P_PRUNE=0.10 / P_ADDRANDOM=0.03`，层数 1-4、每层 20-200、总隐藏 ≤400。



## 其他
 - 使用中文回复，英文思考