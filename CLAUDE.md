# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

- Git remote: `origin` → `https://github.com/UGhost-X/GeneralVision.git`
- **Before modifying any code** (editing, creating, or deleting code files), first commit and push all pending changes to the remote:
  ```bash
  git add -A
  git commit -m "<描述本次改动>"
  git push origin <branch>
  ```
- 每次修改代码之前，必须先将当前已有的改动提交并推送到远程仓库 `origin`，然后再开始本次修改。

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

## 其他
 - 使用中文回复，英文思考