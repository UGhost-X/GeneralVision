"""Deep-learning ROI enhancement: HAT (real-world 4x super-resolution) and Restormer (deblur/denoise).

PyTorch CPU inference with tiled processing so small/large ROIs both work.
Model files are optional: missing torch or weights raise clear errors with setup hints.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("ROI_MODELS_DIR", str(BASE_DIR / "models")))

HAT_WEIGHTS = {
    "real_hat_gan_sharper": (Path("hat") / "Real_HAT_GAN_sharper.pth", 170_277_017),
    "hat_srx4_imagenet": (Path("hat") / "HAT_SRx4_ImageNet-pretrain.pth", 85_137_601),
}
RESTORMER_WEIGHTS = {
    "motion_deblurring": (Path("restormer") / "motion_deblurring.pth", 104_700_429),
    "single_image_defocus_deblurring": (Path("restormer") / "single_image_defocus_deblurring.pth", 104_700_429),
    "real_denoising": (Path("restormer") / "real_denoising.pth", 104_611_957),
    "gaussian_gray_denoising_sigma25": (Path("restormer") / "gaussian_gray_denoising_sigma25.pth", 104_601_589),
}

TORCH_INSTALL_HINT = (
    "未安装 PyTorch。请先安装：uv pip install torch einops --index-url https://download.pytorch.org/whl/cpu "
    "（国内可先下载 CPU wheel：scripts/download_models.py 会自动下载到 %TEMP%\\roi_models\\torch_cpu.whl，"
    "再用 uv pip install 该文件）"
)

_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise RuntimeError(TORCH_INSTALL_HINT)
    return _torch


_device_cache = None


def _device() -> str:
    """Prefer CUDA when available (auto-detected), fall back to CPU."""
    global _device_cache
    if _device_cache is None:
        torch = _get_torch()
        _device_cache = "cuda" if torch.cuda.is_available() else "cpu"
    return _device_cache


def _as_gray_u8(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def _file_ok(rel: Path, expected: int) -> tuple[bool, int]:
    path = MODELS_DIR / rel
    size = path.stat().st_size if path.exists() else 0
    return size >= expected, size


def dl_status() -> dict[str, Any]:
    torch_version = None
    torch_ok = False
    cuda = False
    device_name = None
    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        pass
    return {
        "torch": torch_ok,
        "torch_version": torch_version,
        "cuda": cuda,
        "device": device_name,
        "hat": {name: {"ready": _file_ok(rel, exp)[0], "size": _file_ok(rel, exp)[1], "expected": exp}
                for name, (rel, exp) in HAT_WEIGHTS.items()},
        "restormer": {name: {"ready": _file_ok(rel, exp)[0], "size": _file_ok(rel, exp)[1], "expected": exp}
                      for name, (rel, exp) in RESTORMER_WEIGHTS.items()},
    }


def _load_state(torch, path: Path):
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"模型文件 {path} 损坏或无法加载：{exc}") from exc
    if isinstance(state, dict):
        for key in ("params", "params_ema", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


# ---------------------------------------------------------------------------
# HAT (4x real-world super-resolution / restoration)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_hat(weight: str):
    torch = _get_torch()
    if weight not in HAT_WEIGHTS:
        raise ValueError(f"未知 HAT 权重: {weight}")
    rel, expected = HAT_WEIGHTS[weight]
    ok, size = _file_ok(rel, expected)
    if not ok:
        raise FileNotFoundError(
            f"缺少 HAT 模型文件 {MODELS_DIR / rel}（当前 {size}，需要 {expected} 字节）。"
            "请运行 scripts/download_models.py 下载模型。")
    import hat_arch_standalone as hat_arch
    model = hat_arch.HAT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    state = _load_state(torch, MODELS_DIR / rel)
    model.load_state_dict(state, strict=True)
    model.to(_device())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Restormer (deblur / denoise)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_restormer(task: str):
    torch = _get_torch()
    if task not in RESTORMER_WEIGHTS:
        raise ValueError(f"未知 Restormer 任务: {task}")
    rel, expected = RESTORMER_WEIGHTS[task]
    ok, size = _file_ok(rel, expected)
    if not ok:
        raise FileNotFoundError(
            f"缺少 Restormer 模型文件 {MODELS_DIR / rel}（当前 {size}，需要 {expected} 字节）。"
            "请运行 scripts/download_models.py 下载模型。")
    import restormer_arch
    # Official configs: deblurring tasks use WithBias, denoising tasks use BiasFree.
    layer_norm = {"motion_deblurring": "WithBias",
                  "single_image_defocus_deblurring": "WithBias",
                  "real_denoising": "BiasFree",
                  "gaussian_gray_denoising_sigma25": "BiasFree"}.get(task, "WithBias")
    in_ch = 1 if task == "gaussian_gray_denoising_sigma25" else 3
    model = restormer_arch.Restormer(
        inp_channels=in_ch,
        out_channels=in_ch,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type=layer_norm,
        dual_pixel_task=False,
    )
    state = _load_state(torch, MODELS_DIR / rel)
    model.load_state_dict(state, strict=True)
    model.to(_device())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# tiled inference
# ---------------------------------------------------------------------------

def _to_rgb_tensor(torch, gray_u8: np.ndarray, channels: int = 3):
    if channels == 1:
        arr = gray_u8[None, None].astype(np.float32) / 255.0
    else:
        rgb = np.repeat(gray_u8[..., None], 3, axis=2)  # H x W x 3
        arr = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr)).to(_device())


def _from_tensor_first_channel(torch, tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()[0]  # 3 x H x W
    return np.clip(arr[0] * 255.0, 0.0, 255.0).astype(np.uint8)


def _ramp(size: int, overlap: int) -> np.ndarray:
    r = np.ones(size, np.float32)
    ov = min(int(overlap), size // 2)
    if ov > 0:
        # Ramp down to a small non-zero weight so single-covered border pixels
        # still normalize back to the true model output (no black seam lines).
        r[:ov] = np.linspace(0.1, 1.0, ov, dtype=np.float32)
        r[-ov:] = np.linspace(1.0, 0.1, ov, dtype=np.float32)
    return r


def _tile_mask(shape, overlap: int) -> np.ndarray:
    h, w = shape
    return np.outer(_ramp(h, overlap), _ramp(w, overlap))


def _tiled_infer(model, gray_u8: np.ndarray, scale: int, tile: int, overlap: int, window_mult: int, channels: int = 3) -> np.ndarray:
    """Run a PyTorch model on a grayscale image with overlapping tiles + feathered stitching.

    The model must accept a (1, 3, H, W) tensor in [0, 1] and return the same shape
    with spatial size = input * scale. Only channel 0 of the output is kept.
    """
    torch = _get_torch()
    h, w = gray_u8.shape
    out_h, out_w = h * scale, w * scale
    tile = max(int(tile), window_mult)
    overlap = max(0, min(int(overlap), tile // 2))

    def run_one(patch: np.ndarray):
        ph = (window_mult - patch.shape[0] % window_mult) % window_mult
        pw = (window_mult - patch.shape[1] % window_mult) % window_mult
        if ph or pw:
            patch = cv2.copyMakeBorder(patch, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        x = _to_rgb_tensor(torch, patch, channels)
        with torch.no_grad():
            y = model(x)
        return _from_tensor_first_channel(torch, y)

    if h <= tile and w <= tile:
        out = run_one(gray_u8)
        return out[:out_h, :out_w]

    acc = np.zeros((out_h, out_w), np.float64)
    wsum = np.zeros((out_h, out_w), np.float32)
    step = tile - overlap
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            patch = gray_u8[y0:y1, x0:x1]
            out = run_one(patch)[:patch.shape[0] * scale, :patch.shape[1] * scale]  # crop padding
            mask = _tile_mask((patch.shape[0] * scale, patch.shape[1] * scale), overlap * scale)
            acc[y0 * scale:y1 * scale, x0 * scale:x1 * scale] += out.astype(np.float64) * mask
            wsum[y0 * scale:y1 * scale, x0 * scale:x1 * scale] += mask
    wsum[wsum < 1e-6] = 1.0
    return np.clip(acc / wsum, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _sharpness(gray: np.ndarray) -> float:
    if gray.size == 0 or min(gray.shape) < 3:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())


def _metrics(gray: np.ndarray, processed: np.ndarray, reference: np.ndarray | None = None) -> dict[str, Any]:
    """Sharpness before/after. `reference` is the fair baseline at the output size
    (for HAT 4x output we compare against the original upscaled with bicubic)."""
    if reference is None:
        reference = gray
    same = processed
    if same.shape != reference.shape:
        same = cv2.resize(processed, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    before = _sharpness(reference)
    after = _sharpness(same)
    return {
        "sharpness_before": round(before, 2),
        "sharpness_after": round(after, 2),
        "degraded": bool(after < before * 0.95),
    }


def _blend(gray: np.ndarray, processed: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return processed
    if strength >= 1:
        return processed
    base = gray
    if base.shape != processed.shape:
        base = cv2.resize(base, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(base.astype(np.float64) * (1.0 - strength) + processed.astype(np.float64) * strength,
                   0.0, 255.0).astype(np.uint8)


def hat_enhance(gray: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    params = params or {}
    weight = str(params.get("weight", "real_hat_gan_sharper"))
    tile = int(np.clip(int(params.get("tile_size", 128)), 64, 512))
    overlap = max(8, min(48, tile // 8))
    output_scale = str(params.get("output_scale", "4x"))
    started = time.perf_counter()
    model = _load_hat(weight)
    out = _tiled_infer(model, gray, scale=4, tile=tile, overlap=overlap, window_mult=16)
    if output_scale == "1x":
        final = cv2.resize(out, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        final = out
    final = _blend(gray, final, float(params.get("blend", 0.0)))
    reference = gray if final.shape == gray.shape else cv2.resize(gray, (final.shape[1], final.shape[0]), interpolation=cv2.INTER_CUBIC)
    metrics = _metrics(gray, final, reference)
    metrics.update({
        "model": "HAT",
        "weight": weight,
        "scale": round(final.shape[1] / gray.shape[1], 2),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    })
    return {
        "processed": final,
        "overlay": cv2.cvtColor(final, cv2.COLOR_GRAY2BGR),
        "metrics": metrics,
        "psf_estimated": None,
    }


def restormer_enhance(gray: np.ndarray, params: dict[str, Any] | None = None) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    params = params or {}
    task = str(params.get("task", "motion_deblurring"))
    tile = int(np.clip(int(params.get("tile_size", 256)), 64, 512))
    overlap = max(8, min(48, tile // 8))
    started = time.perf_counter()
    model = _load_restormer(task)
    in_ch = 1 if task == "gaussian_gray_denoising_sigma25" else 3
    out = _tiled_infer(model, gray, scale=1, tile=tile, overlap=overlap, window_mult=8, channels=in_ch)
    final = _blend(gray, out, float(params.get("blend", 0.0)))
    metrics = _metrics(gray, final)
    metrics.update({
        "model": "Restormer",
        "task": task,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    })
    return {
        "processed": final,
        "overlay": cv2.cvtColor(final, cv2.COLOR_GRAY2BGR),
        "metrics": metrics,
        "psf_estimated": None,
    }
