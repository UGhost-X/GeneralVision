"""ROI processing algorithms for the annotation + ROI workbench.

All algorithms are implemented with numpy + OpenCV only (no scipy/skimage),
operating on a grayscale ROI crop and returning a dict with:
  processed  : np.ndarray uint8 grayscale (restored image or binary edge map)
  overlay    : np.ndarray uint8 BGR (same size, colored overlay)
  metrics    : dict of scalar metrics
  psf_estimated : dict | None (blind deconvolution PSF estimation result)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

EPS = 1e-12

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_gray_u8(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image.copy()
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance as a sharpness metric."""
    gray = _as_gray_u8(gray)
    if gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def _fft_convolve(image: np.ndarray, kernel: np.ndarray, flip: bool = False) -> np.ndarray:
    """FFT-based full-size convolution (same shape as image)."""
    k = kernel[::-1, ::-1] if flip else kernel
    fk = np.fft.fft2(k, s=image.shape)
    return np.fft.ifft2(np.fft.fft2(image) * fk).real


def _blend_strength(gray: np.ndarray, restored: np.ndarray, strength: float) -> np.ndarray:
    """Blend restored with the original to keep output faithful (anti-artifact)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s >= 1.0:
        return restored
    return gray.astype(np.float64) * (1.0 - s) + restored * s


def _clipped_ratio(image: np.ndarray) -> float:
    """Fraction of pixels saturated to 0/255 (ringing artifact indicator)."""
    if image.size == 0:
        return 0.0
    return float(np.mean((image == 0) | (image == 255)))


def _clip_added(before: np.ndarray, after: np.ndarray) -> float:
    """Fraction of pixels newly saturated to 0/255 vs the input (true artifact rate)."""
    b = (before == 0) | (before == 255)
    a = (after == 0) | (after == 255)
    return float(np.mean(a & ~b))


def _visual_edge(gray: np.ndarray, edges: np.ndarray, binary_output: bool) -> np.ndarray:
    """Edge result view: pure binary map, or recognizable gray image + white edges."""
    if binary_output:
        return edges.copy()
    vis = gray.copy()
    vis[edges > 0] = 255
    return vis


def gaussian_psf(sigma: float, kernel_size: int | None = None) -> np.ndarray:
    sigma = max(float(sigma), 0.1)
    k = kernel_size or int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    k = max(k, 3)
    ax = np.arange(-(k // 2), k // 2 + 1, dtype=np.float64)
    x, y = np.meshgrid(ax, ax)
    p = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    s = p.sum()
    return p / s if s > 0 else p


def motion_psf(length: float, angle_deg: float) -> np.ndarray:
    length = max(float(length), 1.0)
    size = max(int(np.ceil(length)) + 2, 3)
    if size % 2 == 0:
        size += 1
    psf = np.zeros((size, size), np.float64)
    center = (size - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    steps = max(int(np.ceil(length)) + 1, 2)
    for t in np.linspace(-length / 2.0, length / 2.0, steps):
        x, y = center + dx * t, center + dy * t
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            psf[yi, xi] += 1.0
    s = psf.sum()
    return psf / s if s > 0 else psf


# ---------------------------------------------------------------------------
# PSF estimation (heuristics used by blind deconvolution when estimate_psf=on)
# ---------------------------------------------------------------------------


def estimate_gaussian_sigma(gray: np.ndarray) -> tuple[float, bool]:
    """Estimate a Gaussian blur sigma using a reblur heuristic.

    Returns (sigma, fallback_used). The estimate is heuristic, not a guarantee.
    """
    gray = _as_gray_u8(gray)
    if gray.size < 25 or min(gray.shape) < 7:
        return 1.0, True
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    if int(edges.sum()) < 20:
        return 1.0, True
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    base = float(gm[edges > 0].mean())
    if base < 1e-9:
        return 1.0, True
    best_sigma, best_score = 1.0, float("inf")
    for sigma in np.arange(0.25, 5.01, 0.25):
        bl = cv2.GaussianBlur(blurred, (0, 0), float(sigma))
        gx2 = cv2.Sobel(bl, cv2.CV_64F, 1, 0, ksize=3)
        gy2 = cv2.Sobel(bl, cv2.CV_64F, 0, 1, ksize=3)
        ratio = float(np.hypot(gx2, gy2)[edges > 0].mean()) / base
        score = abs(ratio - 0.5)
        if score < best_score:
            best_score, best_sigma = score, float(sigma)
    sigma_est = min(max(best_sigma * 0.6, 0.3), 5.0)
    return round(sigma_est, 2), False


def estimate_motion_params(gray: np.ndarray) -> tuple[float, float, bool]:
    """Estimate motion blur length/angle via the real cepstrum.

    Returns (length, angle_deg, fallback_used).
    """
    gray = _as_gray_u8(gray)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 6.0, 0.0, True
    f = np.fft.fft2(gray.astype(np.float64))
    logmag = np.log(np.abs(f) + 1e-12)
    cep = np.fft.fftshift(np.fft.ifft2(logmag).real)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - cy, xx - cx)
    cep_masked = np.where(dist > 3, cep, np.inf)
    idx = np.unravel_index(np.argmin(cep_masked), cep.shape)
    dy, dx = idx[0] - cy, idx[1] - cx
    length = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if length < 2.0 or length > 40.0:
        return 6.0, 0.0, True
    return round(length, 1), round(angle, 1), False


def estimate_psf_params(gray: np.ndarray, psf_model: str) -> dict[str, Any]:
    if psf_model == "motion":
        length, angle, fallback = estimate_motion_params(gray)
        return {"length": length, "angle": angle, "method": "cepstrum", "fallback": fallback}
    sigma, fallback = estimate_gaussian_sigma(gray)
    return {"sigma": sigma, "method": "reblur", "fallback": fallback}


# ---------------------------------------------------------------------------
# Wiener filter
# ---------------------------------------------------------------------------


def wiener_filter(
    gray: np.ndarray,
    psf_model: str = "gaussian",
    sigma: float = 1.5,
    length: float = 9.0,
    angle: float = 0.0,
    nsr: float = 0.05,
    strength: float = 0.8,
) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    psf = motion_psf(length, angle) if psf_model == "motion" else gaussian_psf(sigma)
    pad = max(psf.shape)
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT).astype(np.float64)
    H = np.fft.fft2(psf, s=padded.shape)
    Hc = np.conj(H)
    W = Hc / (np.abs(H) ** 2 + max(float(nsr), 1e-6))
    restored = np.fft.ifft2(np.fft.fft2(padded) * W).real
    restored = restored[pad:-pad, pad:-pad]
    restored = np.nan_to_num(restored, nan=0.0, posinf=255.0, neginf=0.0)
    restored = _blend_strength(gray, restored, strength)
    restored = np.clip(restored, 0.0, 255.0).astype(np.uint8)
    before = _sharpness(gray)
    after = _sharpness(restored)
    overlay = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
    return {
        "processed": restored,
        "overlay": overlay,
        "metrics": {
            "sharpness_before": round(before, 2),
            "sharpness_after": round(after, 2),
            "clipped_ratio": round(_clipped_ratio(restored), 4),
            "clip_added": round(_clip_added(gray, restored), 4),
            "degraded": bool(after < before),
            "psf": psf_model,
            "nsr": round(float(nsr), 4),
            "strength": round(float(strength), 2),
        },
        "psf_estimated": None,
    }


# ---------------------------------------------------------------------------
# Blind (semi-blind) deconvolution via Richardson-Lucy
# ---------------------------------------------------------------------------


def richardson_lucy(
    gray: np.ndarray,
    psf: np.ndarray,
    iterations: int = 30,
) -> tuple[np.ndarray, int, bool]:
    """RL deconvolution with numerical guards. Returns (restored, run_iters, converged)."""
    obs = np.clip(gray.astype(np.float64), 0.0, None)
    est = obs.copy()
    psf = psf / max(float(psf.sum()), 1e-12)
    psf_flip = psf[::-1, ::-1]
    upper = max(float(obs.max()) * 2.0, 255.0)
    converged = False
    run = 0
    prev_change = float("inf")
    for run in range(1, int(iterations) + 1):
        denom = np.maximum(_fft_convolve(est, psf), 1e-8)
        ratio = obs / denom
        new_est = est * _fft_convolve(ratio, psf_flip)
        new_est = np.clip(new_est, 0.0, upper)
        if not np.all(np.isfinite(new_est)):
            run -= 1
            break
        change = float(np.abs(new_est - est).mean() / max(est.mean(), 1e-9))
        est = new_est
        if change < 1e-4 and prev_change < 1e-4:
            converged = True
            break
        prev_change = change
    return est, run, converged


def blind_deconvolution(
    gray: np.ndarray,
    psf_model: str = "gaussian",
    sigma: float = 1.5,
    length: float = 9.0,
    angle: float = 0.0,
    iterations: int = 8,
    estimate_psf: bool = False,
    post_smooth: float = 1.5,
    strength: float = 0.5,
) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    psf_estimated = None
    if estimate_psf:
        psf_estimated = estimate_psf_params(gray, psf_model)
        if psf_model == "motion":
            length = float(psf_estimated["length"])
            angle = float(psf_estimated["angle"])
        else:
            sigma = float(psf_estimated["sigma"])
    psf = motion_psf(length, angle) if psf_model == "motion" else gaussian_psf(sigma)
    pad = max(psf.shape)
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    restored, run, converged = richardson_lucy(padded, psf, iterations)
    restored = restored[pad:-pad, pad:-pad]
    if float(post_smooth) > 0:
        restored = cv2.GaussianBlur(
            np.clip(restored, 0.0, 255.0).astype(np.uint8), (0, 0), float(post_smooth)
        ).astype(np.float64)
    restored = _blend_strength(gray, restored, strength)
    restored_u8 = np.clip(restored, 0.0, 255.0).astype(np.uint8)
    before = _sharpness(gray)
    after = _sharpness(restored_u8)
    overlay = cv2.cvtColor(restored_u8, cv2.COLOR_GRAY2BGR)
    return {
        "processed": restored_u8,
        "overlay": overlay,
        "metrics": {
            "sharpness_before": round(before, 2),
            "sharpness_after": round(after, 2),
            "clipped_ratio": round(_clipped_ratio(restored_u8), 4),
            "clip_added": round(_clip_added(gray, restored_u8), 4),
            "degraded": bool(after < before),
            "iterations_run": run,
            "converged": bool(converged),
            "psf": psf_model,
            "psf_sigma": round(float(sigma), 2) if psf_model == "gaussian" else None,
            "psf_length": round(float(length), 1) if psf_model == "motion" else None,
            "psf_angle": round(float(angle), 1) if psf_model == "motion" else None,
            "post_smooth": round(float(post_smooth), 2),
            "strength": round(float(strength), 2),
        },
        "psf_estimated": psf_estimated,
    }


# ---------------------------------------------------------------------------
# First-order derivative extremum edge detection
# ---------------------------------------------------------------------------


def _nms(gm: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Vectorized non-maximum suppression along gradient direction."""
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    q = np.zeros(gm.shape, np.uint8)
    q[(angle >= 22.5) & (angle < 67.5)] = 1   # 45 deg
    q[(angle >= 67.5) & (angle < 112.5)] = 2  # 90 deg
    q[(angle >= 112.5) & (angle < 157.5)] = 3 # 135 deg
    p = np.pad(gm, 1)
    n1 = np.zeros_like(gm)
    n2 = np.zeros_like(gm)
    n1[q == 0] = p[1:-1, 0:-2][q == 0]
    n2[q == 0] = p[1:-1, 2:][q == 0]
    n1[q == 1] = p[0:-2, 2:][q == 1]
    n2[q == 1] = p[2:, 0:-2][q == 1]
    n1[q == 2] = p[0:-2, 1:-1][q == 2]
    n2[q == 2] = p[2:, 1:-1][q == 2]
    n1[q == 3] = p[0:-2, 0:-2][q == 3]
    n2[q == 3] = p[2:, 2:][q == 3]
    return np.where((gm >= n1) & (gm >= n2), gm, 0.0)


def first_derivative_edges(
    gray: np.ndarray,
    operator: str = "sobel",
    direction: str = "mag",
    sigma: float = 0.0,
    threshold: float = 40.0,
    nms: bool = True,
    binary_output: bool = False,
) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    src = cv2.GaussianBlur(gray, (0, 0), float(sigma)) if sigma > 0 else gray
    if operator == "scharr":
        gx = cv2.Scharr(src, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(src, cv2.CV_64F, 0, 1)
    elif operator == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float64)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float64)
        gx = cv2.filter2D(src, cv2.CV_64F, kx)
        gy = cv2.filter2D(src, cv2.CV_64F, ky)
    else:  # sobel
        gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
    if direction == "x":
        g = np.abs(gx)
    elif direction == "y":
        g = np.abs(gy)
    else:
        g = np.hypot(gx, gy)
        if nms:
            g = _nms(g, gx, gy)
    edges = (g >= max(float(threshold), 0.0)).astype(np.uint8) * 255
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[edges > 0] = (0, 0, 255)
    return {
        "processed": _visual_edge(gray, edges, bool(binary_output)),
        "overlay": overlay,
        "metrics": {
            "edge_count": int(np.count_nonzero(edges)),
            "mean_gradient": round(float(g[edges > 0].mean()) if np.any(edges) else 0.0, 3),
            "operator": operator,
            "direction": direction,
            "binary_output": bool(binary_output),
        },
        "psf_estimated": None,
    }


# ---------------------------------------------------------------------------
# Second-order derivative extremum (LoG zero crossing) edge detection
# ---------------------------------------------------------------------------


def second_derivative_edges(
    gray: np.ndarray,
    sigma: float = 1.0,
    slope_threshold: float = 8.0,
    binary_output: bool = False,
) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    src = cv2.GaussianBlur(gray, (0, 0), float(sigma)) if sigma > 0 else gray
    lap = cv2.Laplacian(src, cv2.CV_64F, ksize=3)
    h, w = lap.shape
    padded = np.pad(lap, ((1, 1), (1, 1)))
    out = np.zeros((h, w), np.uint8)
    thr = max(float(slope_threshold), 0.0)
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = padded[1:h + 1, 1:w + 1]
        b = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
        cross = (a * b < 0) & (np.abs(a - b) >= thr)
        out[cross] = 255
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[out > 0] = (0, 0, 255)
    return {
        "processed": _visual_edge(gray, out, bool(binary_output)),
        "overlay": overlay,
        "metrics": {
            "edge_count": int(np.count_nonzero(out)),
            "sigma": round(float(sigma), 2),
            "binary_output": bool(binary_output),
        },
        "psf_estimated": None,
    }


# ---------------------------------------------------------------------------
# Zernike moment subpixel edge extraction
# ---------------------------------------------------------------------------

def _radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial Zernike polynomial R_nm(rho) for the orders used here."""
    if (n, m) == (1, 1):
        return rho
    if (n, m) == (2, 0):
        return 2.0 * rho ** 2 - 1.0
    raise ValueError(f"unsupported Zernike order ({n},{m})")


@lru_cache(maxsize=8)
def _area_zernike_kernel(n_mask: int, order: int, m: int, real: bool, sub: int = 16) -> np.ndarray:
    """Zernike moment kernel by numerical area integration.

    Each coefficient integrates R_nm(rho) * cos/sin(m*theta) over the pixel's
    square clipped to the unit disk, with the continuous orthonormal
    normalization ((n+1)/pi).  This reproduces the classical 5x5 kernels and
    extends consistently to 7x7 (validated on synthetic step edges).
    """
    half = n_mask / 2.0
    kernel = np.zeros((n_mask, n_mask))
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for i in range(n_mask):
        for j in range(n_mask):
            cx = i - (n_mask - 1) / 2.0
            cy = j - (n_mask - 1) / 2.0
            val = 0.0
            for oy in offs:
                for ox in offs:
                    x, y = cx + ox, cy + oy
                    rho = np.hypot(x, y) / half
                    if rho > 1.0:
                        continue
                    theta = np.arctan2(y, x)
                    r = _radial_poly(order, m, rho)
                    if m == 0:
                        v = r
                    elif real:
                        v = r * np.cos(m * theta)
                    else:
                        v = r * np.sin(m * theta)
                    val += v
            kernel[j, i] = val * ((order + 1) / np.pi) * (1.0 / (half * half)) * (1.0 / sub ** 2)
    return kernel


def _zernike_kernels(mask: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 7 if int(mask) == 7 else 5
    k11r = _area_zernike_kernel(n, 1, 1, True)
    k11i = _area_zernike_kernel(n, 1, 1, False)
    k20 = _area_zernike_kernel(n, 2, 0, True)
    return k11r, k11i, k20


def zernike_edges(
    gray: np.ndarray,
    mask: int = 5,
    threshold: float = 40.0,
    subpixel: bool = True,
    edge_width: float = 0.5,
    binary_output: bool = False,
) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    mask = 7 if int(mask) == 7 else 5
    k11r, k11i, k20 = _zernike_kernels(mask)
    half = mask // 2
    h, w = gray.shape
    if h < mask or w < mask:
        raise ValueError(f"ROI 太小，Zernike 需要至少 {mask}x{mask} 像素")

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    cand = gm >= max(float(threshold), 1.0)

    points: list[list[float]] = []
    ew = max(float(edge_width), 0.0)
    ew2 = ew * ew
    radius_px = mask / 2.0
    for y in range(half, h - half):
        for x in range(half, w - half):
            if not cand[y, x]:
                continue
            win = gray[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
            r11 = float(np.sum(win * k11r))
            i11 = float(np.sum(win * k11i))
            z20 = float(np.sum(win * k20))
            mag = np.hypot(r11, i11)
            if mag < 1e-9:
                continue
            angle = np.arctan2(i11, r11)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            ratio = z20 / mag
            if ew > 0.01:
                disc = (ew2 - 1.0) ** 2 - 2.0 * ew2 * ratio
                if disc < 0:
                    continue
                location = (1.0 - ew2 - np.sqrt(disc)) / ew2
            else:
                location = ratio
            if abs(location) >= 0.9:
                continue
            if subpixel:
                px = x + radius_px * location * cos_a
                py = y + radius_px * location * sin_a
            else:
                px, py = float(x), float(y)
            points.append([round(px, 2), round(py, 2)])

    edge_map = np.zeros((h, w), np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for px, py in points:
        xi, yi = int(round(px)), int(round(py))
        if 0 <= xi < w and 0 <= yi < h:
            edge_map[yi, xi] = 255
            cv2.circle(overlay, (xi, yi), 1, (0, 255, 0), -1)
    return {
        "processed": _visual_edge(gray, edge_map, bool(binary_output)),
        "overlay": overlay,
        "metrics": {
            "point_count": len(points),
            "mask": mask,
            "subpixel": bool(subpixel),
            "binary_output": bool(binary_output),
        },
        "psf_estimated": None,
    }


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

ALGORITHMS = ("wiener", "blind_deconv", "deriv1", "deriv2", "zernike", "hat", "restormer")


def process_roi(gray: np.ndarray, algorithm: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    gray = _as_gray_u8(gray)
    if gray.size < 25 or min(gray.shape) < 5:
        raise ValueError("ROI 太小（至少 5x5 像素）")
    params = params or {}
    if algorithm == "wiener":
        return wiener_filter(
            gray,
            psf_model=params.get("psf_model", "gaussian"),
            sigma=float(params.get("sigma", 1.5)),
            length=float(params.get("length", 9.0)),
            angle=float(params.get("angle", 0.0)),
            nsr=float(params.get("nsr", 0.05)),
            strength=float(params.get("strength", 0.8)),
        )
    if algorithm == "blind_deconv":
        return blind_deconvolution(
            gray,
            psf_model=params.get("psf_model", "gaussian"),
            sigma=float(params.get("sigma", 1.5)),
            length=float(params.get("length", 9.0)),
            angle=float(params.get("angle", 0.0)),
            iterations=int(params.get("iterations", 8)),
            estimate_psf=bool(params.get("estimate_psf", False)),
            post_smooth=float(params.get("post_smooth", 1.5)),
            strength=float(params.get("strength", 0.5)),
        )
    if algorithm == "deriv1":
        return first_derivative_edges(
            gray,
            operator=params.get("operator", "sobel"),
            direction=params.get("direction", "mag"),
            sigma=float(params.get("sigma", 0.0)),
            threshold=float(params.get("threshold", 40.0)),
            nms=bool(params.get("nms", True)),
            binary_output=bool(params.get("binary_output", False)),
        )
    if algorithm == "deriv2":
        return second_derivative_edges(
            gray,
            sigma=float(params.get("sigma", 1.0)),
            slope_threshold=float(params.get("slope_threshold", 8.0)),
            binary_output=bool(params.get("binary_output", False)),
        )
    if algorithm == "zernike":
        return zernike_edges(
            gray,
            mask=int(params.get("mask", 5)),
            threshold=float(params.get("threshold", 40.0)),
            subpixel=bool(params.get("subpixel", True)),
            edge_width=float(params.get("edge_width", 0.5)),
            binary_output=bool(params.get("binary_output", False)),
        )
    if algorithm == "hat":
        from roi_dl import hat_enhance
        return hat_enhance(gray, params)
    if algorithm == "restormer":
        from roi_dl import restormer_enhance
        return restormer_enhance(gray, params)
    raise ValueError(f"未知算法: {algorithm}")




