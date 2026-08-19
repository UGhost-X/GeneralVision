"""Shared preprocessing for low-texture and specular inspection images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreparedImage:
    gray: np.ndarray
    clahe: np.ndarray
    fused: np.ndarray
    highlight_mask: np.ndarray
    gradient: np.ndarray
    edges: np.ndarray
    variants: tuple[np.ndarray, ...]


def _as_gray_u8(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image.copy()
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _gamma_variant(gray: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(lut, gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.LUT(gray, lut)


def _highlight_mask(gray: np.ndarray) -> np.ndarray:
    if int(gray.max()) < 235:
        return np.zeros_like(gray, dtype=np.uint8)

    percentile = float(np.percentile(gray, 99.5))
    threshold = int(np.clip(max(245.0, percentile), 245.0, 252.0))
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _exposure_fusion(
    gray: np.ndarray, clahe: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    dark = _gamma_variant(gray, 1.55)
    bright = _gamma_variant(gray, 0.72)
    exposures = [gray, dark, bright]

    try:
        bgr_exposures = [
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in exposures
        ]
        merged = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=0.0,
            exposure_weight=1.0,
        ).process(bgr_exposures)
        fused = np.clip(
            cv2.cvtColor((merged * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY),
            0,
            255,
        ).astype(np.uint8)
    except (AttributeError, cv2.error, ValueError):
        # Keep a deterministic fallback for OpenCV builds without Mertens.
        fused = gray.copy()
        shadow = gray < 48
        highlight = gray > 205
        fused[shadow] = bright[shadow]
        fused[highlight] = dark[highlight]

    fused = cv2.addWeighted(fused, 0.65, clahe, 0.35, 0.0)
    return fused, (gray, clahe, fused, dark, bright)


def _multiscale_gradient(variants: tuple[np.ndarray, ...]) -> np.ndarray:
    gradient = np.zeros_like(variants[0], dtype=np.float32)
    for variant in variants[:3]:
        for sigma in (0.8, 1.6, 3.2):
            smoothed = cv2.GaussianBlur(variant, (0, 0), sigmaX=sigma)
            grad_x = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
            grad_y = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
            gradient = np.maximum(gradient, cv2.magnitude(grad_x, grad_y))
    return cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )


def _multiscale_edges(
    variants: tuple[np.ndarray, ...],
    gradient: np.ndarray,
    highlight_mask: np.ndarray,
    blur_size: int,
    canny_low: int,
    canny_high: int,
) -> np.ndarray:
    edge_maps: list[np.ndarray] = []
    for variant in variants[:3]:
        for sigma in (0.0, 1.2, 2.4):
            source = (
                variant
                if sigma == 0.0
                else cv2.GaussianBlur(variant, (0, 0), sigmaX=sigma)
            )
            if blur_size > 1:
                source = cv2.medianBlur(source, blur_size)
            edge_maps.append(
                cv2.Canny(
                    source,
                    canny_low,
                    canny_high,
                    L2gradient=True,
                )
            )

    gradient_threshold = max(18, int(canny_low * 0.65))
    edge_maps.append(
        np.where(gradient >= gradient_threshold, 255, 0).astype(np.uint8)
    )
    edges = np.maximum.reduce(edge_maps)

    # Remove saturated interiors while keeping the boundary around them.
    interior = cv2.erode(
        highlight_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    edges[interior > 0] = 0
    return edges


def prepare_image(
    gray: np.ndarray,
    blur_size: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
) -> PreparedImage:
    base = _as_gray_u8(gray)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(base)
    fused, variants = _exposure_fusion(base, clahe)
    highlight_mask = _highlight_mask(base)
    gradient = _multiscale_gradient(variants)
    edges = _multiscale_edges(
        variants,
        gradient,
        highlight_mask,
        max(1, int(blur_size) | 1),
        max(1, int(canny_low)),
        max(1, int(canny_high)),
    )
    return PreparedImage(
        gray=base,
        clahe=clahe,
        fused=fused,
        highlight_mask=highlight_mask,
        gradient=gradient,
        edges=edges,
        variants=variants,
    )
