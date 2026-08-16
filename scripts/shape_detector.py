#!/usr/bin/env python3
"""Polygon shape detection for rectangles, chamfered rectangles, trapezoids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class ShapeResult:
    shape: str
    x: float
    y: float
    width: float
    height: float
    angle: float
    area: float
    score: float
    vertices: int
    fill_ratio: float = 1.0
    chamfer: float | None = None
    top_width: float | None = None
    bottom_width: float | None = None
    points: list[list[float]] = field(default_factory=list)

    @property
    def radius(self) -> float:
        return max(self.width, self.height) / 2.0

    def as_dict(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "shape": self.shape,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "angle": round(self.angle, 2),
            "area": round(self.area, 2),
            "fill_ratio": round(self.fill_ratio, 3),
            "score": round(self.score, 4),
            "vertices": self.vertices,
            "chamfer": round(self.chamfer, 3) if self.chamfer is not None else None,
            "top_width": (
                round(self.top_width, 3) if self.top_width is not None else None
            ),
            "bottom_width": (
                round(self.bottom_width, 3)
                if self.bottom_width is not None
                else None
            ),
            "points": self.points,
        }


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm == 0:
        return 90.0
    cosine = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _parallel_score(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return abs(float(np.dot(v1, v2)) / (norm1 * norm2))


def _rectangle_from_points(points: np.ndarray) -> ShapeResult | None:
    if len(points) != 4:
        return None
    vectors = np.roll(points, -1, axis=0) - points
    angles = [
        _angle_deg(vectors[index], vectors[(index + 1) % 4])
        for index in range(4)
    ]
    max_deviation = max(abs(angle - 90.0) for angle in angles)
    if max_deviation > 18.0:
        return None
    rect = cv2.minAreaRect(points.astype(np.float32))
    (center_x, center_y), (box_w, box_h), angle = rect
    width, height = max(box_w, box_h), min(box_w, box_h)
    score = max(0.0, 1.0 - max_deviation / 90.0)
    return ShapeResult(
        shape="rectangle",
        x=float(center_x),
        y=float(center_y),
        width=float(width),
        height=float(height),
        angle=float(angle),
        area=float(cv2.contourArea(points.astype(np.int32))),
        score=score,
        vertices=4,
        points=points.astype(float).tolist(),
    )


def _chamfered_rectangle_from_points(points: np.ndarray) -> ShapeResult | None:
    if len(points) != 8:
        return None
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(vectors, axis=1)
    sorted_lengths = np.sort(lengths)
    short_lengths = sorted_lengths[:4]
    long_lengths = sorted_lengths[4:]
    if (
        len(short_lengths) != 4
        or len(long_lengths) != 4
        or float(long_lengths.mean()) / max(float(short_lengths.mean()), 1e-9) < 1.5
    ):
        return None

    rect = cv2.minAreaRect(points.astype(np.float32))
    (center_x, center_y), (box_w, box_h), angle = rect
    width, height = max(box_w, box_h), min(box_w, box_h)
    chamfer = float(short_lengths.mean())
    score = 0.9
    return ShapeResult(
        shape="chamfered_rectangle",
        x=float(center_x),
        y=float(center_y),
        width=float(width),
        height=float(height),
        angle=float(angle),
        area=float(cv2.contourArea(points.astype(np.int32))),
        score=score,
        vertices=8,
        chamfer=chamfer,
        points=points.astype(float).tolist(),
    )


def _trapezoid_from_points(points: np.ndarray) -> ShapeResult | None:
    if len(points) != 4:
        return None
    vectors = np.roll(points, -1, axis=0) - points
    pairs = [(0, 2), (1, 3)]
    parallel_pairs = [
        (index_a, index_b)
        for index_a, index_b in pairs
        if _parallel_score(vectors[index_a], vectors[index_b]) > 0.92
    ]
    if len(parallel_pairs) != 1:
        return None

    base_index, opposite_index = parallel_pairs[0]
    base_points = points[[base_index, (base_index + 1) % 4]]
    opposite_points = points[[opposite_index, (opposite_index + 1) % 4]]
    base_width = float(np.linalg.norm(vectors[base_index]))
    opposite_width = float(np.linalg.norm(vectors[opposite_index]))
    center = points.mean(axis=0)
    height = float(np.linalg.norm(opposite_points.mean(axis=0) - base_points.mean(axis=0)))
    base_angle = float(np.degrees(np.arctan2(vectors[base_index][1], vectors[base_index][0])))
    return ShapeResult(
        shape="trapezoid",
        x=float(center[0]),
        y=float(center[1]),
        width=max(base_width, opposite_width),
        height=height,
        angle=base_angle,
        area=float(cv2.contourArea(points.astype(np.int32))),
        score=0.9,
        vertices=4,
        top_width=min(base_width, opposite_width),
        bottom_width=max(base_width, opposite_width),
        points=points.astype(float).tolist(),
    )


def _deduplicate(results: list[ShapeResult]) -> list[ShapeResult]:
    kept: list[ShapeResult] = []
    for result in sorted(results, key=lambda item: item.area, reverse=True):
        if all(
            np.hypot(result.x - other.x, result.y - other.y) > 10
            for other in kept
        ):
            kept.append(result)
    return kept


def detect_shapes(
    gray: np.ndarray,
    shape_type: str,
    blur_size: int = 1,
    canny_low: int = 50,
    canny_high: int = 150,
    min_area: float = 100.0,
    max_area: float = 100000000.0,
    min_side: float = 5.0,
    max_side: float = 10000.0,
    min_fill_ratio: float = 0.2,
    fill_mode: str = "auto",
) -> list[ShapeResult]:
    if shape_type not in ("rectangle", "chamfered_rectangle", "trapezoid"):
        return []

    blurred = cv2.medianBlur(gray, max(1, blur_size) | 1)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = gray.shape
    results: list[ShapeResult] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < min_area or area > max_area:
            continue
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        if min(w, h) < min_side or max(w, h) > max_side:
            continue
        box = cv2.minAreaRect(contour)
        box_area = float(box[1][0] * box[1][1])
        fill_ratio = area / box_area if box_area > 0 else 0.0
        if fill_mode == "filled" and fill_ratio < min_fill_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        for epsilon_factor in (0.012, 0.02, 0.03):
            approx = cv2.approxPolyDP(
                contour, epsilon_factor * perimeter, True
            ).reshape(-1, 2).astype(np.float64)
            if not cv2.isContourConvex(approx.astype(np.int32)):
                continue
            approx_perimeter = cv2.arcLength(approx.astype(np.int32), True)
            perimeter_ratio = (
                perimeter / approx_perimeter if approx_perimeter > 0 else 0.0
            )
            if perimeter_ratio < 0.6 or perimeter_ratio > 2.0:
                continue
            if fill_mode == "outline":
                pass
            result = None
            if shape_type == "rectangle" and len(approx) == 4:
                result = _rectangle_from_points(approx)
            elif shape_type == "chamfered_rectangle" and len(approx) == 8:
                result = _chamfered_rectangle_from_points(approx)
            elif shape_type == "trapezoid" and len(approx) == 4:
                result = _trapezoid_from_points(approx)
            if result is not None:
                result.fill_ratio = fill_ratio
                results.append(result)
                break

    return _deduplicate(results)
