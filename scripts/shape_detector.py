#!/usr/bin/env python3
"""Polygon shape detection for rectangles, chamfered rectangles, trapezoids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from image_preprocess import prepare_image


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


@dataclass
class RobustLine:
    origin: np.ndarray
    direction: np.ndarray
    points: np.ndarray
    residual: float


def _fit_robust_line(points: np.ndarray) -> RobustLine | None:
    if points.shape[0] < 4:
        return None

    points = points.astype(np.float64)
    inliers = np.ones(points.shape[0], dtype=bool)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    origin = points.mean(axis=0)

    for _ in range(8):
        work = points[inliers]
        if work.shape[0] < 4:
            break
        weights = np.ones(work.shape[0], dtype=np.float64)
        if work.shape[0] > 4:
            centered = work - work.mean(axis=0)
            covariance = (centered * weights[:, None]).T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            direction = eigenvectors[:, int(np.argmax(eigenvalues))]
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
        origin = np.average(work, axis=0, weights=weights)
        residuals = np.abs(
            direction[0] * (points[:, 1] - origin[1])
            - direction[1] * (points[:, 0] - origin[0])
        )
        median_residual = float(np.median(residuals))
        deviation = float(
            np.median(np.abs(residuals - median_residual))
        )
        scale = max(0.6, 1.4826 * deviation)
        cutoff = max(1.5, 2.8 * scale + 0.5)
        next_inliers = residuals <= cutoff
        if np.count_nonzero(next_inliers) < 4:
            break
        if np.array_equal(next_inliers, inliers):
            inliers = next_inliers
            break
        inliers = next_inliers

    work = points[inliers]
    if work.shape[0] < 4:
        return None
    origin = work.mean(axis=0)
    centered = work - origin
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    residuals = np.abs(
        direction[0] * (work[:, 1] - origin[1])
        - direction[1] * (work[:, 0] - origin[0])
    )
    return RobustLine(
        origin=origin,
        direction=direction,
        points=work,
        residual=float(np.median(residuals)),
    )


def _side_points(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    side = end - start
    length = float(np.linalg.norm(side))
    if length <= 1e-6:
        return np.empty((0, 2), dtype=np.float64)
    direction = side / length
    normal = np.array([-direction[1], direction[0]])
    midpoint = (start + end) / 2.0
    if float(np.dot(normal, midpoint - center)) < 0:
        normal = -normal

    relative = points - midpoint
    distance = np.abs(relative @ normal)
    projection = relative @ direction
    band = max(3.0, min(14.0, 0.035 * length))
    selected = (
        (distance <= band)
        & (projection >= -band)
        & (projection <= length + band)
    )
    if np.count_nonzero(selected) >= 4:
        return points[selected]

    nearest = np.argsort(distance)
    return points[nearest[: min(points.shape[0], 32)]]


def _line_intersection(
    normal_a: np.ndarray,
    value_a: float,
    normal_b: np.ndarray,
    value_b: float,
) -> np.ndarray | None:
    matrix = np.vstack((normal_a, normal_b))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        return None
    return np.linalg.solve(matrix, np.array([value_a, value_b]))


def _fit_constrained_rectangle(
    points: np.ndarray,
    contour_area: float,
) -> ShapeResult | None:
    if points.shape[0] < 16:
        return None

    initial_box = cv2.minAreaRect(points.astype(np.float32))
    box_points = cv2.boxPoints(initial_box).astype(np.float64)
    center = box_points.mean(axis=0)
    side_fits: list[RobustLine] = []
    side_coverages: list[float] = []
    for index in range(4):
        start = box_points[index]
        end = box_points[(index + 1) % 4]
        selected = _side_points(points, start, end, center)
        fitted = _fit_robust_line(selected)
        if fitted is None:
            return None
        side_fits.append(fitted)
        side_vector = end - start
        side_length = float(np.linalg.norm(side_vector))
        side_direction = side_vector / max(side_length, 1e-9)
        projection = (fitted.points - start) @ side_direction
        coverage = (
            float(np.clip((projection.max() - projection.min()) / side_length, 0.0, 1.0))
            if projection.size
            else 0.0
        )
        side_coverages.append(coverage)

    def aligned(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return second if float(np.dot(first, second)) >= 0 else -second

    direction_a = side_fits[0].direction + aligned(
        side_fits[0].direction,
        side_fits[2].direction,
    )
    direction_a /= max(float(np.linalg.norm(direction_a)), 1e-9)
    direction_b = np.array([-direction_a[1], direction_a[0]])
    raw_b = side_fits[1].direction + aligned(
        side_fits[1].direction,
        side_fits[3].direction,
    )
    if float(np.dot(direction_b, raw_b)) < 0:
        direction_b = -direction_b

    directions = [direction_a, direction_b, direction_a, direction_b]
    normals: list[np.ndarray] = []
    values: list[float] = []
    residuals: list[float] = []
    for index, fitted in enumerate(side_fits):
        normal = np.array(
            [-directions[index][1], directions[index][0]],
            dtype=np.float64,
        )
        side_midpoint = (
            box_points[index] + box_points[(index + 1) % 4]
        ) / 2.0
        if float(np.dot(normal, side_midpoint - center)) < 0:
            normal = -normal
        value = float(np.median(fitted.points @ normal))
        normals.append(normal)
        values.append(value)
        residuals.append(float(np.median(np.abs(fitted.points @ normal - value))))

    corners: list[np.ndarray] = []
    for index in range(4):
        corner = _line_intersection(
            normals[index],
            values[index],
            normals[(index + 1) % 4],
            values[(index + 1) % 4],
        )
        if corner is None:
            return None
        corners.append(corner)

    polygon = np.asarray(corners, dtype=np.float64)
    side_lengths = np.linalg.norm(
        np.roll(polygon, -1, axis=0) - polygon,
        axis=1,
    )
    if float(np.min(side_lengths)) < 5.0:
        return None
    polygon_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
    if polygon_area <= 1.0:
        return None

    diagonal = float(np.linalg.norm(polygon[2] - polygon[0]))
    line_score = max(
        0.0,
        1.0 - float(np.mean(residuals)) / max(2.0, 0.015 * diagonal),
    )
    coverage_score = float(
        np.mean([min(1.0, coverage / 0.55) for coverage in side_coverages])
    )
    side_support = float(
        np.mean([min(1.0, fit.points.shape[0] / 12.0) for fit in side_fits])
    )
    score = max(0.0, min(1.0, line_score * side_support * coverage_score))
    width = float((side_lengths[0] + side_lengths[2]) / 2.0)
    height = float((side_lengths[1] + side_lengths[3]) / 2.0)
    angle = float(np.degrees(np.arctan2(direction_a[1], direction_a[0])))
    fill_ratio = contour_area / polygon_area if polygon_area > 0 else 0.0
    return ShapeResult(
        shape="rectangle",
        x=float(polygon[:, 0].mean()),
        y=float(polygon[:, 1].mean()),
        width=max(width, height),
        height=min(width, height),
        angle=angle,
        area=polygon_area,
        score=score,
        vertices=4,
        fill_ratio=fill_ratio,
        points=polygon.tolist(),
    )


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
            np.hypot(result.x - other.x, result.y - other.y)
            > max(10.0, 0.15 * min(result.width, result.height))
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

    prepared = prepare_image(
        gray,
        blur_size=blur_size,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    edges = prepared.edges
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
        contour_points = contour.reshape(-1, 2).astype(np.float64)
        if shape_type == "rectangle":
            result = _fit_constrained_rectangle(contour_points, area)
            if result is None:
                continue
            result_points = np.asarray(result.points, dtype=np.float64)
            if (
                np.any(result_points[:, 0] < 0)
                or np.any(result_points[:, 1] < 0)
                or np.any(result_points[:, 0] >= width)
                or np.any(result_points[:, 1] >= height)
                or result.score < 0.25
            ):
                continue
            if fill_mode == "filled" and result.fill_ratio < min_fill_ratio:
                continue
            results.append(result)
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
