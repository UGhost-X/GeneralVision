#!/usr/bin/env python3
"""Detect and fit circular holes in an image.

The script uses OpenCV Hough circle detection to find candidates, then refines
each candidate with least-squares circle and ellipse fits on Canny edge points.
Results include roundness/ellipticity and are written as CSV, JSON, and an
overlay PNG next to the input image.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE = Path(
    r"C:\Users\UGhost-X\Desktop\139c26a4\raw_image_1_ann1.png"
)


@dataclass
class FittedCircle:
    x: float
    y: float
    radius: float
    major_radius: float
    minor_radius: float
    ellipse_angle: float
    roundness: float
    shape: str
    residual: float
    ellipse_residual: float
    support: float
    contrast: float
    inside_mean: float
    outside_mean: float
    edge_sharpness: float
    inside_edge_density: float
    edge_points: int

    @property
    def score(self) -> float:
        return self.support * (1.0 - min(self.residual, 0.5))

    @property
    def diameter(self) -> float:
        return self.radius * 2.0

    @property
    def ellipticity(self) -> float:
        return 1.0 - self.roundness

    @property
    def residual_pixels(self) -> float:
        return self.residual * self.radius

    @property
    def ellipse_residual_pixels(self) -> float:
        return self.ellipse_residual * (
            (self.major_radius + self.minor_radius) / 2.0
        )

    def as_dict(self, index: int) -> dict[str, float | int | str]:
        return {
            "index": index,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "radius": round(self.radius, 3),
            "diameter": round(self.diameter, 3),
            "major_radius": round(self.major_radius, 3),
            "minor_radius": round(self.minor_radius, 3),
            "ellipse_angle": round(self.ellipse_angle, 2),
            "roundness": round(self.roundness, 3),
            "ellipticity": round(self.ellipticity, 3),
            "shape": self.shape,
            "residual": round(self.residual, 5),
            "residual_px": round(self.residual_pixels, 3),
            "ellipse_residual": round(self.ellipse_residual, 5),
            "ellipse_residual_px": round(self.ellipse_residual_pixels, 3),
            "inside_mean": round(self.inside_mean, 2),
            "outside_mean": round(self.outside_mean, 2),
            "edge_sharpness": round(self.edge_sharpness, 3),
            "inside_edge_density": round(self.inside_edge_density, 3),
            "support": round(self.support, 3),
            "contrast": round(self.contrast, 2),
            "score": round(self.score, 5),
            "edge_points": self.edge_points,
        }


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def fit_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    """Fit a circle using the algebraic least-squares solution."""
    if points.shape[0] < 6:
        return None

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    target = x * x + y * y

    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    center_x, center_y, constant = solution
    radius_sq = center_x * center_x + center_y * center_y + constant
    if radius_sq <= 0:
        return None

    return center_x, center_y, float(np.sqrt(radius_sq))


def fit_ellipse(
    points: np.ndarray, tolerance: float = 0.12
) -> tuple[float, float, float, float, float] | None:
    """Fit an ellipse and return center, major radius, minor radius, angle."""
    if points.shape[0] < 5:
        return None

    points = points.astype(np.float32)
    try:
        ellipse = cv2.fitEllipse(points)
    except cv2.error:
        return None

    for _ in range(3):
        (center_x, center_y), (axis_a, axis_b), angle = ellipse
        major_radius = max(axis_a, axis_b) / 2.0
        minor_radius = min(axis_a, axis_b) / 2.0
        if major_radius <= 0 or minor_radius <= 0:
            return None

        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        dx = points[:, 0] - center_x
        dy = points[:, 1] - center_y
        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy
        normalized = np.hypot(
            x_rot / major_radius, y_rot / minor_radius
        )
        inliers = np.abs(normalized - 1.0) <= tolerance
        if np.count_nonzero(inliers) < 5:
            break
        if np.count_nonzero(inliers) == points.shape[0]:
            break

        points = points[inliers]
        try:
            ellipse = cv2.fitEllipse(points)
        except cv2.error:
            break

    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    major_radius = max(axis_a, axis_b) / 2.0
    minor_radius = min(axis_a, axis_b) / 2.0
    # cv2.fitEllipse returns the orientation of the first/minor axis.  Convert it
    # to the major-axis angle so callers can draw with (major_radius, minor_radius).
    major_angle = (angle - 90.0) % 180.0
    return center_x, center_y, major_radius, minor_radius, major_angle


def ellipse_support(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    major_radius: float,
    minor_radius: float,
    angle: float,
    tolerance: float = 0.15,
) -> tuple[float, float, int]:
    """Compute angular support and residual for an ellipse fit."""
    if points.shape[0] < 5 or major_radius <= 0 or minor_radius <= 0:
        return 0.0, 1.0, 0

    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    x_rot = cos_a * dx + sin_a * dy
    y_rot = -sin_a * dx + cos_a * dy
    normalized = np.hypot(
        x_rot / major_radius, y_rot / minor_radius
    )
    inlier_mask = np.abs(normalized - 1.0) <= tolerance
    inliers = points[inlier_mask]

    angles = np.degrees(np.arctan2(dy, dx)) % 360.0
    bins = np.zeros(36, dtype=np.int32)
    for angle_value in angles[inlier_mask]:
        bins[int(angle_value // 10.0) % 36] += 1
    support = float(np.count_nonzero(bins)) / 36.0
    residual = (
        float(np.mean(np.abs(normalized[inlier_mask] - 1.0)))
        if np.count_nonzero(inlier_mask)
        else 1.0
    )
    return support, residual, int(np.count_nonzero(inlier_mask))


def refine_circle(
    gray: np.ndarray,
    edges: np.ndarray,
    candidate: tuple[float, float, float],
    roundness_threshold: float,
) -> FittedCircle | None:
    center_x0, center_y0, radius0 = candidate
    height, width = gray.shape
    patch_margin = int(np.ceil(max(1.8 * radius0, radius0 + 15.0))) + 2
    x_min = max(0, int(center_x0 - patch_margin))
    x_max = min(width, int(center_x0 + patch_margin) + 1)
    y_min = max(0, int(center_y0 - patch_margin))
    y_max = min(height, int(center_y0 + patch_margin) + 1)
    if x_max <= x_min or y_max <= y_min:
        return None

    local_edges = edges[y_min:y_max, x_min:x_max]
    local_gray = gray[y_min:y_max, x_min:x_max]
    local_xs = np.arange(x_min, x_max, dtype=np.float64)[None, :]
    local_ys = np.arange(y_min, y_max, dtype=np.float64)[:, None]
    distances = np.hypot(local_xs - center_x0, local_ys - center_y0)
    annulus = (
        (distances >= max(2.0, 0.75 * radius0))
        & (distances <= 1.25 * radius0)
        & (local_edges > 0)
    )
    positions = np.argwhere(annulus)
    if positions.shape[0] < 30:
        return None

    # Edge pixels are returned as (row, column), but the circle fit uses (x, y).
    all_points = np.column_stack(
        (
            positions[:, 1].astype(np.float64) + x_min,
            positions[:, 0].astype(np.float64) + y_min,
        )
    )
    points = all_points.copy()

    circle = fit_circle(points)
    if circle is None:
        return None

    center_x, center_y, radius = circle
    for _ in range(3):
        radial_error = np.abs(
            np.hypot(points[:, 0] - center_x, points[:, 1] - center_y) - radius
        )
        inliers = radial_error <= max(3.0, 0.15 * radius)
        points = points[inliers]
        if points.shape[0] < 30:
            return None
        circle = fit_circle(points)
        if circle is None:
            return None
        center_x, center_y, radius = circle

    residual = float(
        np.mean(
            np.abs(
                np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
                - radius
            )
        )
        / radius
    )
    circle_angles = np.degrees(
        np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)
    ) % 360.0
    circle_bins = np.zeros(36, dtype=np.int32)
    for angle in circle_angles:
        circle_bins[int(angle // 10.0) % 36] += 1
    circle_support = float(np.count_nonzero(circle_bins)) / 36.0

    ellipse = fit_ellipse(all_points)
    if ellipse is None:
        return None
    center_x_e, center_y_e, major_radius, minor_radius, ellipse_angle = ellipse
    roundness = minor_radius / major_radius if major_radius > 0 else 0.0
    shape = "round" if roundness >= roundness_threshold else "elliptical"
    ellipse_support_value, ellipse_residual, _ = ellipse_support(
        all_points,
        center_x_e,
        center_y_e,
        major_radius,
        minor_radius,
        ellipse_angle,
    )
    inside_mask = distances < 0.6 * radius0
    outside_mask = (distances >= 1.3 * radius0) & (distances <= 1.8 * radius0)
    inside = local_gray[inside_mask]
    outside = local_gray[outside_mask]
    contrast = (
        abs(float(inside.mean()) - float(outside.mean()))
        if inside.size and outside.size
        else 0.0
    )
    inside_mean = float(inside.mean()) if inside.size else 0.0
    outside_mean = float(outside.mean()) if outside.size else 0.0
    inside_edge_density = (
        float(local_edges[inside_mask].mean())
        if np.count_nonzero(inside_mask)
        else 0.0
    )
    grad_x = cv2.Sobel(local_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(local_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.hypot(grad_x, grad_y)
    edge_ring = (distances >= 0.75 * radius0) & (distances <= 1.25 * radius0)
    edge_sharpness = (
        float(grad_magnitude[edge_ring].mean())
        if np.count_nonzero(edge_ring)
        else 0.0
    )

    return FittedCircle(
        x=center_x_e,
        y=center_y_e,
        radius=radius,
        major_radius=major_radius,
        minor_radius=minor_radius,
        ellipse_angle=ellipse_angle,
        roundness=roundness,
        shape=shape,
        residual=residual,
        ellipse_residual=ellipse_residual,
        support=circle_support,
        contrast=contrast,
        inside_mean=inside_mean,
        outside_mean=outside_mean,
        edge_sharpness=edge_sharpness,
        inside_edge_density=inside_edge_density,
        edge_points=int(points.shape[0]),
    )


def non_max_suppression(
    circles: list[FittedCircle], min_distance: float
) -> list[FittedCircle]:
    kept: list[FittedCircle] = []
    for circle in sorted(circles, key=lambda item: item.score, reverse=True):
        if all(
            np.hypot(circle.x - existing.x, circle.y - existing.y) >= min_distance
            for existing in kept
        ):
            kept.append(circle)
    return kept


def detect_holes_from_image(
    gray: np.ndarray,
    min_radius: int,
    max_radius: int,
    hough_threshold: int,
    min_support: float,
    max_residual: float,
    roundness_threshold: float,
    blur_size: int,
    canny_low: int,
    canny_high: int,
    use_contours: bool,
    reject_highlight: bool,
    highlight_threshold: float,
    use_highlight_sharpness: bool,
    highlight_sharpness: float,
    reject_texture: bool,
    max_inside_edge_density: float,
) -> list[FittedCircle]:
    blurred = cv2.medianBlur(gray, blur_size)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    candidates: list[tuple[float, float, float]] = []

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, min_radius * 2),
        param1=80,
        param2=hough_threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        candidates.extend(
            (float(x), float(y), float(r))
            for x, y, r in np.round(circles[0]).astype(np.float64)
        )

    if use_contours:
        closed_edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        contours, _ = cv2.findContours(
            closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        height, width = gray.shape
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue
            if area < 200 or area > 30000:
                continue
            if w < 15 or h < 15 or w > 220 or h > 220:
                continue
            if len(contour) < 5:
                continue
            try:
                (center_x, center_y), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
            except cv2.error:
                continue
            major_radius = max(axis_a, axis_b) / 2.0
            minor_radius = min(axis_a, axis_b) / 2.0
            if major_radius < min_radius or major_radius > max_radius:
                continue
            if major_radius < 8 or minor_radius < 8:
                continue
            ellipse_area = np.pi * major_radius * minor_radius
            fill_ratio = area / ellipse_area if ellipse_area > 0 else 0.0
            if fill_ratio < 0.45 or fill_ratio > 1.15:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter)
                if perimeter > 0
                else 0.0
            )
            if circularity < 0.20:
                continue
            candidates.append((center_x, center_y, major_radius))

    fitted: list[FittedCircle] = []
    for candidate in candidates:
        result = refine_circle(
            gray,
            edges,
            candidate,
            roundness_threshold,
        )
        if result is None:
            continue
        if result.support < min_support or result.residual > max_residual:
            continue
        if reject_highlight:
            bright_gap = result.inside_mean - result.outside_mean
            if (
                use_highlight_sharpness
                and highlight_sharpness > 0
                and result.edge_sharpness < highlight_sharpness
            ):
                bright_gap *= 1.5
            if bright_gap > highlight_threshold:
                continue
        if (
            reject_texture
            and result.inside_edge_density > max_inside_edge_density
        ):
            continue
        fitted.append(result)

    return non_max_suppression(fitted, min_distance=min_radius)


def detect_holes(
    image_path: Path,
    min_radius: int,
    max_radius: int,
    hough_threshold: int,
    min_support: float,
    max_residual: float,
    roundness_threshold: float,
    blur_size: int,
    canny_low: int,
    canny_high: int,
    use_contours: bool,
    reject_highlight: bool,
    highlight_threshold: float,
    use_highlight_sharpness: bool,
    highlight_sharpness: float,
    reject_texture: bool,
    max_inside_edge_density: float,
) -> list[FittedCircle]:
    gray = load_gray(image_path)
    return detect_holes_from_image(
        gray=gray,
        min_radius=min_radius,
        max_radius=max_radius,
        hough_threshold=hough_threshold,
        min_support=min_support,
        max_residual=max_residual,
        roundness_threshold=roundness_threshold,
        blur_size=blur_size,
        canny_low=canny_low,
        canny_high=canny_high,
        use_contours=use_contours,
        reject_highlight=reject_highlight,
        highlight_threshold=highlight_threshold,
        use_highlight_sharpness=use_highlight_sharpness,
        highlight_sharpness=highlight_sharpness,
        reject_texture=reject_texture,
        max_inside_edge_density=max_inside_edge_density,
    )


def write_outputs(
    image_path: Path,
    holes: list[FittedCircle],
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [circle.as_dict(index) for index, circle in enumerate(holes, start=1)]

    csv_path = output_dir / f"{prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / f"{prefix}.json"
    json_path.write_text(
        json.dumps({"image": str(image_path), "holes": rows}, indent=2),
        encoding="utf-8",
    )

    color = cv2.imread(str(image_path))
    if color is not None:
        for circle in holes:
            center = (int(round(circle.x)), int(round(circle.y)))
            if circle.shape == "round":
                cv2.circle(
                    color,
                    center,
                    int(round(circle.radius)),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                ellipse_axes = (
                    int(round(circle.major_radius)),
                    int(round(circle.minor_radius)),
                )
                cv2.ellipse(
                    color,
                    center,
                    ellipse_axes,
                    circle.ellipse_angle,
                    0,
                    360,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.circle(color, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
        overlay_path = output_dir / f"{prefix}_overlay.png"
        cv2.imwrite(str(overlay_path), color)
        print(f"Overlay saved: {overlay_path}")

    print(f"CSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit circular holes from an image and save results."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Input image path (default: the provided raw_image_1_ann1.png).",
    )
    parser.add_argument(
        "--image",
        dest="image_option",
        type=Path,
        help="Alternative to the positional image argument.",
    )
    parser.add_argument("--min-radius", type=int, default=10)
    parser.add_argument("--max-radius", type=int, default=60)
    parser.add_argument("--hough-threshold", type=int, default=40)
    parser.add_argument("--min-support", type=float, default=0.60)
    parser.add_argument("--max-residual", type=float, default=0.20)
    parser.add_argument("--roundness-threshold", type=float, default=0.90)
    parser.add_argument("--blur-size", type=int, default=5)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--contours", action="store_true")
    parser.add_argument("--reject-highlight", action="store_true")
    parser.add_argument("--highlight-threshold", type=float, default=40.0)
    parser.add_argument("--use-highlight-sharpness", action="store_true")
    parser.add_argument("--highlight-sharpness", type=float, default=0.0)
    parser.add_argument("--reject-texture", action="store_true")
    parser.add_argument("--max-inside-edge-density", type=float, default=8.0)
    parser.add_argument(
        "--roi",
        metavar="X,Y,W,H",
        help="Only keep hole centers inside the given pixel ROI.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", default="circle_holes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = (args.image_option or args.image or DEFAULT_IMAGE).expanduser()
    holes = detect_holes(
        image_path=image_path,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        hough_threshold=args.hough_threshold,
        min_support=args.min_support,
        max_residual=args.max_residual,
        roundness_threshold=args.roundness_threshold,
        blur_size=args.blur_size,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        use_contours=args.contours,
        reject_highlight=args.reject_highlight,
        highlight_threshold=args.highlight_threshold,
        use_highlight_sharpness=args.use_highlight_sharpness,
        highlight_sharpness=args.highlight_sharpness,
        reject_texture=args.reject_texture,
        max_inside_edge_density=args.max_inside_edge_density,
    )

    holes.sort(key=lambda item: (round(item.x / 200.0), round(item.y / 200.0), item.y))
    if args.roi:
        roi_x, roi_y, roi_w, roi_h = (
            float(value) for value in args.roi.replace(",", " ").split()
        )
        holes = [
            hole
            for hole in holes
            if roi_x <= hole.x <= roi_x + roi_w
            and roi_y <= hole.y <= roi_y + roi_h
        ]
    print(f"Detected {len(holes)} holes in {image_path}")
    print("index  x       y       radius  major  minor  round  shape  residual  support")
    for index, circle in enumerate(holes, start=1):
        print(
            f"{index:5d}  {circle.x:7.2f}  {circle.y:7.2f}  "
            f"{circle.radius:6.2f}  {circle.major_radius:6.2f}  {circle.minor_radius:6.2f}  "
            f"{circle.roundness:5.3f}  {circle.shape:9s}  "
            f"{circle.residual:7.3f}  {circle.support:6.2f}"
        )

    output_dir = args.output_dir or image_path.parent
    write_outputs(image_path, holes, output_dir, args.prefix)


if __name__ == "__main__":
    main()
