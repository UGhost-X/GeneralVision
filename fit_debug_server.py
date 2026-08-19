#!/usr/bin/env python3
"""Local web app for debugging circle/ellipse hole fitting."""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fit_circle_holes as fitting  # noqa: E402
import shape_detector as shape_detection  # noqa: E402


HTML_PATH = BASE_DIR / "fit_debug.html"
SAMPLE_DIR = Path(r"C:\Users\Administrator\Desktop\新建文件夹\one2all")
SAMPLES = {
    "test-1": SAMPLE_DIR / "test-1.png",
    "multi-demo-4": SAMPLE_DIR / "multi-demo-4-h-w.png",
    "demo-2": SAMPLE_DIR / "demo-2.png",
    "raw_image_1_ann1": SAMPLE_DIR / "raw_image_1_ann1-bak.png",
}
IMAGE_CACHE: dict[str, dict[str, Any]] = {}


def _json_body(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_sample(name: str) -> np.ndarray | None:
    path = SAMPLES.get(name)
    if path is None:
        candidate = Path(name)
        if candidate.exists():
            path = candidate
    if path is None or not path.exists():
        return None
    return fitting.read_image(path, cv2.IMREAD_COLOR)


def _decode_image(data_url: str) -> np.ndarray | None:
    if "," not in data_url:
        return None
    payload = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return image


def _encode_png(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _data_url(encoded: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{encoded}"


def _cache_image(color: np.ndarray, overlay: np.ndarray) -> str:
    image_id = uuid.uuid4().hex
    IMAGE_CACHE[image_id] = {
        "color": color,
        "overlay": overlay,
        "created": time.time(),
    }
    while len(IMAGE_CACHE) > 8:
        oldest = min(IMAGE_CACHE, key=lambda key: IMAGE_CACHE[key]["created"])
        IMAGE_CACHE.pop(oldest, None)
    return image_id


def _expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    options: list[tuple[str, list[Any]]] = []
    for key, spec in grid.items():
        if not isinstance(spec, dict):
            continue
        low = spec.get("min", spec.get("start", 0))
        high = spec.get("max", spec.get("end", low))
        step = spec.get("step", 1)
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            count = int(round((high - low) / step)) + 1
            values = [low + step * index for index in range(max(1, count))]
        else:
            values = [low, high]
        options.append((key, values))
    keys = [key for key, _ in options]
    value_lists = [values for _, values in options]
    results: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        results.append(dict(zip(keys, combo)))
    return results


def _params_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params", {})
    defaults = {
        "target_type": "hole",
        "min_radius": 10,
        "max_radius": 60,
        "min_area": 100.0,
        "max_area": 100000000.0,
        "min_side": 5.0,
        "max_side": 10000.0,
        "min_fill_ratio": 0.2,
        "fill_mode": "auto",
        "hough_threshold": 40,
        "min_support": 0.60,
        "max_residual": 0.20,
        "roundness_threshold": 0.90,
        "blur_size": 5,
        "canny_low": 50,
        "canny_high": 150,
        "use_contours": True,
        "reject_highlight": False,
        "highlight_threshold": 40.0,
        "use_highlight_sharpness": False,
        "highlight_sharpness": 0.0,
        "reject_texture": True,
        "max_inside_edge_density": 8.0,
    }
    result = dict(defaults)
    if isinstance(params, dict):
        target_type = params.get("target_type", "hole")
        if target_type == "nut":
            result["reject_texture"] = False
            result["reject_highlight"] = False
        elif target_type == "generic":
            result["reject_texture"] = False
            result["reject_highlight"] = False
        elif target_type == "hole":
            result["reject_texture"] = True
            result["reject_highlight"] = False
        for key in defaults:
            if key in params and params[key] is not None:
                result[key] = params[key]
    result["blur_size"] = max(1, int(result["blur_size"]) | 1)
    result["use_contours"] = bool(result["use_contours"])
    return result


def _filter_roi(holes: list[fitting.FittedCircle], roi: list[float] | None):
    if not roi or len(roi) != 4:
        return holes
    x, y, w, h = (float(v) for v in roi)
    return [
        hole
        for hole in holes
        if x <= hole.x <= x + w and y <= hole.y <= y + h
    ]


def _evaluate(
    expected: list[dict[str, Any]],
    holes: list[fitting.FittedCircle],
    match_tolerance: float,
) -> dict[str, Any]:
    if not expected:
        return {
            "has_truth": False,
            "true_positives": None,
            "false_positives": None,
            "false_negatives": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_center_error_px": None,
            "missed": [],
            "false_detections": [],
        }

    detections = list(holes)
    used = [False] * len(detections)
    matches: list[tuple[int, int, float]] = []
    for expected_index, item in enumerate(expected):
        exp_x = float(item.get("x", 0))
        exp_y = float(item.get("y", 0))
        exp_r = float(item.get("radius", match_tolerance))
        tolerance = max(match_tolerance, exp_r * 0.25)
        best_index = -1
        best_distance = tolerance
        for detection_index, hole in enumerate(detections):
            if used[detection_index]:
                continue
            distance = float(np.hypot(hole.x - exp_x, hole.y - exp_y))
            if distance <= best_distance:
                best_distance = distance
                best_index = detection_index
        if best_index >= 0:
            used[best_index] = True
            matches.append((expected_index, best_index, best_distance))

    true_positive = len(matches)
    false_positive = len(detections) - true_positive
    false_negative = len(expected) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    average_error = (
        float(np.mean([distance for _, _, distance in matches])) if matches else None
    )

    matched_expected = {expected_index for expected_index, _, _ in matches}
    matched_detection = {detection_index for _, detection_index, _ in matches}
    missed = [
        {
            "x": float(expected[index].get("x", 0)),
            "y": float(expected[index].get("y", 0)),
            "radius": float(expected[index].get("radius", 0) or 0),
        }
        for index in range(len(expected))
        if index not in matched_expected
    ]
    false_detections = [
        {
            "x": float(hole.x),
            "y": float(hole.y),
            "radius": float(hole.radius),
            "shape": hole.shape,
        }
        for index, hole in enumerate(detections)
        if index not in matched_detection
    ]
    return {
        "has_truth": True,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_center_error_px": (
            round(average_error, 3) if average_error is not None else None
        ),
        "missed": missed,
        "false_detections": false_detections,
    }


def _draw_overlay(
    color: np.ndarray,
    holes: list[Any],
    expected: list[dict[str, Any]],
    roi: list[float] | None,
) -> np.ndarray:
    overlay = color.copy()
    for index, hole in enumerate(holes, start=1):
        center = (int(round(hole.x)), int(round(hole.y)))
        polygon_points = getattr(hole, "points", None)
        if polygon_points:
            polygon = np.asarray(polygon_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                overlay,
                [polygon],
                True,
                (255, 128, 0),
                2,
                cv2.LINE_AA,
            )
        elif hole.shape == "round":
            cv2.circle(
                overlay,
                center,
                int(round(hole.radius)),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            axes = (
                int(round(hole.major_radius)),
                int(round(hole.minor_radius)),
            )
            cv2.ellipse(
                overlay,
                center,
                axes,
                hole.ellipse_angle,
                0,
                360,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.circle(overlay, center, 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for item in expected:
        x = int(round(float(item.get("x", 0))))
        y = int(round(float(item.get("y", 0))))
        r = int(round(float(item.get("radius", 20) or 20)))
        cv2.drawMarker(
            overlay,
            (x, y),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(overlay, (x, y), max(5, r), (0, 255, 255), 1, cv2.LINE_AA)

    if roi and len(roi) == 4:
        x, y, w, h = (int(round(float(v))) for v in roi)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 128, 0), 2)
    return overlay


def _serialize_hole(index: int, hole: fitting.FittedCircle) -> dict[str, Any]:
    return hole.as_dict(index)


def _run_fit(
    payload: dict[str, Any],
    params: dict[str, Any] | None = None,
    include_assets: bool = True,
) -> dict[str, Any]:
    color = None
    if payload.get("image"):
        color = _decode_image(str(payload["image"]))
    elif payload.get("sample"):
        color = _read_sample(str(payload["sample"]))
    if color is None:
        raise ValueError("image or sample is required")

    gray = (
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if color.ndim == 3
        else color
    )
    final_params = _params_from_payload(payload) if params is None else params
    target_type = final_params.get("target_type", "hole")
    polygon_types = ("rectangle", "chamfered_rectangle", "trapezoid")
    if target_type in polygon_types:
        holes = shape_detection.detect_shapes(
            gray=gray,
            shape_type=target_type,
            blur_size=int(final_params["blur_size"]),
            canny_low=int(final_params["canny_low"]),
            canny_high=int(final_params["canny_high"]),
            min_area=float(final_params["min_area"]),
            max_area=float(final_params["max_area"]),
            min_side=float(final_params["min_side"]),
            max_side=float(final_params["max_side"]),
            min_fill_ratio=float(final_params["min_fill_ratio"]),
            fill_mode=str(final_params["fill_mode"]),
        )
    else:
        detection_params = {
            key: value
            for key, value in final_params.items()
            if key
            not in (
                "target_type",
                "min_area",
                "max_area",
                "min_side",
                "max_side",
                "min_fill_ratio",
                "fill_mode",
            )
        }
        holes = fitting.detect_holes_from_image(gray=gray, **detection_params)
    roi = payload.get("roi")
    holes = _filter_roi(holes, roi)

    expected = payload.get("expected", [])
    if not isinstance(expected, list):
        expected = []
    match_tolerance = float(payload.get("match_tolerance", 20.0))
    metrics = _evaluate(expected, holes, match_tolerance)

    result: dict[str, Any] = {
        "width": int(color.shape[1]),
        "height": int(color.shape[0]),
        "results": [
            _serialize_hole(index, hole) for index, hole in enumerate(holes, start=1)
        ],
        "metrics": metrics,
        "params": final_params,
        "elapsed_ms": 0,
    }
    if include_assets:
        overlay = _draw_overlay(color, holes, expected, roi)
        result["image_id"] = _cache_image(color, overlay)
    return result


class FitDebugHandler(BaseHTTPRequestHandler):
    server_version = "FitDebug/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/fit_debug.html"):
            if not HTML_PATH.exists():
                self._send_json({"error": "fit_debug.html not found"}, 404)
                return
            self._send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/samples":
            self._send_json(
                {
                    "samples": [
                        {"name": name, "path": str(path), "exists": path.exists()}
                        for name, path in SAMPLES.items()
                    ]
                }
            )
            return
        if parsed.path == "/api/image":
            self._serve_cached_image(parse_qs(parsed.query))
            return
        self._send_json({"error": "not found"}, 404)

    def _serve_cached_image(self, query: dict[str, list[str]]) -> None:
        image_id = query.get("image_id", [None])[0]
        mode = query.get("mode", ["overlay"])[0]
        item = IMAGE_CACHE.get(image_id or "")
        if item is None:
            self._send_json({"error": "image cache expired"}, 404)
            return
        image = item.get("color") if mode == "original" else item.get("overlay")
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            self._send_json({"error": "image encoding failed"}, 500)
            return
        self._send_bytes(encoded.tobytes(), "image/png")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = _json_body(self.rfile.read(length))
        try:
            if parsed.path == "/api/fit":
                started = time.perf_counter()
                result = _run_fit(payload)
                result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
                self._send_json(result)
                return
            if parsed.path == "/api/grid":
                self._send_json(self._grid(payload))
                return
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        self._send_json({"error": "not found"}, 404)

    def _grid(self, payload: dict[str, Any]) -> dict[str, Any]:
        grid_spec = payload.get("grid", {})
        if not isinstance(grid_spec, dict):
            raise ValueError("grid is required")
        combinations = _expand_grid(grid_spec)
        max_combinations = int(payload.get("max_combinations", 24))
        if len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]

        base_params = _params_from_payload(payload)
        runs: list[dict[str, Any]] = []
        for combo in combinations:
            params = dict(base_params)
            params.update(combo)
            try:
                started = time.perf_counter()
                result = _run_fit(payload, params, include_assets=False)
                metrics = result["metrics"]
                runs.append(
                    {
                        "params": params,
                        "count": len(result["results"]),
                        "metrics": metrics,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                )
            except Exception:
                continue

        def sort_key(run: dict[str, Any]) -> tuple:
            metrics = run["metrics"]
            f1 = metrics.get("f1")
            if f1 is None:
                return (-1.0, -float(run["count"]))
            return (-float(f1), -float(metrics.get("recall", 0.0)))

        runs.sort(key=sort_key)
        return {
            "total_combinations": len(combinations),
            "runs": runs[:50],
            "has_truth": bool(payload.get("expected")),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Circle/ellipse fit debug app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), FitDebugHandler)
    server.daemon_threads = True
    print(f"Fit debug app: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
