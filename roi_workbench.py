#!/usr/bin/env python3
"""Local web app: image annotation -> ROI extraction -> ROI processing console.

Serves a single-page HTML app with two views:
  1. Annotation workbench: upload an image, draw rectangle ROIs, extract them.
  2. ROI processing console: multi-select ROIs, pick an algorithm
     (Wiener / blind deconvolution / first+second derivative extremum /
     Zernike moment subpixel edge), run batch processing, view and export.

Backend mirrors fit_debug_server.py: stdlib http.server + OpenCV/numpy.
"""

from __future__ import annotations

import argparse
import base64
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

import roi_algorithms as algorithms  # noqa: E402

HTML_PATH = BASE_DIR / "roi_workbench.html"
IMAGE_CACHE: dict[str, dict[str, Any]] = {}
MAX_CACHE = 8
MAX_IMAGE_PIXELS = 60_000_000  # safety cap (~60MP) to avoid OOM

MIN_ROI_SIZE = 5


def _json_body(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _decode_image(data_url: str) -> np.ndarray | None:
    if "," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _encode_png(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _data_url(encoded: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{encoded}"


def _cache_image(color: np.ndarray) -> str:
    image_id = uuid.uuid4().hex
    IMAGE_CACHE[image_id] = {"color": color, "created": time.time()}
    while len(IMAGE_CACHE) > MAX_CACHE:
        oldest = min(IMAGE_CACHE, key=lambda key: IMAGE_CACHE[key]["created"])
        IMAGE_CACHE.pop(oldest, None)
    return image_id


def _crop_roi(color: np.ndarray, roi: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    """Clamp an ROI to the image and crop. Returns (crop, actual_roi)."""
    h_img, w_img = color.shape[:2]
    x = max(0, min(w_img, int(round(float(roi.get("x", 0))))))
    y = max(0, min(h_img, int(round(float(roi.get("y", 0))))))
    w = max(0, int(round(float(roi.get("w", 0)))))
    h = max(0, int(round(float(roi.get("h", 0)))))
    x2 = max(x, min(w_img, x + w))
    y2 = max(y, min(h_img, y + h))
    actual = {"x": float(x), "y": float(y), "w": float(x2 - x), "h": float(y2 - y)}
    crop = color[y:y2, x:x2]
    return crop, actual


def _process_one(color: np.ndarray, roi: dict[str, Any], algorithm: str, params: dict[str, Any]) -> dict[str, Any]:
    roi_id = str(roi.get("id") or roi.get("roi_id") or "")
    crop, actual = _crop_roi(color, roi)
    base = {"roi_id": roi_id, **{k: v for k, v in actual.items()}, "error": None}
    if crop.size == 0 or min(crop.shape[:2]) < MIN_ROI_SIZE:
        base["error"] = "ROI 越界或太小（至少 5x5 像素）"
        return base
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        result = algorithms.process_roi(gray, algorithm, params)
    except Exception as exc:  # includes torch ImportError / missing weight / OOM
        base["error"] = str(exc)
        return base
    original = _data_url(_encode_png(crop))
    processed = _data_url(_encode_png(result["processed"]))
    overlay = _data_url(_encode_png(result["overlay"]))
    base.update(
        {
            "original": original,
            "processed": processed,
            "overlay": overlay,
            "metrics": result["metrics"],
            "psf_estimated": result["psf_estimated"],
        }
    )
    return base


def _perspective_warp(color: np.ndarray, points: list, output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a plane: map a user-selected quadrilateral to a rectangle (top-down view).

    Returns (warped_color, H, src_points, dst_points).
    """
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("需要 4 个角点（左上/右上/右下/左下）")
    h_img, w_img = color.shape[:2]
    src = np.zeros((4, 2), np.float32)
    for i, pt in enumerate(points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"第 {i + 1} 个角点格式错误")
        x = float(pt[0])
        y = float(pt[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"第 {i + 1} 个角点不是有效坐标")
        src[i] = (float(np.clip(x, 0, w_img - 1)), float(np.clip(y, 0, h_img - 1)))

    def dist(a: int, b: int) -> float:
        return float(np.hypot(src[a, 0] - src[b, 0], src[a, 1] - src[b, 1]))

    mode = output.get("mode", "auto")
    if mode == "custom":
        w = max(8, int(output.get("width") or 0))
        h = max(8, int(output.get("height") or 0))
    elif mode == "square":
        # 正方形基准（4 个点在实际物理世界中构成正方形）：输出正方形，
        # 平面上的正圆矫正后仍是正圆。
        side = int(round(max(dist(0, 1), dist(1, 2), dist(2, 3), dist(3, 0))))
        w = h = max(8, min(side, 12000))
    else:
        # auto：默认按投影边长（上/下边取较长者、左/右边取较长者），只去透视、不额外拉伸。
        # 若用户提供物理长宽比 ratio（宽/高），则用它定纵横比，可精确还原正圆。
        w = int(round(max(dist(0, 1), dist(2, 3))))
        h = int(round(max(dist(1, 2), dist(3, 0))))
        ratio = float(output.get("ratio") or 0)
        if ratio > 0.01:
            h = max(8, int(round(w / ratio)))
        w = max(8, min(w, 12000))
        h = max(8, min(h, 12000))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}.get(
        str(output.get("interpolation", "linear")), cv2.INTER_LINEAR)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(color, matrix, (w, h), flags=interp,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, matrix, src, dst


class RoiWorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "RoiWorkbench/1.0"

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
        if parsed.path in ("/", "/index.html", "/roi_workbench.html"):
            if not HTML_PATH.exists():
                self._send_json({"error": "roi_workbench.html not found"}, 404)
                return
            self._send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/image":
            self._serve_cached_image(parse_qs(parsed.query))
            return
        if parsed.path == "/api/dl_status":
            try:
                import roi_dl
                self._send_json({"ok": True, "status": roi_dl.dl_status()})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)})
            return
        self._send_json({"error": "not found"}, 404)

    def _serve_cached_image(self, query: dict[str, list[str]]) -> None:
        image_id = query.get("image_id", [None])[0]
        item = IMAGE_CACHE.get(image_id or "")
        if item is None:
            self._send_json({"error": "image cache expired"}, 404)
            return
        ok, encoded = cv2.imencode(".png", item["color"])
        if not ok:
            self._send_json({"error": "image encoding failed"}, 500)
            return
        self._send_bytes(encoded.tobytes(), "image/png")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = _json_body(self.rfile.read(length))
        try:
            if parsed.path == "/api/upload":
                self._upload(payload)
                return
            if parsed.path == "/api/process":
                self._process(payload)
                return
            if parsed.path == "/api/perspective":
                self._perspective(payload)
                return
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        self._send_json({"error": "not found"}, 404)

    def _upload(self, payload: dict[str, Any]) -> None:
        image = _decode_image(payload.get("image", ""))
        if image is None or image.size == 0:
            self._send_json({"error": "无法解码上传图片"}, 400)
            return
        if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
            self._send_json({"error": "图片过大（超过 60MP）"}, 400)
            return
        image_id = _cache_image(image)
        self._send_json(
            {
                "image_id": image_id,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }
        )

    def _process(self, payload: dict[str, Any]) -> None:
        image_id = payload.get("image_id", "")
        item = IMAGE_CACHE.get(image_id or "")
        if item is None:
            self._send_json({"error": "图片缓存已过期，请重新上传"}, 400)
            return
        algorithm = payload.get("algorithm", "")
        if algorithm not in algorithms.ALGORITHMS:
            self._send_json({"error": f"未知算法: {algorithm}"}, 400)
            return
        rois = payload.get("rois")
        if not isinstance(rois, list) or not rois:
            self._send_json({"error": "未选择任何 ROI"}, 400)
            return
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            self._send_json({"error": "params 必须是对象"}, 400)
            return
        started = time.perf_counter()
        results = [_process_one(item["color"], roi, algorithm, params) for roi in rois]
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        self._send_json({"results": results, "elapsed_ms": elapsed_ms, "algorithm": algorithm})


    def _perspective(self, payload: dict[str, Any]) -> None:
        image_id = payload.get("image_id", "")
        item = IMAGE_CACHE.get(image_id or "")
        if item is None:
            self._send_json({"error": "图片缓存已过期，请重新上传"}, 400)
            return
        try:
            warped, matrix, src_pts, dst_pts = _perspective_warp(
                item["color"], payload.get("points") or [], payload.get("output") or {})
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        new_id = _cache_image(warped)
        self._send_json({
            "image_id": new_id,
            "width": int(warped.shape[1]),
            "height": int(warped.shape[0]),
            "matrix": matrix.round(6).tolist(),
            "src_points": src_pts.round(2).tolist(),
            "dst_points": dst_pts.round(2).tolist(),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI annotation + processing workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RoiWorkbenchHandler)
    server.daemon_threads = True
    print(f"ROI workbench: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


