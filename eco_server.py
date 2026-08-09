from __future__ import annotations

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from eco_engine import EcoConfig, Ecosystem, NumpyEncoder


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "eco_game.html"

ecosystem: Optional[Ecosystem] = None
lock = threading.Lock()


def _request_json(body: bytes) -> Dict[str, Any]:
    if not body:
        return {}
    try:
        data = json.loads(body.decode("utf-8"))
        if isinstance(data, dict):
            return data
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {}


def _query_digit(query: Dict[str, list[str]]) -> Optional[int]:
    raw = query.get("digit", [None])[0]
    if raw is None:
        return None
    try:
        digit = int(raw)
    except (TypeError, ValueError):
        return None
    if digit < 0:
        return None
    return digit % 10


class EcoHandler(BaseHTTPRequestHandler):
    server_version = "EcoGame/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            cls=NumpyEncoder,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self) -> None:
        if not HTML_PATH.exists():
            self._send_json({"error": "eco_game.html not found"}, 404)
            return
        data = HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_digit_image(self) -> None:
        assert ecosystem is not None
        with lock:
            food = ecosystem.current_food
        try:
            from PIL import Image

            image = (
                np.asarray(food["image"], dtype=np.float32).reshape(28, 28) * 255.0
            ).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image, mode="L").resize(
                (280, 280), Image.NEAREST
            )
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            data = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except ImportError:
            self._send_json(food)

    def _state(self) -> Dict[str, Any]:
        assert ecosystem is not None
        with lock:
            return ecosystem.state()

    def _step(self, digit: Optional[int]) -> Dict[str, Any]:
        assert ecosystem is not None
        with lock:
            return ecosystem.step(digit)

    def _update_config(self, body: Dict[str, Any]) -> Dict[str, Any]:
        assert ecosystem is not None
        with lock:
            reset_requested = bool(body.get("reset") or "init_pop" in body)
            ecosystem.config.update(body)
            if reset_requested:
                ecosystem.reset()
            return ecosystem.state()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html", "/eco_game.html"):
            self._send_html()
            return
        if path == "/api/state":
            self._send_json(self._state())
            return
        if path == "/api/config":
            self._send_json(self._state()["config"])
            return
        if path == "/api/digit_image":
            self._send_digit_image()
            return
        if path in ("/api/step", "/api/manual_feed"):
            self._send_json(self._step(_query_digit(query)))
            return
        if path == "/api/reset":
            self._send_json(self._update_config({"reset": True}))
            return
        self._send_json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body = _request_json(self.rfile.read(length))

        if path in ("/api/step", "/api/manual_feed"):
            self._send_json(self._step(_query_digit({"digit": [str(body.get("digit"))]})))
            return
        if path == "/api/config":
            self._send_json(self._update_config(body))
            return
        if path == "/api/reset":
            self._send_json(self._update_config({"reset": True}))
            return
        self._send_json({"error": "not found", "path": path}, 404)


def main() -> None:
    global ecosystem
    parser = argparse.ArgumentParser(description="LIF ecology game server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--survival-rounds", type=int, default=None)
    args = parser.parse_args()

    config = EcoConfig()
    if args.population is not None:
        config.init_pop = max(60, min(1000, args.population))
    if args.capacity is not None:
        config.capacity = max(100, min(10000, args.capacity))
    if args.survival_rounds is not None:
        config.survival_rounds = max(10, min(30, args.survival_rounds))

    ecosystem = Ecosystem(config=config, seed=args.seed)
    server = ThreadingHTTPServer((args.host, args.port), EcoHandler)
    server.daemon_threads = True
    print(f"LIF ecology server: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
