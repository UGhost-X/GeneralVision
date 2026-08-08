# eco_server.py
"""LIF 生态游戏本地服务：托管前端 + 推演/状态 API（纯 stdlib http.server）。"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import eco_engine as eco

PORT_DEFAULT = 8765
ROOT = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(ROOT, "eco_game.html")


class EcoHandler(BaseHTTPRequestHandler):
    engine: eco.Ecosystem  # 类属性由 make_server 注入
    lock = threading.Lock()

    # ---- helpers ----
    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "请求体不是合法 JSON"}, 400)
            return None

    def do_GET(self):
        p = self.path
        if p in ("/", "/index.html"):
            try:
                with open(HTML, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                self._json({"error": "eco_game.html 不存在"}, 500); return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif p == "/api/state":
            with self.lock:
                self._json(self.engine.get_state())
        elif p.startswith("/api/digit_image/"):
            try:
                idx = int(p.rsplit("/", 1)[1])
            except (ValueError, IndexError):
                self._json({"error": "无效的数字图像索引"}, 400); return
            try:
                with self.lock:
                    self._json(self.engine.get_digit_image(idx))
            except IndexError:
                self._json({"error": "数字图像索引越界"}, 400)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/step":
            with self.lock:
                events, stats = self.engine.step_round()
            self._json({"round": stats["round"], "events": events, "stats": stats})
        elif self.path == "/api/manual_feed":
            body = self._read_body()
            if body is None:
                return
            try:
                digit = int(body.get("digit", 0))
            except (ValueError, TypeError):
                self._json({"error": "digit 必须是 0-9 的整数"}, 400); return
            if not (0 <= digit <= 9):
                self._json({"error": f"digit {digit} 超出 0-9 范围"}, 400); return
            with self.lock:
                st = self.engine.get_state()
                name = body.get("name") or st["stats"]["best_name"]
                try:
                    self._json(self.engine.manual_feed(name, digit))
                except StopIteration:
                    self._json({"error": f"no organism named {name}"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):  # 静音访问日志，避免刷屏
        pass


def _build_server(seed: int = 0, port: int = 0):
    engine = eco.Ecosystem(seed=seed)
    handler = type("Handler", (EcoHandler,), {"engine": engine})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def run_server_in_thread(seed: int = 0, port: int = 0):
    """测试用：随机空闲端口，返回 (port, server)。"""
    server = _build_server(seed=seed, port=port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server.server_address[1], server


def main():
    ap = argparse.ArgumentParser(description="LIF 生态游戏服务")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    server = _build_server(seed=args.seed, port=args.port)
    print(f"LIF 生态游戏已启动: http://127.0.0.1:{args.port}  (seed={args.seed})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
