#!/usr/bin/env python3
"""Inference worker for llm_chat_server.

Reads one JSON request from stdin, streams JSON lines to stdout:
  {"ready": true}            -> model loaded
  {"delta": "<text>"}        -> streamed token
  {"done": true}             -> generation finished
  {"error": "<message>"}     -> error (then exit)

Images in messages are force-downscaled to <= MAX_EDGE px before inference,
so huge uploads can never stall the vision encoder.
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MTMDChatHandler

MAX_EDGE = 1024  # 图片最长边（服务端强制）


def _force_utf8() -> None:
    """Windows 管道默认用 GBK 写 stdout，必须强制 UTF-8，否则中文乱码。"""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
JPEG_QUALITY = 88


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _downscale_data_url(data_url: str) -> str:
    if not data_url.startswith("data:"):
        return data_url
    _, _, b64 = data_url.partition(",")
    raw = base64.b64decode(b64)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("无法解码上传的图片")
    h, w = arr.shape[:2]
    long_edge = max(h, w)
    if long_edge > MAX_EDGE:
        scale = MAX_EDGE / long_edge
        arr = cv2.resize(
            arr,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise ValueError("图片压缩失败")
    return f"data:image/jpeg;base64,{base64.b64encode(enc.tobytes()).decode()}"


def _process_messages(messages: list) -> None:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    iu = part["image_url"]
                    url = iu["url"] if isinstance(iu, dict) else iu
                    if isinstance(url, str) and url.startswith("data:"):
                        new_url = _downscale_data_url(url)
                        if isinstance(iu, dict):
                            iu["url"] = new_url
                        else:
                            part["image_url"] = new_url


def main() -> None:
    _force_utf8()
    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    mmproj_path = sys.argv[2] if len(sys.argv) > 2 and Path(sys.argv[2]).exists() else None
    n_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    try:
        kwargs: dict = {"model_path": model_path, "n_ctx": n_ctx, "n_gpu_layers": 0, "verbose": False}
        if n_threads and n_threads > 0:
            kwargs["n_threads"] = n_threads
        if mmproj_path:
            kwargs["chat_handler"] = MTMDChatHandler(
                clip_model_path=mmproj_path, use_gpu=False, verbose=False
            )
        llm = Llama(**kwargs)
        _emit({"ready": True})
    except Exception:
        _emit({"error": "模型加载失败: " + traceback.format_exc(limit=1)})
        sys.exit(1)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            messages = list(req.get("messages") or [])
            system_prompt = (req.get("system_prompt") or "").strip()
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            _process_messages(messages)

            params: dict = {
                "messages": messages,
                "max_tokens": max(1, min(int(req.get("max_tokens", 512)), 8192)),
                "temperature": float(req.get("temperature", 0.7)),
                "top_p": float(req.get("top_p", 0.95)),
                "stream": True,
            }
            for chunk in llm.create_chat_completion(**params):
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    _emit({"delta": text})
            _emit({"done": True})
        except Exception:
            _emit({"error": "推理出错: " + traceback.format_exc(limit=1)})


if __name__ == "__main__":
    main()

