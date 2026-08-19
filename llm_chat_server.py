#!/usr/bin/env python3
"""Local web chat app for the Qwen3.5 multimodal GGUF model (text + image).

Inference runs in a separate worker process (llm_infer_worker.py), spawned per
request. A hung/crashed worker only kills itself; the web server watches it with
a timeout and reports errors instead of hanging forever.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import subprocess
import sys
import threading

import cv2
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "Qwen3.5-0.8B-Q4_K_M.gguf"
DEFAULT_MMPROJ = BASE_DIR / "mmproj-F16.gguf"
WORKER = BASE_DIR / "llm_infer_worker.py"
HTML_PATH = BASE_DIR / "llm_chat.html"

SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import roi_library  # noqa: E402

NO_DATA_TIMEOUT = 180.0  # 秒：worker 超过该时长无输出则杀掉并报错

args: argparse.Namespace
_busy = False
_busy_lock = threading.Lock()

_roi_lib: roi_library.RoiLibrary | None = None
_roi_lib_mtime: float | None = None
_roi_lock = threading.Lock()


def _library_signature(d: Path):
    """返回 (文件名, mtime) 排序元组，增删改任一文件都会变化。"""
    try:
        return tuple(sorted((f.name, f.stat().st_mtime) for f in d.iterdir() if f.is_file()))
    except OSError:
        return ()


def _get_roi_library() -> roi_library.RoiLibrary | None:
    """懒加载 ROI 库；目录内容变化（增/删/改）时自动重建。"""
    global _roi_lib, _roi_lib_mtime
    d = Path(args.roi_library)
    if not d.is_dir():
        return None
    with _roi_lock:
        sig = _library_signature(d)
        if _roi_lib is None or _roi_lib_mtime != sig:
            _roi_lib = roi_library.RoiLibrary(d)
            _roi_lib_mtime = sig
    return _roi_lib


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen3.5 multimodal chat web app")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Main GGUF model file")
    p.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ, help="Vision mmproj GGUF file (optional)")
    p.add_argument("--n-ctx", type=int, default=8192, help="Context window size")
    p.add_argument("--roi-library", type=Path, default=BASE_DIR / "data" / "roi_library", help="ROI 库目录")
    p.add_argument("--n-threads", type=int, default=None, help="CPU threads (default: auto)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not args.model.exists():
        raise FileNotFoundError(f"model file not found: {args.model}")
    print(f"[info] model: {args.model}")
    if args.mmproj and args.mmproj.exists():
        print(f"[info] vision: {args.mmproj}")
    print("[info] server ready (inference in worker process)")
    yield


app = FastAPI(title="Qwen3.5 Chat", lifespan=lifespan)


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _stream_worker(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """调用推理子进程，产出 {'delta'|'done'|'error'} 字典流。"""
    global _busy
    with _busy_lock:
        if _busy:
            yield {"error": "上一个请求仍在处理中，请等待完成后再试"}
            return
        _busy = True

    proc: subprocess.Popen | None = None
    timed_out = False
    try:
        mmproj_arg = str(args.mmproj) if args.mmproj and args.mmproj.exists() else ""
        cmd = [
            sys.executable,
            str(WORKER),
            str(args.model),
            mmproj_arg,
            str(args.n_ctx),
            str(args.n_threads or 0),
        ]
        env = dict(__import__('os').environ)
        env['PYTHONUTF8'] = '1'  # 保证 worker 子进程 UTF-8 输出，避免中文乱码
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(BASE_DIR),
            env=env,
        )
        proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]

        out_q: queue.Queue[bytes | None] = queue.Queue()

        def _reader() -> None:
            try:
                for raw in proc.stdout:  # type: ignore[union-attr]
                    out_q.put(raw)
            finally:
                out_q.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        got_done = False
        got_error = False
        while True:
            try:
                raw = out_q.get(timeout=NO_DATA_TIMEOUT)
            except queue.Empty:
                timed_out = True
                yield {"error": f"处理超时（{int(NO_DATA_TIMEOUT)} 秒无输出），已终止本次请求，请重试"}
                proc.kill()
                break
            if raw is None:
                break  # worker 输出结束
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("ready"):
                continue
            if obj.get("error"):
                got_error = True
                yield {"error": obj["error"]}
                continue
            if obj.get("done"):
                got_done = True
                yield {"done": True}
                break
            if obj.get("delta"):
                yield {"delta": obj["delta"]}

        if not got_done and not got_error and not timed_out:
            rc = proc.poll()
            yield {"error": f"生成进程意外退出（code={rc}），请重试"}
    except Exception as exc:  # noqa: BLE001
        yield {"error": str(exc)}
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        with _busy_lock:
            _busy = False


def _sse_stream(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for ev in events:
        yield _sse(ev)


@app.post("/api/chat")
def chat(payload: dict[str, Any]) -> StreamingResponse:
    return StreamingResponse(_sse_stream(_stream_worker(payload)), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/roi/library")
def roi_library_list() -> JSONResponse:
    lib = _get_roi_library()
    if lib is None:
        return JSONResponse({"ok": False, "error": f"ROI 库目录不存在: {args.roi_library}"})
    return JSONResponse({"ok": True, "dir": str(args.roi_library), "names": lib.names, "count": len(lib)})


@app.post("/api/roi/upload")
def roi_upload(payload: dict[str, Any]) -> JSONResponse:
    data_url = payload.get("image") or ""
    name = payload.get("name") or "roi"
    if not data_url.startswith("data:"):
        return JSONResponse({"ok": False, "error": "缺少图片数据"})
    try:
        img = roi_library.load_image_from_data_url(data_url)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"图片无效: {exc}"})

    # 文件名净化：只保留字母数字 _-. ，防止路径穿越/特殊字符
    safe = "".join(c for c in Path(name).name if c.isalnum() or c in "._-").strip()
    if not safe:
        safe = "roi"
    if not safe.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        safe += ".png"

    lib_dir = Path(args.roi_library)
    try:
        lib_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return JSONResponse({"ok": False, "error": f"无法创建库目录: {exc}"})

    final = lib_dir / safe
    stem, suffix = final.stem, final.suffix
    i = 1
    while final.exists():
        final = lib_dir / f"{stem}_{i}{suffix}"
        i += 1
    ok = cv2.imwrite(str(final), img)
    if not ok:
        return JSONResponse({"ok": False, "error": "图片保存失败"})
    lib = _get_roi_library()
    return JSONResponse({"ok": True, "name": final.name, "library_size": len(lib) if lib else 0})


@app.delete("/api/roi/{name}")
def roi_delete(name: str) -> JSONResponse:
    safe = Path(name).name  # 防路径穿越
    path = Path(args.roi_library) / safe
    if not path.is_file():
        return JSONResponse({"ok": False, "error": f"不存在: {safe}"})
    try:
        path.unlink()
    except OSError as exc:
        return JSONResponse({"ok": False, "error": f"删除失败: {exc}"})
    return JSONResponse({"ok": True, "name": safe})


@app.get("/api/roi/image/{name}")
def roi_library_image(name: str):  # noqa: ANN201
    safe = Path(name).name  # 防路径穿越
    path = Path(args.roi_library) / safe
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


DEFAULT_ROI_SYSTEM_PROMPT = (
    "你是一个严谨的工业 ROI 图像比对专家，只依据图片中的实际内容客观判断，"
    "不要臆测或编造，也不要被文件名误导。"
)

DEFAULT_ROI_USER_PROMPT = (
    "第一张图是【待检 ROI】，后面的图是库中的 ROI 图，标注为：{labels}。"
    "请仔细观察并逐张比较待检图与库中每张图，判断待检 ROI 与库中哪一张最相似，"
    "以及整体上是否与库中某张相似。严格按以下格式回答，不要输出多余内容：\n"
    "最相似: <字母:文件名 或 无>\n"
    "是否相似: <是/否>\n"
    "理由: <一两句话>"
)


def _downscale_for_llm(data_url: str, max_edge: int = 768) -> str:
    """把图缩到 <= max_edge，减少多图对比时的视觉 token 数量。"""
    _, _, b64 = data_url.partition(",")
    raw = base64.b64decode(b64)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("无法解码图片")
    h, w = arr.shape[:2]
    long_edge = max(h, w)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        arr = cv2.resize(arr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("图片压缩失败")
    return f"data:image/jpeg;base64,{base64.b64encode(enc.tobytes()).decode()}"


def _roi_llm_payload(
    query_data_url: str,
    names: list[str],
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    lib = _get_roi_library()
    if lib is None:
        raise FileNotFoundError(f"ROI 库目录不存在: {args.roi_library}")

    content: list[Any] = [{"type": "image_url", "image_url": {"url": _downscale_for_llm(query_data_url)}}]
    labels: list[str] = []
    for i, name in enumerate(names, 1):
        path = Path(args.roi_library) / Path(name).name  # 防路径穿越
        if not path.is_file():
            continue
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": _downscale_for_llm(f"data:image/jpeg;base64,{b64}")}})
        labels.append(f"{chr(64 + i)}: {name}")

    if not labels:
        raise ValueError("ROI 库为空，无法比对")

    system_text = (system_prompt or "").strip() or DEFAULT_ROI_SYSTEM_PROMPT
    user_template = (user_prompt or "").strip() or DEFAULT_ROI_USER_PROMPT
    prompt_text = (
        user_template
        .replace("{labels}", "，".join(labels))
        .replace("{query}", "待检 ROI")
    )

    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    content.append({"type": "text", "text": prompt_text})
    messages.append({"role": "user", "content": content})
    return {
        "messages": messages,
        "max_tokens": 220,
        "temperature": 0.2,
        "top_p": 0.9,
    }


@app.post("/api/roi/llm-match")
def roi_llm_match(payload: dict[str, Any]):  # noqa: ANN201
    data_url = payload.get("image") or ""
    names = payload.get("names") or []
    if not data_url.startswith("data:"):
        return JSONResponse({"ok": False, "error": "缺少图片数据"})
    try:
        llm_payload = _roi_llm_payload(
            data_url,
            names,
            system_prompt=payload.get("system_prompt"),
            user_prompt=payload.get("prompt") or payload.get("user_prompt"),
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)})
    return StreamingResponse(_sse_stream(_stream_worker(llm_payload)), media_type="text/event-stream")


@app.post("/api/roi/match")
def roi_match(payload: dict[str, Any]) -> JSONResponse:
    lib = _get_roi_library()
    if lib is None:
        return JSONResponse({"ok": False, "error": f"ROI 库目录不存在: {args.roi_library}"})
    data_url = payload.get("image") or ""
    if not data_url.startswith("data:"):
        return JSONResponse({"ok": False, "error": "缺少图片数据"})
    top_k = max(1, min(int(payload.get("top_k", 3)), 20))
    threshold = float(payload.get("threshold", 0.55))
    try:
        img = roi_library.load_image_from_data_url(data_url)
        results = lib.match(img, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)})
    return JSONResponse({
        "ok": True,
        "library_size": len(lib),
        "threshold": threshold,
        "matches": [r.as_dict() for r in results],
    })


@app.get("/health")
def health() -> dict[str, Any]:
    vision = bool(args.mmproj and args.mmproj.exists())
    busy = bool(_busy)
    return {"ok": True, "model": str(args.model), "vision": vision, "busy": busy}


def main() -> None:
    global args
    args = build_parser().parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

