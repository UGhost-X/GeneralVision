#!/usr/bin/env python3
"""Load a GGUF model and run text inference (single prompt or interactive chat)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from llama_cpp import Llama


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference with a GGUF model via llama.cpp")
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "Qwen3.5-0.8B-Q4_K_M.gguf",
        help="Path to the .gguf model file",
    )
    p.add_argument("--prompt", type=str, default='你好', help="Single-turn prompt (omit for interactive chat)")
    p.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt for chat mode")
    p.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p")
    p.add_argument("--n-ctx", type=int, default=2048, help="Context window size")
    p.add_argument("--n-threads", type=int, default=None, help="Number of CPU threads (default: auto)")
    p.add_argument("--verbose", action="store_true", help="Print llama.cpp verbose logs")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if not args.model.exists():
        print(f"[error] model file not found: {args.model}", file=sys.stderr)
        return 1

    print(f"[info] loading model: {args.model}", file=sys.stderr)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=0,  # CPU-only inference
        verbose=args.verbose,
    )
    print(f"[info] model loaded in {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    def show_result(out: dict) -> None:
        usage = out.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        print(out["choices"][0]["message"]["content"].strip())
        elapsed = out.get("_elapsed", 0.0)
        if elapsed > 0:
            print(
                f"\n[stats] prompt={prompt_tokens} tok, "
                f"generated={completion_tokens} tok, "
                f"elapsed={elapsed:.2f}s, "
                f"speed={completion_tokens / elapsed:.1f} tok/s",
                file=sys.stderr,
            )

    if args.prompt is not None:
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        out["_elapsed"] = time.perf_counter() - t0
        show_result(out)
        return 0

    # Interactive chat
    print("Interactive chat mode. Type 'exit' or Ctrl+C to quit.\n", file=sys.stderr)
    history: list[dict[str, str]] = [{"role": "system", "content": args.system}]
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user})
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=history,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        out["_elapsed"] = time.perf_counter() - t0
        answer = out["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": answer})
        show_result(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
