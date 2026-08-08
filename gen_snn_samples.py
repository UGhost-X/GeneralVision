"""
Extract a compact MNIST subset for snn_demo.html and inline it into the page.

For each digit: 40 train samples (STDP learning) + 6 test samples (inference).
Pixels are stored as base64 of raw 0..255 bytes (784 per sample) so the
self-contained page stays small and needs no network at runtime.

Usage: .venv/Scripts/python.exe gen_snn_samples.py snn_demo.html
"""
import base64
import json
import re
import sys
from pathlib import Path

from torchvision import datasets

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MARKER = "/*__SNN_SAMPLES__*/"
TRAIN_PER_DIGIT = 40
TEST_PER_DIGIT = 6


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: gen_snn_samples.py <snn_demo.html>")
        sys.exit(1)
    html_path = Path(sys.argv[1])

    ds = datasets.MNIST(DATA_DIR, train=True, download=False)
    imgs = ds.data.numpy()
    labels = ds.targets.numpy()

    samples = []
    for d in range(10):
        idx = labels == d
        x = imgs[idx]
        n = len(x)
        train = x[:TRAIN_PER_DIGIT]
        test = x[TRAIN_PER_DIGIT:TRAIN_PER_DIGIT + TEST_PER_DIGIT]
        for split, group in (("train", train), ("test", test)):
            for px in group:
                raw = px.reshape(-1).tobytes()
                samples.append({"p": base64.b64encode(raw).decode(), "l": d, "s": split})

    json_str = json.dumps(samples, separators=(",", ":"))
    html = html_path.read_text(encoding="utf-8")
    pattern = re.escape(MARKER) + r"\s*null;"
    new_html, n = re.subn(pattern, json_str + ";", html, count=1)
    if n == 0:
        print(f"WARN: marker {MARKER!r} not found in {html_path}")
        sys.exit(1)
    html_path.write_text(new_html, encoding="utf-8")
    print(f"injected {len(samples)} samples ({len(json_str)/1024:.0f} KB json) into {html_path}")


if __name__ == "__main__":
    main()
