"""
Train a tiny CNN on MNIST and export its weights for the HTML demo (index.html).

The network architecture here MUST match the JS forward pass in index.html:
    Conv2d(1->16, 3x3, pad=1) + ReLU -> MaxPool2d(2)
    Conv2d(16->32, 3x3, pad=1) + ReLU -> MaxPool2d(2)
    Flatten(1568) -> Linear(1568->64) + ReLU -> Linear(64->10) -> Softmax

Usage:
    python train_mnist.py                       # train + write mnist_weights.json
    python train_mnist.py --inject index.html   # train, then inline weights into index.html
    python train_mnist.py --inject-only index.html  # inline existing weights (no retrain)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS_PATH = ROOT / "mnist_weights.json"
OUT_CH = {"conv1": 16, "conv2": 32}
MEAN, STD = 0.1307, 0.2810

# Frontend placeholder marker for --inject.
# The marker is a comment immediately before the placeholder value (null);
# injection replaces "<marker>null;" with "<json>;".
MARKER = "/*__MNIST_WEIGHTS__*/"


def inject_weights(html_path_str: str, weights: dict) -> bool:
    """Inline `weights` JSON into index.html by replacing the marker block."""
    import re

    html_path = Path(html_path_str)
    html = html_path.read_text(encoding="utf-8")
    pattern = re.escape(MARKER) + r"[^;]*;"
    json_str = json.dumps(weights, ensure_ascii=False)
    new_html, n = re.subn(pattern, json_str + ";", html, count=1)
    if n == 0:
        print(f"WARN: placeholder {MARKER!r} not found in {html_path}")
        return False
    html_path.write_text(new_html, encoding="utf-8")
    print(f"weights injected into {html_path}")
    return True


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, OUT_CH["conv1"], 3, padding=1),   # 0  conv1
        nn.ReLU(),                                      # 1
        nn.MaxPool2d(2),                                # 2
        nn.Conv2d(OUT_CH["conv1"], OUT_CH["conv2"], 3, padding=1),  # 3 conv2
        nn.ReLU(),                                      # 4
        nn.MaxPool2d(2),                                # 5
        nn.Flatten(),                                   # 6
        nn.Linear(7 * 7 * OUT_CH["conv2"], 64),         # 7  fc1
        nn.ReLU(),                                      # 8
        nn.Linear(64, 10),                              # 9  fc2
    )


def layer_names() -> dict:
    return {"0": "conv1", "3": "conv2", "7": "fc1", "9": "fc2"}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train(model: nn.Module, train_loader: DataLoader, epochs: int, device: torch.device) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            total += y.numel()
            correct += (out.argmax(1) == y).sum().item()
        print(f"  epoch {epoch:2d}/{epochs}  loss {loss_sum/total:.4f}  acc {correct/total*100:.2f}%")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    probs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = torch.softmax(model(x), dim=1)
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
        probs.append((x.cpu(), y.cpu(), out.cpu()))
    return correct / total, probs


@torch.no_grad()
def pick_samples(model: nn.Module, loader: DataLoader, device: torch.device,
                 digits: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)) -> list:
    """Pick one test sample per digit, preferring high-confidence correct ones."""
    model.eval()
    best = {}  # label -> (prob, pixels)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = torch.softmax(model(x), dim=1)
        for i in range(x.size(0)):
            label = int(y[i])
            if label not in digits:
                continue
            conf = float(out[i, label])
            if out[i].argmax() != y[i]:
                conf = -1.0  # prefer correct samples; keep as last resort
            if label not in best or conf > best[label][0]:
                pixels = (x[i].squeeze().mul(STD).add(MEAN).mul(255)
                          .clamp(0, 255).round().reshape(-1).tolist())
                best[label] = (conf, pixels)
    return [{"label": d, "pixels": best[d][1]} for d in digits]


def export_weights(model: nn.Module, test_acc: float, samples: list) -> dict:
    names = layer_names()
    out = {
        "meta": {
            "input_size": 28,
            "num_classes": 10,
            "mean": MEAN,
            "std": STD,
            "params": count_params(model),
            "test_acc": round(test_acc, 4),
        },
        "samples": samples,
    }
    for key, tensor in model.state_dict().items():
        idx, kind = key.split(".")
        name = names[idx]
        tensor = tensor.to("cpu")
        flat = [round(float(v), 4) for v in tensor.flatten().tolist()]
        if kind == "weight":
            out[name] = {"w": flat, "w_shape": list(tensor.shape)}
        else:
            out[name]["b"] = flat
            out[name]["b_shape"] = list(tensor.shape)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tiny MNIST CNN and export weights for index.html")
    ap.add_argument("--inject", metavar="INDEX_HTML", help="inline weights into index.html (after training)")
    ap.add_argument("--inject-only", metavar="INDEX_HTML",
                    help="inline existing mnist_weights.json without retraining")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.inject_only:
        with open(WEIGHTS_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        inject_weights(args.inject_only, existing)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    print(f"downloading MNIST to {DATA_DIR} if needed ...")
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model().to(device)
    print(f"model params: {count_params(model):,}")
    print("training ...")
    train(model, train_loader, args.epochs, device)

    print("evaluating on test set ...")
    acc, _ = evaluate(model, test_loader, device)
    print(f"test accuracy: {acc*100:.2f}%")

    print("picking sample digits ...")
    samples = pick_samples(model, test_loader, device)

    weights = export_weights(model, acc, samples)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False)
    print(f"weights written to {WEIGHTS_PATH} "
          f"({WEIGHTS_PATH.stat().st_size/1024:.0f} KB)")

    if args.inject:
        inject_weights(args.inject, weights)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
