#!/usr/bin/env python3
"""Download HAT / Restormer / torch-CPU models with resume + retries."""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = Path(os.environ.get("TEMP", ".")) / "roi_models_download.log"

JOBS = [
    # (urls, dest, expected_bytes)  -- first URL that works wins
    (["https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth"],
     ROOT / "models" / "hat" / "Real_HAT_GAN_sharper.pth", 170277017),
    (["https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth"],
     ROOT / "models" / "hat" / "HAT_SRx4_ImageNet-pretrain.pth", 85137601),
    (["https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
      "https://huggingface.co/deepinv/Restormer/resolve/main/motion_deblurring.pth"],
     ROOT / "models" / "restormer" / "motion_deblurring.pth", 104700429),
    (["https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth",
      "https://huggingface.co/deepinv/Restormer/resolve/main/single_image_defocus_deblurring.pth"],
     ROOT / "models" / "restormer" / "single_image_defocus_deblurring.pth", 104700429),
    (["https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth",
      "https://huggingface.co/deepinv/Restormer/resolve/main/real_denoising.pth"],
     ROOT / "models" / "restormer" / "real_denoising.pth", 104611957),
    (["https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_gray_denoising_sigma25.pth",
      "https://huggingface.co/deepinv/Restormer/resolve/main/gaussian_gray_denoising_sigma25.pth"],
     ROOT / "models" / "restormer" / "gaussian_gray_denoising_sigma25.pth", 104601589),
    (["https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.13.0%2Bcpu-cp310-cp310-win_amd64.whl"],
     Path(os.environ.get("TEMP", ".")) / "roi_models" / "torch_cpu.whl", 121891790),
]

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    for urls, dest, expected in JOBS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        name = dest.name
        size = dest.stat().st_size if dest.exists() else 0
        if size >= expected:
            log(f"SKIP {name} ({size})")
            continue
        log(f"START {name} (have {size}/{expected})")
        ok = False
        for url in urls:
            for attempt in range(12):
                if dest.exists() and dest.stat().st_size >= expected:
                    ok = True
                    break
                cmd = ["curl.exe", "-sL", "--retry", "5", "--retry-delay", "3", "-C", "-",
                       "--max-time", "500", "-o", str(dest), url,
                       "-w", f"{name} src={url.split('/')[2]} attempt={attempt} http=%{{http_code}} size=%{{size_download}}\n"]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=520)
                    if r.stdout:
                        log(r.stdout.strip())
                    if r.stderr:
                        log(r.stderr.strip()[-300:])
                except subprocess.TimeoutExpired:
                    log(f"{name} attempt={attempt} TIMEOUT")
                size = dest.stat().st_size if dest.exists() else 0
                log(f"{name} progress {size}/{expected}")
                if size >= expected:
                    ok = True
                    break
                time.sleep(3)
            if ok:
                break
            log(f"{name} switch source")
        log(f"DONE  {name} -> {size}" if ok else f"FAIL  {name} -> {size}")
    log("ALL DONE")

if __name__ == "__main__":
    main()
