#!/usr/bin/env python3
"""ROI 图片库相似度匹配模块。

对库中的每张 ROI 图预计算特征：
  1. pHash 感知哈希              -> 近重复/轻微变化（强信号，可单独判定）
  2. NCC 归一化互相关（64x64）   -> 同姿态 ROI 的结构相似度
  3. SIFT/ORB 特征点匹配         -> 同目标不同角度/尺度
  4. HSV 颜色直方图              -> 颜色相似度（低权重辅助）

综合分数 = 加权求和，且 pHash 极高时直接判定相似（近重复覆盖）。

用法:
  python scripts/roi_library.py --library <目录> --query <图片> [--top-k 3] [--json]
  python scripts/roi_library.py --library <目录> --list
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PHASH_SIZE = 32   # pHash 缩放尺寸
NCC_SIZE = 64     # NCC 归一化尺寸


@dataclass
class MatchResult:
    name: str
    phash_score: float
    ncc_score: float
    shape_score: float
    feature_score: float
    hist_score: float
    combined_score: float
    similar: bool
    override: bool = False
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "phash_score": round(self.phash_score, 4),
            "ncc_score": round(self.ncc_score, 4),
            "shape_score": round(self.shape_score, 4),
            "feature_score": round(self.feature_score, 4),
            "hist_score": round(self.hist_score, 4),
            "combined_score": round(self.combined_score, 4),
            "similar": self.similar,
            "near_duplicate_override": self.override,
            "details": self.details,
        }


# ---------- 特征函数 ----------
def _phash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    low = dct[:8, :8]
    bits = (low > low.mean()).flatten()
    val = 0
    for i, b in enumerate(bits[:64]):
        if b:
            val |= 1 << i
    return val


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _ncc_score(a64: np.ndarray, b64: np.ndarray) -> float:
    a = a64.astype(np.float32) - a64.astype(np.float32).mean()
    b = b64.astype(np.float32) - b64.astype(np.float32).mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return max(0.0, float(np.dot(a.ravel(), b.ravel()) / denom))


def _feature_matcher():
    """优先 SIFT，失败则回退 ORB。"""
    try:
        return cv2.SIFT_create(nfeatures=800)
    except (AttributeError, cv2.error):
        return cv2.ORB_create(nfeatures=800)


def _feature_match_score(des_q, des_lib, norm) -> tuple[float, dict]:
    if des_q is None or des_lib is None or len(des_q) < 4 or len(des_lib) < 4:
        return 0.0, {"good_matches": 0, "total_q": 0, "total_lib": 0}
    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(des_q, des_lib, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    ratio = min(1.0, len(good) / max(len(des_q), 1))
    return float(ratio), {"good_matches": len(good), "total_q": len(des_q), "total_lib": len(des_lib)}


def _largest_contour(gray: np.ndarray):
    """提取最大外轮廓（用于形状比较）。"""
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bin_img) > 127:  # 深色形状在浅色底 -> 反转
        bin_img = cv2.bitwise_not(bin_img)
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _shape_score(gray_a: np.ndarray, gray_b: np.ndarray, k: float = 5.0) -> float:
    """基于 Hu 矩（matchShapes）的形状相似度，旋转/尺度/平移不变。"""
    ca = _largest_contour(gray_a)
    cb = _largest_contour(gray_b)
    if ca is None or cb is None:
        return 0.0
    d = cv2.matchShapes(ca, cb, 1, 0)  # method I2
    return float(np.exp(-k * d))


def _hist_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    def hist(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])

    ha, hb = hist(img_a), hist(img_b)
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    return max(0.0, float(corr))


class RoiLibrary:
    def __init__(
        self,
        library_dir: str | Path,
        phash_weight: float = 0.30,
        ncc_weight: float = 0.15,
        shape_weight: float = 0.35,
        feature_weight: float = 0.15,
        hist_weight: float = 0.05,
        similar_threshold: float = 0.55,
        near_dup_threshold: float = 0.94,
    ):
        self.library_dir = Path(library_dir)
        self.phash_weight = phash_weight
        self.ncc_weight = ncc_weight
        self.shape_weight = shape_weight
        self.feature_weight = feature_weight
        self.hist_weight = hist_weight
        self.similar_threshold = similar_threshold
        self.near_dup_threshold = near_dup_threshold
        self._entries: dict[str, dict] = {}
        self._matcher = None
        self._norm = None
        self._init_matcher()
        self.load()

    def _init_matcher(self) -> None:
        self._matcher = _feature_matcher()
        self._norm = cv2.NORM_L2 if isinstance(self._matcher, cv2.SIFT) else cv2.NORM_HAMMING

    # ---------- 库构建 ----------
    def load(self) -> None:
        if not self.library_dir.is_dir():
            raise FileNotFoundError(f"ROI 库目录不存在: {self.library_dir}")
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = sorted(
            p for p in self.library_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        # 空库也允许（网页端可空库显示），只是没有任何条目
        for path in files:
            self.add(path)

    def add(self, path: str | Path) -> None:
        path = Path(path)
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"无法读取图片: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = self._matcher.detectAndCompute(gray, None)
        self._entries[path.name] = {
            "path": path,
            "phash": _phash64(gray),
            "gray": gray,
            "ncc": cv2.resize(gray, (NCC_SIZE, NCC_SIZE), interpolation=cv2.INTER_AREA),
            "des": des,
            "image": img,
        }

    @property
    def names(self) -> list[str]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ---------- 匹配 ----------
    def match(self, query_img: np.ndarray, top_k: int = 3) -> list[MatchResult]:
        gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        _, des_q = self._matcher.detectAndCompute(gray, None)
        ncc_q = cv2.resize(gray, (NCC_SIZE, NCC_SIZE), interpolation=cv2.INTER_AREA)
        phash_q = _phash64(gray)

        results: list[MatchResult] = []
        for name, e in self._entries.items():
            phash_score = 1.0 - _hamming(phash_q, e["phash"]) / 64.0
            ncc_score = _ncc_score(ncc_q, e["ncc"])
            shape_score = _shape_score(gray, e["gray"])
            feature_score, feat_details = _feature_match_score(des_q, e["des"], self._norm)
            hist_score = _hist_score(query_img, e["image"])
            combined = (
                self.phash_weight * phash_score
                + self.ncc_weight * ncc_score
                + self.shape_weight * shape_score
                + self.feature_weight * feature_score
                + self.hist_weight * hist_score
            )
            override = (
                phash_score >= self.near_dup_threshold and ncc_score >= 0.92
            )
            similar = override or combined >= self.similar_threshold
            results.append(
                MatchResult(
                    name=name,
                    phash_score=phash_score,
                    ncc_score=ncc_score,
                    shape_score=shape_score,
                    feature_score=feature_score,
                    hist_score=hist_score,
                    combined_score=combined,
                    similar=similar,
                    override=override,
                    details={
                        "phash_distance": 64 - round(phash_score * 64),
                        **feat_details,
                    },
                )
            )
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    def match_path(self, query_path: str | Path, top_k: int = 3) -> list[MatchResult]:
        img = cv2.imread(str(query_path))
        if img is None:
            raise ValueError(f"无法读取查询图: {query_path}")
        return self.match(img, top_k=top_k)



def load_image_from_data_url(data_url: str) -> np.ndarray:
    """从 data URL（base64）解码图片，供 Web 接口使用。"""
    if not data_url.startswith("data:"):
        raise ValueError("不是合法的 data URL")
    _, _, b64 = data_url.partition(",")
    raw = base64.b64decode(b64)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("无法解码上传的图片")
    return arr

def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(description="ROI 图片库相似度匹配")
    p.add_argument("--library", required=True, help="ROI 库目录")
    p.add_argument("--query", help="查询图片路径")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.55, help="相似度阈值(0~1)")
    p.add_argument("--list", action="store_true", help="列出库中 ROI")
    p.add_argument("--json", action="store_true", help="JSON 输出")
    args = p.parse_args()

    try:
        lib = RoiLibrary(args.library, similar_threshold=args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    if args.list:
        for name in lib.names:
            print(name)
        return 0

    if not args.query:
        print("[error] 需要 --query 或 --list", file=sys.stderr)
        return 1

    results = lib.match_path(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2))
        return 0

    print(f"ROI 库: {args.library} ({len(lib)} 张)")
    print(f"查询图: {args.query}")
    print(f"{'排名':<4}{'ROI 文件':<26}{'pHash':<8}{'NCC':<8}{'形状':<8}{'特征':<8}{'直方图':<8}{'综合':<8}{'判定'}")
    for i, r in enumerate(results, 1):
        verdict = "相似 [Y]" if r.similar else "不相似 [N]"
        if r.override:
            verdict += "(近重复)"
        print(
            f"{i:<4}{r.name:<26}{r.phash_score:<8.3f}{r.ncc_score:<8.3f}"
            f"{r.shape_score:<8.3f}{r.feature_score:<8.3f}{r.hist_score:<8.3f}"
            f"{r.combined_score:<8.3f}{verdict}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
