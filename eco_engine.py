from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit


DATA_DIR = Path(__file__).resolve().parent / "data" / "MNIST" / "raw"

T = 40
INPUT_SIZE = 784
INPUT_SAMPLES_PER_STEP = 12
MAX_HIDDEN_SIZE = 200
MAX_DEPTH = 4
MAX_TOTAL_HIDDEN = 400
HIDDEN_SIZE = 100

LEAK = 0.94
THETA_HIDDEN = 12.0

INPUT_W_SCALE = 0.65
HIDDEN_W_SCALE = 0.75
OUTPUT_W_SCALE = 2.5
HIDDEN_BIAS_SCALE = 0.12
OUTPUT_BIAS_SCALE = 0.05

P_GROW = 0.40
P_SPLIT = 0.15
P_MERGE = 0.10
P_PRUNE = 0.10
P_ADDRANDOM = 0.03
BIG_MUTATION_RATE = 0.001

SURVIVAL_ROUNDS = 20
N_REPRO = 50
ASSORT_STRENGTH = 0.5
CAPACITY = 10000
INIT_POP = 1000
SELECT_PER_DIGIT = 100
FITNESS_LAMBDA = 0.5
SCREEN_POOL_SIZE = 2000
SCREEN_STEPS = 8
SCREEN_SAMPLES = 4
SCREEN_READOUT_SAMPLES_PER_DIGIT = 1
READOUT_LAMBDA = 0.1
P_READOUT_MUTATION = 0.30
P_READOUT_SPARSE_RESET = 0.05
TRAIT_MUTATION_RATE = 0.25
REPRO_SUCCESS_BASE = 0.9
REPRO_WRONG_PENALTY = 0.4
NO_REPRO_DEATH_ROUNDS = 3
REPRO_GROWTH_DIVISOR = 30.0
DENSITY_FLOOR = 0.05
POP_GROWTH = 0.10

MAX_WEIGHTS = (
    (INPUT_SIZE + 1) * MAX_HIDDEN_SIZE
    + (MAX_HIDDEN_SIZE + 1) * MAX_HIDDEN_SIZE
    + (MAX_HIDDEN_SIZE + 1) * 10
)


def _load_idx(path: Path) -> np.ndarray:
    if path.exists():
        raw = path.read_bytes()
    else:
        gz_path = Path(str(path) + ".gz")
        if not gz_path.exists():
            raise FileNotFoundError(f"MNIST file not found: {path}")
        with gzip.open(gz_path, "rb") as fh:
            raw = fh.read()

    magic = int.from_bytes(raw[:4], "big")
    ndim = magic & 0xFF
    dims = [
        int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big")
        for i in range(ndim)
    ]
    payload = np.frombuffer(raw[4 + 4 * ndim :], dtype=np.uint8)
    if ndim == 3:
        return payload.reshape(dims[0], dims[1], dims[2])
    if ndim == 1:
        return payload.reshape(dims[0])
    return payload


_MNIST_CACHE: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _MNIST_CACHE
    if _MNIST_CACHE is None:
        train_images = _load_idx(DATA_DIR / "train-images-idx3-ubyte")
        train_labels = _load_idx(DATA_DIR / "train-labels-idx1-ubyte")
        test_images = _load_idx(DATA_DIR / "t10k-images-idx3-ubyte")
        test_labels = _load_idx(DATA_DIR / "t10k-labels-idx1-ubyte")
        _MNIST_CACHE = (
            train_images.reshape(-1, INPUT_SIZE).astype(np.float32) / 255.0,
            train_labels.astype(np.int32),
            test_images.reshape(-1, INPUT_SIZE).astype(np.float32) / 255.0,
            test_labels.astype(np.int32),
        )
    return _MNIST_CACHE


def _sample_spikes(
    image: np.ndarray,
    rng: np.random.Generator,
    steps: int = T,
    samples_per_step: int = INPUT_SAMPLES_PER_STEP,
) -> Tuple[np.ndarray, np.ndarray]:
    intensity = np.maximum(image.astype(np.float64), 0.0)
    total = float(intensity.sum())
    if total <= 1e-8:
        p = np.full(INPUT_SIZE, 1.0 / INPUT_SIZE, dtype=np.float64)
    else:
        p = intensity ** 1.5
        p /= p.sum()

    idx = rng.choice(
        INPUT_SIZE,
        size=(steps, samples_per_step),
        replace=True,
        p=p,
    ).astype(np.int32)
    vals = np.zeros((steps, samples_per_step), dtype=np.float32)
    vals[image[idx] > 0.08] = 1.0
    return idx, vals


@dataclass
class Genome:
    layer_sizes: Tuple[int, ...]
    weights: np.ndarray
    longevity_bonus: int = 0
    fecundity: float = 1.0
    wrong_tolerance: float = 1.0
    mutation_rate: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_sizes": list(self.layer_sizes),
            "weights_size": int(self.weights.size),
            "longevity_bonus": self.longevity_bonus,
            "fecundity": self.fecundity,
            "wrong_tolerance": self.wrong_tolerance,
            "mutation_rate": self.mutation_rate,
        }


def _random_weights(
    layer_sizes: Sequence[int],
    rng: np.random.Generator,
    silent: bool = False,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    prev = INPUT_SIZE
    for idx, n in enumerate(layer_sizes):
        if silent:
            mat = rng.normal(0.0, 0.005, size=(prev, n)).astype(np.float32)
            bias = rng.normal(0.0, 0.005, size=n).astype(np.float32)
        elif idx == 0:
            mat = rng.normal(0.0, INPUT_W_SCALE, size=(prev, n)).astype(np.float32)
            bias = rng.normal(0.0, HIDDEN_BIAS_SCALE, size=n).astype(np.float32)
        else:
            mat = rng.normal(
                0.0,
                HIDDEN_W_SCALE / math.sqrt(max(prev, 1)),
                size=(prev, n),
            ).astype(np.float32)
            bias = rng.normal(0.0, HIDDEN_BIAS_SCALE, size=n).astype(np.float32)
        parts.append(mat.ravel())
        parts.append(bias)
        prev = n

    out_mat = rng.normal(
        0.0,
        OUTPUT_W_SCALE / math.sqrt(max(prev, 1)),
        size=(prev, 10),
    ).astype(np.float32)
    out_bias = rng.normal(0.0, OUTPUT_BIAS_SCALE, size=10).astype(np.float32)
    parts.append(out_mat.ravel())
    parts.append(out_bias)
    return np.concatenate(parts).astype(np.float32)


def random_genome(rng: Optional[np.random.Generator] = None) -> Genome:
    if rng is None:
        rng = np.random.default_rng()
    return Genome(
        (HIDDEN_SIZE,),
        _random_weights((HIDDEN_SIZE,), rng),
        longevity_bonus=int(rng.integers(0, 3)),
        fecundity=float(rng.uniform(0.8, 1.2)),
        wrong_tolerance=float(rng.uniform(0.8, 1.2)),
        mutation_rate=float(rng.uniform(0.8, 1.2)),
    )


def _unflatten(
    flat: np.ndarray,
    layer_sizes: Sequence[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    hidden_mats: List[np.ndarray] = []
    hidden_biases: List[np.ndarray] = []
    offset = 0
    prev = INPUT_SIZE
    for n in layer_sizes:
        count = prev * n + n
        mat = flat[offset : offset + prev * n].reshape(prev, n)
        bias = flat[offset + prev * n : offset + count]
        hidden_mats.append(mat)
        hidden_biases.append(bias)
        offset += count
        prev = n
    count = prev * 10 + 10
    out_mat = flat[offset : offset + prev * 10].reshape(prev, 10)
    out_bias = flat[offset + prev * 10 : offset + count]
    return hidden_mats, hidden_biases, out_mat, out_bias


def _flatten(
    hidden_mats: Sequence[np.ndarray],
    hidden_biases: Sequence[np.ndarray],
    out_mat: np.ndarray,
    out_bias: np.ndarray,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for mat, bias in zip(hidden_mats, hidden_biases):
        parts.append(np.asarray(mat).ravel())
        parts.append(np.asarray(bias).ravel())
    parts.append(np.asarray(out_mat).ravel())
    parts.append(np.asarray(out_bias).ravel())
    return np.concatenate(parts)


def _output_offset(layer_sizes: Sequence[int]) -> int:
    offset = 0
    prev = INPUT_SIZE
    for n_hidden in layer_sizes:
        offset += prev * n_hidden + n_hidden
        prev = n_hidden
    return offset


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _grow_silent(
    layer_sizes: List[int],
    mats: List[np.ndarray],
    biases: List[np.ndarray],
    out_mat: np.ndarray,
    out_bias: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    if len(layer_sizes) >= MAX_DEPTH or sum(layer_sizes) >= MAX_TOTAL_HIDDEN:
        return False, out_mat, out_bias
    layer_idx = int(rng.integers(len(layer_sizes)))
    n = layer_sizes[layer_idx]
    if n >= MAX_HIDDEN_SIZE:
        return False, out_mat, out_bias
    pos = int(rng.integers(n + 1))
    prev_n = INPUT_SIZE if layer_idx == 0 else layer_sizes[layer_idx - 1]
    next_n = (
        layer_sizes[layer_idx + 1]
        if layer_idx + 1 < len(layer_sizes)
        else 10
    )

    old_in = mats[layer_idx]
    old_out = (
        mats[layer_idx + 1]
        if layer_idx + 1 < len(mats)
        else out_mat
    )
    new_in = np.zeros((prev_n, n + 1), dtype=np.float32)
    new_in[:, :pos] = old_in[:, :pos]
    new_in[:, pos + 1 :] = old_in[:, pos:]
    new_in[:, pos] = rng.normal(0.0, 0.005, size=prev_n).astype(np.float32)

    new_out = np.zeros((n + 1, next_n), dtype=np.float32)
    new_out[:pos, :] = old_out[:pos, :]
    new_out[pos + 1 :, :] = old_out[pos:, :]
    new_out[pos, :] = rng.normal(0.0, 0.005, size=next_n).astype(np.float32)

    mats[layer_idx] = new_in
    biases[layer_idx] = np.concatenate(
        [biases[layer_idx][:pos], np.zeros(1, np.float32), biases[layer_idx][pos:]]
    ).astype(np.float32)
    if layer_idx + 1 < len(mats):
        mats[layer_idx + 1] = new_out
    else:
        out_mat = new_out
    layer_sizes[layer_idx] = n + 1
    return True, out_mat, out_bias


def _split_identity(
    layer_sizes: List[int],
    mats: List[np.ndarray],
    biases: List[np.ndarray],
    out_mat: np.ndarray,
    out_bias: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    if len(layer_sizes) >= MAX_DEPTH:
        return False, out_mat, out_bias
    layer_idx = int(rng.integers(len(layer_sizes)))
    n = layer_sizes[layer_idx]
    if sum(layer_sizes) + n > MAX_TOTAL_HIDDEN:
        return False, out_mat, out_bias

    old_out = mats[layer_idx + 1] if layer_idx + 1 < len(mats) else out_mat
    identity = np.eye(n, dtype=np.float32) * (THETA_HIDDEN + 2.0)
    mats.insert(layer_idx + 1, identity)
    biases.insert(layer_idx + 1, np.zeros(n, dtype=np.float32))
    if layer_idx + 1 < len(mats) - 1:
        mats[layer_idx + 2] = old_out
    else:
        out_mat = old_out
    layer_sizes.insert(layer_idx + 1, n)
    return True, out_mat, out_bias


def _merge_layers(
    layer_sizes: List[int],
    mats: List[np.ndarray],
    biases: List[np.ndarray],
    out_mat: np.ndarray,
    out_bias: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    if len(layer_sizes) < 2:
        return False, out_mat, out_bias
    layer_idx = int(rng.integers(len(layer_sizes) - 1))
    w1 = mats[layer_idx]
    w2 = mats[layer_idx + 1]
    b1 = biases[layer_idx]
    b2 = biases[layer_idx + 1]
    n2 = w2.shape[1]

    merged_w = w1 @ w2
    merged_b = w2.T @ b1 + b2
    old_out = mats[layer_idx + 2] if layer_idx + 2 < len(mats) else out_mat

    mats[layer_idx] = merged_w
    biases[layer_idx] = merged_b
    del mats[layer_idx + 1]
    del biases[layer_idx + 1]
    if layer_idx + 1 < len(mats):
        mats[layer_idx + 1] = old_out
    else:
        out_mat = old_out
    layer_sizes[layer_idx] = n2
    del layer_sizes[layer_idx + 1]
    return True, out_mat, out_bias


def _prune_neurons(
    layer_sizes: List[int],
    mats: List[np.ndarray],
    biases: List[np.ndarray],
    out_mat: np.ndarray,
    out_bias: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    candidates = [i for i, n in enumerate(layer_sizes) if n > 20]
    if not candidates:
        return False, out_mat, out_bias
    layer_idx = int(candidates[rng.integers(len(candidates))])
    n = layer_sizes[layer_idx]
    remove = min(n - 20, max(1, int(n * 0.15)))
    keep_pos = np.ones(n, dtype=bool)
    remove_pos = rng.choice(n, size=remove, replace=False)
    keep_pos[remove_pos] = False

    mats[layer_idx] = mats[layer_idx][:, keep_pos]
    biases[layer_idx] = biases[layer_idx][keep_pos]
    if layer_idx + 1 < len(mats):
        mats[layer_idx + 1] = mats[layer_idx + 1][keep_pos, :]
    else:
        out_mat = out_mat[keep_pos, :]
    layer_sizes[layer_idx] = n - remove
    return True, out_mat, out_bias


def _insert_random_layer(
    layer_sizes: List[int],
    mats: List[np.ndarray],
    biases: List[np.ndarray],
    out_mat: np.ndarray,
    out_bias: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    if len(layer_sizes) >= MAX_DEPTH:
        return False, out_mat, out_bias
    max_add = min(MAX_HIDDEN_SIZE, MAX_TOTAL_HIDDEN - sum(layer_sizes))
    if max_add < 20:
        return False, out_mat, out_bias
    new_n = int(rng.integers(20, max_add + 1))
    pos = int(rng.integers(0, len(layer_sizes) + 1))

    if pos == len(layer_sizes):
        prev_n = layer_sizes[-1] if layer_sizes else INPUT_SIZE
        new_in = rng.normal(
            0.0,
            HIDDEN_W_SCALE / math.sqrt(max(prev_n, 1)),
            size=(prev_n, new_n),
        ).astype(np.float32)
        new_bias = rng.normal(0.0, HIDDEN_BIAS_SCALE, size=new_n).astype(np.float32)
        mats.append(new_in)
        biases.append(new_bias)
        out_mat = rng.normal(
            0.0,
            OUTPUT_W_SCALE / math.sqrt(max(new_n, 1)),
            size=(new_n, 10),
        ).astype(np.float32)
        out_bias = rng.normal(0.0, OUTPUT_BIAS_SCALE, size=10).astype(np.float32)
        layer_sizes.append(new_n)
        return True, out_mat, out_bias

    old_in = mats[pos]
    old_n = layer_sizes[pos]
    prev_n = INPUT_SIZE if pos == 0 else layer_sizes[pos - 1]
    new_in = rng.normal(
        0.0,
        INPUT_W_SCALE if pos == 0 else HIDDEN_W_SCALE / math.sqrt(max(prev_n, 1)),
        size=(prev_n, new_n),
    ).astype(np.float32)
    new_to_old = rng.normal(
        0.0,
        HIDDEN_W_SCALE / math.sqrt(max(new_n, 1)),
        size=(new_n, old_n),
    ).astype(np.float32)
    new_bias = rng.normal(0.0, HIDDEN_BIAS_SCALE, size=new_n).astype(np.float32)
    mats[pos] = new_in
    mats.insert(pos + 1, new_to_old)
    biases.insert(pos + 1, new_bias)
    layer_sizes.insert(pos, new_n)
    return True, out_mat, out_bias


def _mutate_readout(
    out_mat: np.ndarray,
    out_bias: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float,
) -> None:
    if rng.random() < P_READOUT_MUTATION * mutation_rate:
        out_mat += rng.normal(0.0, 0.02, size=out_mat.shape).astype(
            np.float32
        )
        out_bias += rng.normal(0.0, 0.02, size=out_bias.shape).astype(
            np.float32
        )
    if rng.random() < P_READOUT_SPARSE_RESET * mutation_rate:
        mask = rng.random(out_mat.shape) < 0.05
        n_reset = int(mask.sum())
        if n_reset > 0:
            out_mat[mask] = rng.normal(
                0.0, 0.05, size=n_reset
            ).astype(np.float32)


def _mutate_genome(genome: Genome, rng: np.random.Generator) -> Genome:
    mutation_rate = max(0.25, min(3.0, genome.mutation_rate))
    rolls = [
        rng.random() < min(1.0, P_GROW * mutation_rate),
        rng.random() < min(1.0, P_SPLIT * mutation_rate),
        rng.random() < min(1.0, P_MERGE * mutation_rate),
        rng.random() < min(1.0, P_PRUNE * mutation_rate),
        rng.random() < min(1.0, P_ADDRANDOM * mutation_rate),
    ]
    readout_mutation = rng.random() < min(
        1.0, (P_READOUT_MUTATION + P_READOUT_SPARSE_RESET) * mutation_rate
    )
    trait_mutation = rng.random() < TRAIT_MUTATION_RATE

    longevity_bonus = int(genome.longevity_bonus)
    fecundity = float(genome.fecundity)
    wrong_tolerance = float(genome.wrong_tolerance)
    next_mutation_rate = float(genome.mutation_rate)
    if trait_mutation:
        longevity_bonus = max(
            0, min(10, longevity_bonus + int(rng.integers(-1, 2)))
        )
        fecundity = max(
            0.5,
            min(2.0, fecundity * math.exp(rng.normal(0.0, 0.1))),
        )
        wrong_tolerance = max(
            0.5,
            min(3.0, wrong_tolerance * math.exp(rng.normal(0.0, 0.1))),
        )
        next_mutation_rate = max(
            0.5,
            min(3.0, next_mutation_rate * math.exp(rng.normal(0.0, 0.15))),
        )

    if not any(rolls) and not readout_mutation:
        return Genome(
            tuple(genome.layer_sizes),
            genome.weights,
            longevity_bonus,
            fecundity,
            wrong_tolerance,
            next_mutation_rate,
        )

    layer_sizes = list(genome.layer_sizes)
    mats, biases, out_mat, out_bias = _unflatten(
        genome.weights.copy(), layer_sizes
    )

    if rolls[0]:
        _, out_mat, out_bias = _grow_silent(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rolls[1]:
        _, out_mat, out_bias = _split_identity(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rolls[2]:
        _, out_mat, out_bias = _merge_layers(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rolls[3]:
        _, out_mat, out_bias = _prune_neurons(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rolls[4]:
        _, out_mat, out_bias = _insert_random_layer(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if readout_mutation:
        _mutate_readout(out_mat, out_bias, rng, mutation_rate)

    return Genome(
        tuple(layer_sizes),
        _flatten(mats, biases, out_mat, out_bias),
        longevity_bonus,
        fecundity,
        wrong_tolerance,
        next_mutation_rate,
    )


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    age_a: float,
    age_b: float,
    rng: Optional[np.random.Generator] = None,
) -> Genome:
    if rng is None:
        rng = np.random.default_rng()

    score_a = max(age_a + 0.1, 0.1)
    score_b = max(age_b + 0.1, 0.1)
    if rng.random() < score_a / (score_a + score_b):
        donor = parent_a
        other = parent_b
    else:
        donor = parent_b
        other = parent_a

    layer_sizes = list(donor.layer_sizes)
    flat = donor.weights.copy()
    if tuple(layer_sizes) == other.layer_sizes:
        if rng.random() < 0.5:
            readout_offset = _output_offset(layer_sizes)
            flat[readout_offset:] = other.weights[readout_offset:]
        else:
            start = int(rng.integers(0, 2))
            flat[start::2] = other.weights[start::2]
    n_jitter = max(1, int(flat.size * 0.02))
    jitter_pos = rng.integers(0, flat.size, size=n_jitter)
    flat[jitter_pos] += rng.uniform(-0.03, 0.03, size=n_jitter).astype(
        np.float32
    )
    n_big = max(1, int(flat.size * BIG_MUTATION_RATE))
    big_pos = rng.integers(0, flat.size, size=n_big)
    flat[big_pos] += rng.uniform(-0.5, 0.5, size=n_big).astype(np.float32)

    return _mutate_genome(
        Genome(
            tuple(layer_sizes),
            flat,
            int(donor.longevity_bonus),
            float((donor.fecundity + other.fecundity) / 2.0),
            float((donor.wrong_tolerance + other.wrong_tolerance) / 2.0),
            float((donor.mutation_rate + other.mutation_rate) / 2.0),
        ),
        rng,
    )


@njit(cache=True)
def _forward_core_multi(
    weights: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    spike_idx: np.ndarray,
    spike_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_pop = weights.shape[0]
    n_steps = spike_idx.shape[0]
    n_samples = spike_idx.shape[1]
    hidden_counts = np.zeros((n_pop, MAX_HIDDEN_SIZE), dtype=np.float32)
    logits = np.zeros((n_pop, 10), dtype=np.float32)

    for i in range(n_pop):
        depth = 0
        for d in range(MAX_DEPTH):
            if sizes[i, d] > 0:
                depth += 1

        mem = np.zeros((MAX_DEPTH, MAX_HIDDEN_SIZE), dtype=np.float32)
        spk = np.zeros((MAX_DEPTH, MAX_HIDDEN_SIZE), dtype=np.float32)

        for t in range(n_steps):
            for l in range(depth):
                n = sizes[i, l]
                for j in range(n):
                    spk[l, j] = 0.0

            for l in range(depth):
                n = sizes[i, l]
                start = offsets[i, l]
                if l == 0:
                    for k in range(n_samples):
                        idx = spike_idx[t, k]
                        val = spike_vals[t, k]
                        if val != 0.0:
                            base = start + idx * n
                            for j in range(n):
                                mem[l, j] += weights[i, base + j] * val
                else:
                    prev_n = sizes[i, l - 1]
                    base = start
                    for j in range(n):
                        acc = 0.0
                        for p in range(prev_n):
                            if spk[l - 1, p] != 0.0:
                                acc += (
                                    weights[i, base + p * n + j]
                                    * spk[l - 1, p]
                                )
                        mem[l, j] += acc

                prev_n = INPUT_SIZE if l == 0 else sizes[i, l - 1]
                bias_start = start + prev_n * n
                for j in range(n):
                    mem[l, j] = LEAK * mem[l, j] + weights[i, bias_start + j]

                best = -1
                best_val = -1.0
                for j in range(n):
                    if mem[l, j] >= THETA_HIDDEN and mem[l, j] > best_val:
                        best = j
                        best_val = mem[l, j]
                if best >= 0:
                    spk[l, best] = 1.0
                    mem[l, best] = 0.0

            last_n = sizes[i, depth - 1]
            for p in range(last_n):
                if spk[depth - 1, p] != 0.0:
                    hidden_counts[i, p] += 1.0

        last_n = sizes[i, depth - 1]
        start = offsets[i, 4]
        for o in range(10):
            acc = weights[i, start + last_n * 10 + o]
            for p in range(last_n):
                acc += (
                    hidden_counts[i, p] / n_steps
                ) * weights[i, start + p * 10 + o]
            logits[i, o] = acc

    return logits, hidden_counts


def _pack_batch(
    genomes: Sequence[Genome],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(genomes)
    weights = np.zeros((n, MAX_WEIGHTS), dtype=np.float32)
    sizes = np.zeros((n, MAX_DEPTH), dtype=np.int32)
    offsets = np.zeros((n, 5), dtype=np.int32)
    lengths = np.zeros((n, 5), dtype=np.int32)

    for i, genome in enumerate(genomes):
        offset = 0
        prev = INPUT_SIZE
        flat = genome.weights
        for layer_idx, n_hidden in enumerate(genome.layer_sizes):
            length = prev * n_hidden + n_hidden
            weights[i, offset : offset + length] = flat[
                offset : offset + length
            ]
            sizes[i, layer_idx] = n_hidden
            offsets[i, layer_idx] = offset
            lengths[i, layer_idx] = length
            offset += length
            prev = n_hidden
        length = prev * 10 + 10
        weights[i, offset : offset + length] = flat[offset : offset + length]
        offsets[i, 4] = offset
        lengths[i, 4] = length

    return weights, sizes, offsets, lengths


def _stack_genome_group(
    genomes: Sequence[Genome],
    layer_sizes: Sequence[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    n_pop = len(genomes)
    mats: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    offset = 0
    prev = INPUT_SIZE

    for n_hidden in layer_sizes:
        count = prev * n_hidden + n_hidden
        mat = np.empty((n_pop, prev, n_hidden), dtype=np.float32)
        bias = np.empty((n_pop, n_hidden), dtype=np.float32)
        for i, genome in enumerate(genomes):
            mat[i] = genome.weights[
                offset : offset + prev * n_hidden
            ].reshape(prev, n_hidden)
            bias[i] = genome.weights[
                offset + prev * n_hidden : offset + count
            ]
        mats.append(mat)
        biases.append(bias)
        offset += count
        prev = n_hidden

    count = prev * 10 + 10
    out_mat = np.empty((n_pop, prev, 10), dtype=np.float32)
    out_bias = np.empty((n_pop, 10), dtype=np.float32)
    for i, genome in enumerate(genomes):
        out_mat[i] = genome.weights[offset : offset + prev * 10].reshape(prev, 10)
        out_bias[i] = genome.weights[offset + prev * 10 : offset + count]
    return mats, biases, out_mat, out_bias


def _vectorized_wta(mem: np.ndarray) -> np.ndarray:
    fire = mem >= THETA_HIDDEN
    best = np.argmax(np.where(fire, mem, -1e9), axis=1)
    spk = np.zeros_like(mem)
    rows = np.flatnonzero(fire.any(axis=1))
    spk[rows, best[rows]] = 1.0
    mem[rows, best[rows]] = 0.0
    return spk


def _forward_group_vectorized(
    mats: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    out_mat: np.ndarray,
    out_bias: np.ndarray,
    spike_idx: np.ndarray,
    spike_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_pop = mats[0].shape[0]
    n_steps = spike_idx.shape[0]
    depth = len(mats)
    mem = [
        np.zeros((n_pop, mats[layer_idx].shape[2]), dtype=np.float32)
        for layer_idx in range(depth)
    ]
    spk: List[np.ndarray] = [None] * depth
    last_n = mats[-1].shape[2]
    hidden_counts = np.zeros((n_pop, last_n), dtype=np.float32)

    for t in range(n_steps):
        selected = np.take(mats[0], spike_idx[t], axis=1)
        acc = np.einsum("k,nkj->nj", spike_vals[t], selected)
        mem[0] = mem[0] * LEAK + acc + biases[0]
        spk[0] = _vectorized_wta(mem[0])

        for layer_idx in range(1, depth):
            acc = np.einsum(
                "np,npj->nj", spk[layer_idx - 1], mats[layer_idx]
            )
            mem[layer_idx] = (
                mem[layer_idx] * LEAK + acc + biases[layer_idx]
            )
            spk[layer_idx] = _vectorized_wta(mem[layer_idx])

        hidden_counts += spk[-1]

    hidden_rates = hidden_counts / float(n_steps)
    logits = np.einsum("np,npo->no", hidden_rates, out_mat) + out_bias
    return logits, hidden_rates


def _forward_numba_batch(
    genomes: Sequence[Genome],
    spike_idx: np.ndarray,
    spike_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    weights, sizes, offsets, lengths = _pack_batch(genomes)
    logits, hidden_counts = _forward_core_multi(
        weights, sizes, offsets, lengths, spike_idx, spike_vals
    )
    return logits, hidden_counts


def forward(
    genomes: Sequence[Genome],
    spikes: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    spike_idx, spike_vals = spikes
    if not genomes:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0, 10), dtype=np.float32),
        )

    groups: Dict[Tuple[int, ...], List[int]] = {}
    for idx, genome in enumerate(genomes):
        groups.setdefault(tuple(genome.layer_sizes), []).append(idx)

    predictions = np.empty(len(genomes), dtype=np.int32)
    rates = np.empty((len(genomes), 10), dtype=np.float32)
    for layer_sizes, indices in groups.items():
        selected = [genomes[i] for i in indices]
        if len(selected) >= 8:
            mats, biases, out_mat, out_bias = _stack_genome_group(
                selected, layer_sizes
            )
            logits, _ = _forward_group_vectorized(
                mats,
                biases,
                out_mat,
                out_bias,
                spike_idx,
                spike_vals,
            )
            predictions[indices] = logits.argmax(axis=1).astype(np.int32)
            rates[indices] = _softmax(logits)
        else:
            logits, _ = _forward_numba_batch(selected, spike_idx, spike_vals)
            predictions[indices] = logits.argmax(axis=1).astype(np.int32)
            rates[indices] = _softmax(logits)

    return predictions, rates


def extract_hidden_features(
    genomes: Sequence[Genome],
    spikes: Tuple[np.ndarray, np.ndarray],
) -> List[np.ndarray]:
    spike_idx, spike_vals = spikes
    if not genomes:
        return []

    groups: Dict[Tuple[int, ...], List[int]] = {}
    for idx, genome in enumerate(genomes):
        groups.setdefault(tuple(genome.layer_sizes), []).append(idx)

    features: List[Optional[np.ndarray]] = [None] * len(genomes)
    for layer_sizes, indices in groups.items():
        selected = [genomes[i] for i in indices]
        if len(selected) >= 8:
            mats, biases, out_mat, out_bias = _stack_genome_group(
                selected, layer_sizes
            )
            _, hidden_rates = _forward_group_vectorized(
                mats,
                biases,
                out_mat,
                out_bias,
                spike_idx,
                spike_vals,
            )
            for j, idx in enumerate(indices):
                features[idx] = hidden_rates[j].astype(np.float32)
        else:
            _, hidden_counts = _forward_numba_batch(
                selected, spike_idx, spike_vals
            )
            n_steps = spike_idx.shape[0]
            last_n = layer_sizes[-1]
            for j, idx in enumerate(indices):
                features[idx] = (
                    hidden_counts[j, :last_n] / float(n_steps)
                ).astype(np.float32)

    return [feature for feature in features if feature is not None]


@dataclass
class EcoConfig:
    survival_rounds: int = SURVIVAL_ROUNDS
    n_repro: int = N_REPRO
    assort_strength: float = ASSORT_STRENGTH
    capacity: int = CAPACITY
    init_pop: int = INIT_POP
    density_floor: float = DENSITY_FLOOR
    pop_growth: float = POP_GROWTH
    feed_interval: float = 5.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "survival_rounds": self.survival_rounds,
            "n_repro": self.n_repro,
            "assort_strength": self.assort_strength,
            "capacity": self.capacity,
            "init_pop": self.init_pop,
            "density_floor": self.density_floor,
            "pop_growth": self.pop_growth,
            "feed_interval": self.feed_interval,
        }

    def update(self, values: Dict[str, object]) -> None:
        if "survival_rounds" in values:
            self.survival_rounds = int(
                min(30, max(10, int(values["survival_rounds"])))
            )
        if "n_repro" in values:
            self.n_repro = int(min(100, max(10, int(values["n_repro"]))))
        if "assort_strength" in values:
            self.assort_strength = float(
                min(1.0, max(0.0, float(values["assort_strength"])))
            )
        if "capacity" in values:
            self.capacity = int(min(10000, max(100, int(values["capacity"]))))
        if "init_pop" in values:
            self.init_pop = int(min(1000, max(60, int(values["init_pop"]))))
        if "density_floor" in values:
            self.density_floor = float(
                min(1.0, max(0.01, float(values["density_floor"])))
            )
        if "pop_growth" in values:
            self.pop_growth = float(
                min(0.5, max(0.0, float(values["pop_growth"])))
            )
        if "feed_interval" in values:
            self.feed_interval = float(
                min(10.0, max(1.0, float(values["feed_interval"])))
            )


@dataclass
class Organism:
    uid: int
    genome: Genome
    age: int = 0
    alive: bool = True
    correct: bool = False
    prediction: int = -1
    born_round: int = 0
    died_round: Optional[int] = None
    death_reason: str = ""
    last_digit: int = -1
    digit_preference: int = -1
    failed_repro_rounds: int = 0

    def to_dict(self, include_weights: bool = False) -> Dict[str, object]:
        data = {
            "id": self.uid,
            "age": self.age,
            "alive": self.alive,
            "correct": self.correct,
            "prediction": self.prediction,
            "born_round": self.born_round,
            "died_round": self.died_round,
            "death_reason": self.death_reason,
            "last_digit": self.last_digit,
            "digit_preference": self.digit_preference,
            "failed_repro_rounds": self.failed_repro_rounds,
            "layer_sizes": list(self.genome.layer_sizes),
            "weights_size": int(self.genome.weights.size),
            "longevity_bonus": self.genome.longevity_bonus,
            "fecundity": self.genome.fecundity,
            "wrong_tolerance": self.genome.wrong_tolerance,
            "mutation_rate": self.genome.mutation_rate,
        }
        if include_weights:
            data["weights"] = self.genome.weights.tolist()
        return data


class Ecosystem:
    def __init__(
        self,
        config: Optional[EcoConfig] = None,
        seed: int = 1,
    ) -> None:
        self.config = config or EcoConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._train_images, self._train_labels, self._test_images, self._test_labels = (
            load_mnist()
        )
        self._warmup_numba()
        self._founder_specs: List[Tuple[Genome, int]] = []
        self._founder_per_digit = 0
        self.reset()

    def _warmup_numba(self) -> None:
        warm_rng = np.random.default_rng(0)
        dummy = random_genome(warm_rng)
        spike_idx, spike_vals = _sample_spikes(
            np.zeros(INPUT_SIZE, dtype=np.float32),
            warm_rng,
        )
        _forward_numba_batch([dummy], spike_idx, spike_vals)

    def reset(self) -> None:
        self.round = 0
        self._next_uid = 0
        self.population: List[Organism] = []
        self.cumulative_natural_deaths = 0
        self.cumulative_total_deaths = 0
        self.resets = 0
        self.stopped = False
        self.history: Dict[str, List[object]] = {
            "round": [],
            "alive": [],
            "accuracy": [],
            "wrong_deaths": [],
            "natural_deaths": [],
            "mean_age": [],
            "natural_rate": [],
        }
        self._seed_population(self.config.init_pop)
        self.current_food = self._pick_food()
        self.last_events: Optional[Dict[str, object]] = None

    def _seed_population(self, size: int) -> None:
        per_digit = (
            SELECT_PER_DIGIT if size == INIT_POP else max(1, size // 10)
        )
        target = per_digit * 10
        if (
            len(self._founder_specs) != target
            or self._founder_per_digit != per_digit
        ):
            self._founder_specs = self._select_founding_specs(per_digit)
            self._founder_per_digit = per_digit

        for genome, digit in self._founder_specs:
            organism = Organism(
                uid=self._next_uid,
                genome=Genome(
                    tuple(genome.layer_sizes),
                    genome.weights.copy(),
                    genome.longevity_bonus,
                    genome.fecundity,
                    genome.wrong_tolerance,
                    genome.mutation_rate,
                ),
                born_round=self.round,
                correct=True,
                prediction=digit,
                last_digit=digit,
                digit_preference=digit,
            )
            self._next_uid += 1
            self.population.append(organism)

    def _fitness_from_accuracy(self, accuracy: np.ndarray) -> np.ndarray:
        mean_accuracy = accuracy.mean(axis=1)
        worst_digit = accuracy.min(axis=1)
        return mean_accuracy + FITNESS_LAMBDA * worst_digit

    def _score_digits(
        self,
        genomes: Sequence[Genome],
        spikes_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        accuracy = np.zeros((len(genomes), 10), dtype=np.float32)
        for digit, spikes in spikes_cache.items():
            predictions, _ = forward(genomes, spikes)
            accuracy[:, digit] = (predictions == digit).astype(np.float32)
        return accuracy

    def _fit_readouts(self, genomes: Sequence[Genome]) -> None:
        if not genomes:
            return
        sample_indices: List[int] = []
        sample_labels: List[int] = []
        for digit in range(10):
            candidates = np.flatnonzero(self._train_labels == digit)
            chosen = self.rng.choice(
                candidates,
                size=min(SCREEN_READOUT_SAMPLES_PER_DIGIT, len(candidates)),
                replace=False,
            )
            sample_indices.extend(int(i) for i in chosen)
            sample_labels.extend([digit] * len(chosen))

        feature_sets: List[List[np.ndarray]] = [
            [] for _ in range(len(genomes))
        ]
        for sample_index in sample_indices:
            image = self._train_images[sample_index]
            spikes = _sample_spikes(image, self.rng)
            features = extract_hidden_features(genomes, spikes)
            for i, feature in enumerate(features):
                feature_sets[i].append(feature)

        n_samples = len(sample_indices)
        targets = np.zeros((n_samples, 10), dtype=np.float32)
        targets[np.arange(n_samples), sample_labels] = 1.0

        for i, genome in enumerate(genomes):
            features = np.stack(feature_sets[i]).astype(np.float64)
            augmented = np.column_stack(
                [features, np.ones(n_samples, dtype=np.float64)]
            )
            dim = augmented.shape[1]
            gram = augmented.T @ augmented + READOUT_LAMBDA * np.eye(dim)
            readout = np.linalg.pinv(gram) @ augmented.T @ targets.astype(
                np.float64
            )
            offset = _output_offset(genome.layer_sizes)
            last_n = genome.layer_sizes[-1]
            genome.weights[offset : offset + last_n * 10] = (
                readout[:-1].ravel().astype(np.float32)
            )
            genome.weights[offset + last_n * 10 :] = readout[-1].astype(
                np.float32
            )

    def _select_founding_specs(
        self,
        per_digit: int,
    ) -> List[Tuple[Genome, int]]:
        target = per_digit * 10
        pool_size = max(SCREEN_POOL_SIZE, target * 2)
        foods = [self._pick_food(digit) for digit in range(10)]
        cheap_spikes = {
            digit: _sample_spikes(
                np.asarray(food["image"], dtype=np.float32),
                self.rng,
                steps=SCREEN_STEPS,
                samples_per_step=SCREEN_SAMPLES,
            )
            for digit, food in enumerate(foods)
        }
        full_spikes = {
            digit: _sample_spikes(
                np.asarray(food["image"], dtype=np.float32),
                self.rng,
                steps=T,
                samples_per_step=INPUT_SAMPLES_PER_STEP,
            )
            for digit, food in enumerate(foods)
        }

        genomes: List[Genome] = [
            random_genome(self.rng) for _ in range(pool_size)
        ]
        accuracy = self._score_digits(genomes, cheap_spikes)
        fitness = self._fitness_from_accuracy(accuracy)
        order = np.argsort(-fitness)
        used = np.zeros(len(genomes), dtype=bool)
        selected: List[Tuple[Genome, int]] = []
        digit_counts = [0] * 10

        for digit in range(10):
            candidate_indices = [
                int(idx)
                for idx in order
                if not used[int(idx)] and accuracy[int(idx), digit] > 0
            ]
            if candidate_indices:
                full_predictions, _ = forward(
                    [genomes[idx] for idx in candidate_indices],
                    full_spikes[digit],
                )
                for rank, idx in enumerate(candidate_indices):
                    if digit_counts[digit] >= per_digit:
                        break
                    if full_predictions[rank] == digit:
                        selected.append((genomes[idx], digit))
                        used[idx] = True
                        digit_counts[digit] += 1

        for digit in range(10):
            if digit_counts[digit] >= per_digit:
                continue
            remaining = [int(idx) for idx in order if not used[int(idx)]]
            if not remaining:
                break
            full_predictions, _ = forward(
                [genomes[idx] for idx in remaining],
                full_spikes[digit],
            )
            for rank, idx in enumerate(remaining):
                if digit_counts[digit] >= per_digit:
                    break
                if full_predictions[rank] == digit:
                    selected.append((genomes[idx], digit))
                    used[idx] = True
                    digit_counts[digit] += 1

        self._fit_readouts([genome for genome, _ in selected])
        return selected

    def _pick_food(self, digit: Optional[int] = None) -> Dict[str, object]:
        if digit is None:
            index = int(self.rng.integers(len(self._test_images)))
        else:
            label = int(digit) % 10
            candidates = np.flatnonzero(self._test_labels == label)
            if len(candidates) == 0:
                index = int(self.rng.integers(len(self._test_images)))
            else:
                index = int(candidates[self.rng.integers(len(candidates))])
        label = int(self._test_labels[index])
        return {
            "index": index,
            "label": label,
            "image": self._test_images[index].tolist(),
        }

    def _kill(self, organism: Organism, reason: str) -> None:
        organism.alive = False
        organism.died_round = self.round
        organism.death_reason = reason

    def _survival_histogram(self, organisms: Sequence[Organism]) -> List[int]:
        bins = [0] * (self.config.survival_rounds + 1)
        for organism in organisms:
            bins[min(organism.age, self.config.survival_rounds)] += 1
        return bins

    def _depth_histogram(self, organisms: Sequence[Organism]) -> List[int]:
        bins = [0, 0, 0, 0]
        for organism in organisms:
            depth = min(max(len(organism.genome.layer_sizes), 1), 4)
            bins[depth - 1] += 1
        return bins

    def _assortative_pairs(
        self,
        organisms: Sequence[Organism],
        strength: float,
        rng: np.random.Generator,
    ) -> List[Tuple[Organism, Organism]]:
        remaining = list(organisms)
        rng.shuffle(remaining)
        pairs: List[Tuple[Organism, Organism]] = []
        max_age = max(1.0, float(max((o.age for o in remaining), default=1)))

        while len(remaining) >= 2:
            first = remaining.pop(0)
            if not remaining:
                break
            scores = []
            for second in remaining:
                age_sim = 1.0 - abs(first.age - second.age) / max_age
                digit_bonus = (
                    0.0
                    if first.digit_preference == second.digit_preference
                    else 1.0
                )
                mate_score = 0.5 * age_sim + 0.5 * digit_bonus
                score = (1.0 - strength) * rng.random() + strength * mate_score
                scores.append(max(score, 1e-6))
            scores = np.asarray(scores, dtype=np.float64)
            probs = scores / scores.sum()
            chosen = int(rng.choice(len(remaining), p=probs))
            second = remaining.pop(chosen)
            pairs.append((first, second))

        if len(remaining) == 1 and not pairs:
            pairs.append((remaining[0], remaining[0]))
        return pairs

    def _pair_repro_success(self, first: Organism, second: Organism) -> float:
        probability = REPRO_SUCCESS_BASE
        if not first.correct:
            tolerance = max(0.1, first.genome.wrong_tolerance)
            probability *= REPRO_WRONG_PENALTY ** (1.0 / tolerance)
        if not second.correct:
            tolerance = max(0.1, second.genome.wrong_tolerance)
            probability *= REPRO_WRONG_PENALTY ** (1.0 / tolerance)
        return probability

    def _offspring_counts(
        self,
        pairs: Sequence[Tuple[Organism, Organism]],
        needed: int,
        alpha: float,
        density_factor: float,
    ) -> List[int]:
        if not pairs:
            return []
        raw_weights = []
        for first, second in pairs:
            age_sum = first.age + second.age + 1
            fecundity = (
                first.genome.fecundity * second.genome.fecundity
            )
            raw_weights.append(
                max(
                    1,
                    int(
                        age_sum
                        * self.config.n_repro
                        * alpha
                        * density_factor
                        * fecundity
                    ),
                )
            )
        weights = np.asarray(raw_weights, dtype=np.float64)
        total = float(weights.sum())

        if total <= needed:
            counts = weights.astype(np.int64).tolist()
            extra = needed - int(weights.sum())
            if extra > 0:
                base = extra // len(pairs)
                remainder = extra % len(pairs)
                counts = [c + base for c in counts]
                for i in range(remainder):
                    counts[i] += 1
            return [int(c) for c in counts]

        exact = needed * weights / total
        counts = np.floor(exact).astype(np.int64)
        remainder = needed - int(counts.sum())
        order = np.argsort(-(exact - counts))
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
        return [int(c) for c in counts]

    def _record_history(self, events: Dict[str, object]) -> None:
        self.history["round"].append(events.get("round", self.round))
        self.history["alive"].append(events.get("population_after", 0))
        self.history["accuracy"].append(events.get("accuracy", 0.0))
        self.history["wrong_deaths"].append(events.get("wrong_deaths", 0))
        self.history["natural_deaths"].append(events.get("natural_deaths", 0))
        self.history["mean_age"].append(events.get("mean_age", 0.0))
        self.history["natural_rate"].append(
            self.cumulative_natural_deaths
            / max(1, self.cumulative_total_deaths)
        )
        for key in self.history:
            if len(self.history[key]) > 500:
                self.history[key] = self.history[key][-500:]

    def step(self, digit: Optional[int] = None) -> Dict[str, object]:
        if self.stopped:
            return {
                "round": self.round,
                "stopped": True,
                "message": "stopped",
            }

        self.round += 1
        self.current_food = self._pick_food(digit)
        label = int(self.current_food["label"])
        alive_before = [o for o in self.population if o.alive]

        events: Dict[str, object] = {
            "round": self.round,
            "digit": label,
            "food_index": int(self.current_food["index"]),
            "alive_before": len(alive_before),
            "wrong_deaths": 0,
            "natural_deaths": 0,
            "no_repro_deaths": 0,
            "deaths": [],
            "births": [],
            "replay": False,
            "stopped": False,
        }

        if not alive_before:
            self._seed_population(self.config.init_pop)
            self.resets += 1
            events["replay"] = True
            events["digit_label"] = label
            events["population_after"] = sum(1 for o in self.population if o.alive)
            events["accuracy"] = 0.0
            events["mean_age"] = 0.0
            events["survivors"] = []
            self._record_history(events)
            self.last_events = events
            return events

        spikes = _sample_spikes(np.asarray(self.current_food["image"], np.float32), self.rng)
        predictions, _ = forward([o.genome for o in alive_before], spikes)

        for idx, organism in enumerate(alive_before):
            organism.correct = bool(predictions[idx] == label)
            organism.prediction = int(predictions[idx])
            organism.last_digit = label
            organism.age += 1
            lifespan = self.config.survival_rounds + organism.genome.longevity_bonus
            if organism.age >= lifespan:
                self._kill(organism, "natural")

        survivors = [o for o in alive_before if o.alive]
        natural_deaths = [
            o
            for o in alive_before
            if not o.alive and o.death_reason == "natural"
        ]

        survival_rate = len(survivors) / max(1, len(alive_before))
        alpha = 1.0 / survival_rate if survival_rate > 0.0 else 1.0
        density_factor = max(
            self.config.density_floor,
            1.0 - len(survivors) / max(1, self.config.capacity),
        )
        mean_age = (
            float(np.mean([o.age for o in survivors])) if survivors else 0.0
        )
        age_factor = 1.0 + mean_age / max(1, self.config.survival_rounds)
        growth = int(
            len(survivors)
            * self.config.n_repro
            * self.config.pop_growth
            * alpha
            * age_factor
            * density_factor
            / REPRO_GROWTH_DIVISOR
        )
        needed = max(len(alive_before) - len(survivors), growth)
        target_pop = min(self.config.capacity, len(survivors) + needed)

        offspring = 0
        no_repro_deaths: List[Organism] = []
        if survivors and target_pop > len(survivors):
            pairs = self._assortative_pairs(
                survivors, self.config.assort_strength, self.rng
            )
            paired_ids = {organism.uid for pair in pairs for organism in pair}
            active_pairs = []
            for first, second in pairs:
                if self.rng.random() < self._pair_repro_success(first, second):
                    active_pairs.append((first, second))
                else:
                    first.failed_repro_rounds += 1
                    if first is not second:
                        second.failed_repro_rounds += 1

            counts = self._offspring_counts(
                active_pairs,
                target_pop - len(survivors),
                alpha,
                density_factor,
            )
            for (first, second), count in zip(active_pairs, counts):
                if count > 0:
                    first.failed_repro_rounds = 0
                    if first is not second:
                        second.failed_repro_rounds = 0
                else:
                    first.failed_repro_rounds += 1
                    if first is not second:
                        second.failed_repro_rounds += 1
                for _ in range(count):
                    child = Organism(
                        uid=self._next_uid,
                        genome=crossover(
                            first.genome,
                            second.genome,
                            first.age,
                            second.age,
                            self.rng,
                        ),
                        born_round=self.round,
                        digit_preference=(
                            first.digit_preference
                            if self.rng.random() < 0.5
                            else second.digit_preference
                        ),
                    )
                    self._next_uid += 1
                    self.population.append(child)
                    events["births"].append({"id": child.uid})  # type: ignore[attr-defined]
                    offspring += 1

            for organism in survivors:
                if organism.uid not in paired_ids:
                    organism.failed_repro_rounds += 1

        for organism in survivors:
            if organism.failed_repro_rounds >= NO_REPRO_DEATH_ROUNDS:
                self._kill(organism, "no_repro")
                no_repro_deaths.append(organism)

        survivors_alive = [o for o in survivors if o.alive]

        self.cumulative_total_deaths += len(natural_deaths) + len(
            no_repro_deaths
        )
        self.cumulative_natural_deaths += len(natural_deaths)
        if (
            self.cumulative_total_deaths > 0
            and self.cumulative_natural_deaths / self.cumulative_total_deaths >= 0.95
        ):
            self.stopped = True

        events.update(
            {
                "digit_label": label,
                "alive_after": len(survivors_alive),
                "population_after": len(survivors_alive) + offspring,
                "survivors_before": len(alive_before),
                "survival_rate": survival_rate,
                "alpha": alpha,
                "density_factor": density_factor,
                "wrong_deaths": 0,
                "natural_deaths": len(natural_deaths),
                "no_repro_deaths": len(no_repro_deaths),
                "offspring": offspring,
                "accuracy": sum(1 for o in alive_before if o.correct)
                / max(1, len(alive_before)),
                "mean_age": mean_age,
                "deaths": [
                    {
                        "id": o.uid,
                        "age": o.age,
                        "reason": o.death_reason,
                        "correct": o.correct,
                        "prediction": o.prediction,
                        "layer_sizes": list(o.genome.layer_sizes),
                        "digit_preference": o.digit_preference,
                    }
                    for o in natural_deaths + no_repro_deaths
                ],
                "survivors": [
                    {
                        "id": o.uid,
                        "age": o.age,
                        "correct": o.correct,
                        "prediction": o.prediction,
                        "layer_sizes": list(o.genome.layer_sizes),
                        "born_round": o.born_round,
                        "digit_preference": o.digit_preference,
                        "failed_repro_rounds": o.failed_repro_rounds,
                    }
                    for o in survivors_alive
                ],
                "stopped": self.stopped,
            }
        )
        self._record_history(events)
        self.last_events = events
        return events

    def state(self) -> Dict[str, object]:
        alive = [o for o in self.population if o.alive]
        total_deaths = max(1, self.cumulative_total_deaths)
        return {
            "config": self.config.to_dict(),
            "round": self.round,
            "stopped": self.stopped,
            "resets": self.resets,
            "cumulative_natural_deaths": self.cumulative_natural_deaths,
            "cumulative_total_deaths": self.cumulative_total_deaths,
            "natural_rate": self.cumulative_natural_deaths / total_deaths,
            "population_size": len(alive),
            "current_food": self.current_food,
            "alive": [o.to_dict() for o in alive],
            "history": self.history,
            "survival_histogram": self._survival_histogram(alive),
            "depth_histogram": self._depth_histogram(alive),
            "last_events": self.last_events,
        }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="LIF ecology game engine")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = EcoConfig(init_pop=args.population)
    ecosystem = Ecosystem(config=config, seed=args.seed)
    outputs = []
    for _ in range(args.rounds):
        outputs.append(ecosystem.step())
    outputs.append(ecosystem.state())
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, cls=NumpyEncoder))
    else:
        for events in outputs:
            if "digit_label" in events:
                print(
                    f"round={events['round']} digit={events['digit_label']} "
                    f"alive={events['population_after']} "
                    f"wrong={events['wrong_deaths']} natural={events['natural_deaths']} "
                    f"no_repro={events.get('no_repro_deaths', 0)} "
                    f"accuracy={events['accuracy']:.3f}"
                )
            else:
                print(
                    f"final round={events['round']} pop={events['population_size']} "
                    f"resets={events['resets']}"
                )


if __name__ == "__main__":
    main()
