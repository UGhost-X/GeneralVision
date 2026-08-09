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
INPUT_SAMPLES_PER_STEP = 96
MAX_HIDDEN_SIZE = 200
MAX_DEPTH = 4
MAX_TOTAL_HIDDEN = 400
HIDDEN_SIZE = 100

LEAK = 0.94
THETA_HIDDEN = 12.0
LEAK_OUTPUT = 0.9
THETA_OUTPUT = 0.5

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

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_sizes": list(self.layer_sizes),
            "weights_size": int(self.weights.size),
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
    return Genome((HIDDEN_SIZE,), _random_weights((HIDDEN_SIZE,), rng))


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
        mat = flat[offset : offset + prev * n].reshape(prev, n).astype(np.float32)
        bias = flat[offset + prev * n : offset + count].astype(np.float32)
        hidden_mats.append(mat)
        hidden_biases.append(bias)
        offset += count
        prev = n
    count = prev * 10 + 10
    out_mat = flat[offset : offset + prev * 10].reshape(prev, 10).astype(np.float32)
    out_bias = flat[offset + prev * 10 : offset + count].astype(np.float32)
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
    return np.concatenate(parts).astype(np.float32)


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

    merged_w = (w1.astype(np.float64) @ w2.astype(np.float64)).astype(np.float32)
    merged_b = (
        w2.T.astype(np.float64) @ b1.astype(np.float64) + b2.astype(np.float64)
    ).astype(np.float32)
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


def _mutate_genome(genome: Genome, rng: np.random.Generator) -> Genome:
    layer_sizes = list(genome.layer_sizes)
    mats, biases, out_mat, out_bias = _unflatten(genome.weights, layer_sizes)

    if rng.random() < P_GROW:
        _, out_mat, out_bias = _grow_silent(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rng.random() < P_SPLIT:
        _, out_mat, out_bias = _split_identity(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rng.random() < P_MERGE:
        _, out_mat, out_bias = _merge_layers(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rng.random() < P_PRUNE:
        _, out_mat, out_bias = _prune_neurons(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )
    if rng.random() < P_ADDRANDOM:
        _, out_mat, out_bias = _insert_random_layer(
            layer_sizes, mats, biases, out_mat, out_bias, rng
        )

    return Genome(
        tuple(layer_sizes),
        _flatten(mats, biases, out_mat, out_bias),
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
    mats, biases, out_mat, out_bias = _unflatten(donor.weights, layer_sizes)
    other_mats, other_biases, other_out_mat, other_out_bias = _unflatten(
        other.weights, other.layer_sizes
    )

    shared_depth = min(len(layer_sizes), len(other.layer_sizes))
    for i in range(shared_depth):
        if layer_sizes[i] != other.layer_sizes[i]:
            continue
        n = layer_sizes[i]
        prev_n = INPUT_SIZE if i == 0 else layer_sizes[i - 1]
        mask = rng.random((prev_n, n)) < 0.5
        mats[i] = np.where(mask, mats[i], other_mats[i]).astype(np.float32)
        bias_mask = rng.random(n) < 0.5
        biases[i] = np.where(bias_mask, biases[i], other_biases[i]).astype(np.float32)

    if (
        len(layer_sizes) == len(other.layer_sizes)
        and layer_sizes[-1] == other.layer_sizes[-1]
    ):
        mask = rng.random((layer_sizes[-1], 10)) < 0.5
        out_mat = np.where(mask, out_mat, other_out_mat).astype(np.float32)
        bias_mask = rng.random(10) < 0.5
        out_bias = np.where(bias_mask, out_bias, other_out_bias).astype(np.float32)

    flat = _flatten(mats, biases, out_mat, out_bias)
    jitter = rng.normal(0.0, 0.02, size=flat.shape).astype(np.float32)
    flat += jitter
    big = rng.random(flat.shape) < BIG_MUTATION_RATE
    if int(big.sum()) > 0:
        flat[big] += rng.normal(0.0, 0.3, size=int(big.sum())).astype(np.float32)

    return _mutate_genome(Genome(tuple(layer_sizes), flat), rng)


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
    out_counts = np.zeros((n_pop, 10), dtype=np.int32)
    out_mem = np.zeros((n_pop, 10), dtype=np.float32)
    last_mem = np.zeros((n_pop, 10), dtype=np.float32)

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
            start = offsets[i, 4]
            for o in range(10):
                acc = 0.0
                for p in range(last_n):
                    if spk[depth - 1, p] != 0.0:
                        acc += weights[i, start + p * 10 + o] * spk[depth - 1, p]
                out_mem[i, o] = (
                    LEAK_OUTPUT * out_mem[i, o]
                    + weights[i, start + last_n * 10 + o]
                    + acc
                )
                if out_mem[i, o] >= THETA_OUTPUT:
                    out_mem[i, o] = 0.0
                    out_counts[i, o] += 1

        for o in range(10):
            last_mem[i, o] = out_mem[i, o]

    return out_counts, last_mem


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


def forward(
    genomes: Sequence[Genome],
    spikes: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    spike_idx, spike_vals = spikes
    predictions: List[int] = []
    rates: List[np.ndarray] = []

    for start in range(0, len(genomes), batch_size):
        batch = genomes[start : start + batch_size]
        weights, sizes, offsets, lengths = _pack_batch(batch)
        counts, mem = _forward_core_multi(
            weights, sizes, offsets, lengths, spike_idx, spike_vals
        )
        for i in range(len(batch)):
            if int(counts[i].max()) > 0:
                pred = int(np.argmax(counts[i]))
            else:
                pred = int(np.argmax(mem[i]))
            predictions.append(pred)
            rates.append(counts[i].astype(np.float32) / float(T))

    return np.asarray(predictions, dtype=np.int32), np.asarray(rates, dtype=np.float32)


def wrong_death_prob(survived_rounds: int) -> float:
    if survived_rounds <= 0:
        return 1.0
    if survived_rounds == 1:
        return 0.5
    return min(1.0, 0.5 * 1.5 ** (survived_rounds - 1))


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
            "layer_sizes": list(self.genome.layer_sizes),
            "weights_size": int(self.genome.weights.size),
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
        self.reset()

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
        for _ in range(size):
            organism = Organism(
                uid=self._next_uid,
                genome=random_genome(self.rng),
                born_round=self.round,
            )
            self._next_uid += 1
            self.population.append(organism)

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
                score = (1.0 - strength) * rng.random() + strength * age_sim
                scores.append(max(score, 1e-6))
            scores = np.asarray(scores, dtype=np.float64)
            probs = scores / scores.sum()
            chosen = int(rng.choice(len(remaining), p=probs))
            second = remaining.pop(chosen)
            pairs.append((first, second))

        if len(remaining) == 1 and not pairs:
            pairs.append((remaining[0], remaining[0]))
        return pairs

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
            raw_weights.append(
                max(1, int(age_sum * self.config.n_repro * alpha * density_factor))
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
            if not organism.correct:
                if self.rng.random() < wrong_death_prob(organism.age):
                    self._kill(organism, "wrong")
            if organism.alive:
                organism.age += 1
                if organism.age >= self.config.survival_rounds:
                    self._kill(organism, "natural")

        survivors = [o for o in alive_before if o.alive]
        wrong_deaths = [
            o
            for o in alive_before
            if not o.alive and o.death_reason == "wrong"
        ]
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
            * self.config.pop_growth
            * alpha
            * age_factor
            * density_factor
        )
        needed = max(len(alive_before) - len(survivors), growth)
        target_pop = min(self.config.capacity, len(survivors) + needed)

        offspring = 0
        if survivors and target_pop > len(survivors):
            pairs = self._assortative_pairs(
                survivors, self.config.assort_strength, self.rng
            )
            counts = self._offspring_counts(
                pairs,
                target_pop - len(survivors),
                alpha,
                density_factor,
            )
            for (first, second), count in zip(pairs, counts):
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
                    )
                    self._next_uid += 1
                    self.population.append(child)
                    events["births"].append({"id": child.uid})  # type: ignore[attr-defined]
                    offspring += 1

        self.cumulative_total_deaths += len(wrong_deaths) + len(natural_deaths)
        self.cumulative_natural_deaths += len(natural_deaths)
        if (
            self.cumulative_total_deaths > 0
            and self.cumulative_natural_deaths / self.cumulative_total_deaths >= 0.95
        ):
            self.stopped = True

        events.update(
            {
                "digit_label": label,
                "alive_after": len(survivors),
                "population_after": len(survivors) + offspring,
                "survivors_before": len(alive_before),
                "survival_rate": survival_rate,
                "alpha": alpha,
                "density_factor": density_factor,
                "wrong_deaths": len(wrong_deaths),
                "natural_deaths": len(natural_deaths),
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
                    }
                    for o in wrong_deaths + natural_deaths
                ],
                "survivors": [
                    {
                        "id": o.uid,
                        "age": o.age,
                        "correct": o.correct,
                        "prediction": o.prediction,
                        "layer_sizes": list(o.genome.layer_sizes),
                        "born_round": o.born_round,
                    }
                    for o in survivors
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
                    f"accuracy={events['accuracy']:.3f}"
                )
            else:
                print(
                    f"final round={events['round']} pop={events['population_size']} "
                    f"resets={events['resets']}"
                )


if __name__ == "__main__":
    main()
