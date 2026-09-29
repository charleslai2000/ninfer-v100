#!/usr/bin/env python3
"""Compare same-artifact T018 CPU reference and CUDA taps, in first-divergence order."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np

PAIRS = [
    ("q.bf16", "q"), ("k_pre_scale.bf16", "k_pre_scale"),
    ("k.bf16", "k"), ("v.bf16", "v"),
    ("q_rope.bf16", "q_rope"), ("k_rope.bf16", "k_rope"),
    ("attention_pre_o.bf16", "pre_o"),
    ("o_projection.bf16", "o_projection"), ("output.bf16", "output"),
]


def bf16(path, shape):
    bits = np.fromfile(path, dtype="<u2")
    expected = int(np.prod(shape))
    if bits.size != expected:
        raise ValueError(f"{path}: expected {expected} BF16 elements, got {bits.size}")
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_npz", type=Path)
    ap.add_argument("cuda_tap_dir", type=Path)
    args = ap.parse_args()
    ref = np.load(args.reference_npz)
    T = int(ref["hidden"].shape[0])
    history_tokens = ref["attention_probabilities"].shape[2] - T
    shapes = {"q": (T, 1536), "k_pre_scale": (T, 2, 128), "k": (T, 2, 128), "v": (T, 256),
              "q_rope": (T, 12, 128), "k_rope": (T, 2, 128),
              "pre_o": (T, 12, 128), "o_projection": (T, 3072), "output": (T, 3072)}
    for cuda_name, ref_name in PAIRS:
        actual = bf16(args.cuda_tap_dir / cuda_name, shapes[ref_name])
        expected = np.asarray(ref[ref_name], dtype=np.float32)
        if actual.shape != expected.shape:
            raise ValueError(f"shape mismatch {ref_name}: CUDA={actual.shape} reference={expected.shape}")
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        max_abs = float(np.max(np.abs(delta)))
        denom = float(np.linalg.norm(expected.astype(np.float64)))
        rel_l2 = float(np.linalg.norm(delta) / denom) if denom else float(np.linalg.norm(delta))
        idx = int(np.argmax(np.abs(delta)))
        first_nonfinite = None
        mask = ~(np.isfinite(actual) & np.isfinite(expected))
        if mask.any(): first_nonfinite = int(np.flatnonzero(mask)[0])
        print(json.dumps({"tap": ref_name, "finite": finite, "max_abs": max_abs,
                          "relative_l2": rel_l2, "first_max_index": idx,
                          "first_nonfinite_flat_index": first_nonfinite}))
        if ref_name == "pre_o":
            coord = np.unravel_index(idx, actual.shape)
            print(json.dumps({"tap": ref_name, "coordinate": tuple(int(x) for x in coord),
                              "reference": float(expected[coord]),
                              "cuda": float(actual[coord]),
                              "history_tokens": history_tokens}))
        if not finite:
            print(f"FIRST_NONFINITE_TAP {ref_name}")
            return 2
        # Threshold is diagnostic only; inspect/report the first ordering boundary.
        # Do not turn this helper into a qualification acceptance gate.
        passed = max_abs <= 2e-2
        if not passed:
            print(f"FIRST_DIVERGENT_TAP {ref_name}")
            return 3
    print("T018_SAME_ARTIFACT_TAPS_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
