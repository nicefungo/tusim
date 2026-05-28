#!/usr/bin/env python3
"""
TU CModel — Golden Reference Generator
========================================

Generates reference outputs for MMA operations using NumPy FP64 arithmetic.
The reference is computed in double precision and then cast to FP32 for
comparison with the cmodel's FP16→FP32→FP16 path.

Usage:
    python3 generate_reference.py                    # Generate all test cases
    python3 generate_reference.py --cases identity   # Identity matrices only
    python3 generate_reference.py --cases random     # Random tensors only
    python3 generate_reference.py --num-random 100   # 100 random test cases

Output:
    tests/golden/reference_data/*.json               # Reference outputs

Gap V1: No golden reference → Dual-path verification with NumPy reference.
"""

import json
import os
import struct
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


# Output directory
REF_DIR = Path(__file__).parent / "reference_data"
REF_DIR.mkdir(parents=True, exist_ok=True)


def fp32_to_fp16_bits(values: np.ndarray) -> list:
    """Convert FP32 values to FP16 bit patterns as Python ints."""
    f16 = values.astype(np.float16)
    return [int(x) for x in f16.view(np.uint16).flat]


def fp16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    """Convert FP16 bit patterns back to FP32."""
    f16 = bits.view(np.float16)
    return f16.astype(np.float32)


def mma_reference(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute O = W @ A in FP64 for maximum accuracy.
    W: [M, K], A: [K, N] — both FP16-range values.
    Returns O: [M, N] in FP32.
    """
    # Convert inputs to FP64 for the reference
    W_f64 = W.astype(np.float64)
    A_f64 = A.astype(np.float64)
    O_f64 = W_f64 @ A_f64
    return O_f64.astype(np.float32)


def generate_identity_case(M: int, N: int, K: int, case_id: str) -> dict:
    """Generate an identity-matrix test case."""
    # Identity: W = I (1.0 on diagonal)
    W = np.zeros((M, K), dtype=np.float32)
    d = min(M, K)
    W[np.arange(d), np.arange(d)] = 1.0

    # A = I
    A = np.zeros((K, N), dtype=np.float32)
    d2 = min(K, N)
    A[np.arange(d2), np.arange(d2)] = 1.0

    # Compute reference
    O_ref = mma_reference(W, A)

    # Convert inputs to FP16 bit patterns (as the cmodel would use)
    W_f16_bits = fp32_to_fp16_bits(W)
    A_f16_bits = fp32_to_fp16_bits(A)

    return {
        "case_id": case_id,
        "description": f"Identity: M={M}, N={N}, K={K}",
        "M": M, "N": N, "K": K,
        "has_bias": False,
        # Inputs as FP16 bits (uint16), stored as hex strings for readability
        "W_hex": [f"0x{int(w):04x}" for w in W_f16_bits],
        "A_hex": [f"0x{int(a):04x}" for a in A_f16_bits],
        # Reference output as FP32 values
        "O_ref": O_ref.flatten().tolist(),
        # Also include reference as hex for bit-exact comparison
        "O_ref_hex": [f"0x{int(s):08x}" for s in O_ref.view(np.uint32).flatten()],
        "tolerance": 0.01  # FP16 has ~3 decimal digits of precision
    }


def generate_known_value_case(case_id: str) -> dict:
    """All-0.5 W, all-2.0 A → every output = K * 0.5 * 2.0 = K."""
    M, N, K = 16, 8, 16

    W = np.full((M, K), 0.5, dtype=np.float32)
    A = np.full((K, N), 2.0, dtype=np.float32)

    O_ref = mma_reference(W, A)

    W_f16_bits = fp32_to_fp16_bits(W)
    A_f16_bits = fp32_to_fp16_bits(A)

    return {
        "case_id": case_id,
        "description": f"Known value: all-0.5 W × all-2.0 A → all-{K}.0",
        "M": M, "N": N, "K": K,
        "has_bias": False,
        "W_hex": [f"0x{int(w):04x}" for w in W_f16_bits],
        "A_hex": [f"0x{int(a):04x}" for a in A_f16_bits],
        "O_ref": O_ref.flatten().tolist(),
        "O_ref_hex": [f"0x{int(s):08x}" for s in O_ref.view(np.uint32).flatten()],
        "tolerance": 0.1
    }


def generate_random_case(
    seed: int, M: int, N: int, K: int, case_id: str
) -> dict:
    """Generate a random test case with a known seed."""
    rng = np.random.RandomState(seed)

    # Uniform in [-1, 1], then scale to FP16-safe range
    W = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    A = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)

    O_ref = mma_reference(W, A)

    W_f16_bits = fp32_to_fp16_bits(W)
    A_f16_bits = fp32_to_fp16_bits(A)

    return {
        "case_id": case_id,
        "description": f"Random seed={seed}: M={M}, N={N}, K={K}",
        "M": M, "N": N, "K": K,
        "has_bias": False,
        "seed": seed,
        "W_hex": [f"0x{int(w):04x}" for w in W_f16_bits],
        "A_hex": [f"0x{int(a):04x}" for a in A_f16_bits],
        "O_ref": O_ref.flatten().tolist(),
        "O_ref_hex": [f"0x{int(s):08x}" for s in O_ref.view(np.uint32).flatten()],
        "tolerance": 0.05  # Slightly looser for random values
    }


def generate_edge_case(case_id: str) -> dict:
    """Edge case: M=7, N=5, K=9 with all-ones → output should be all-9.0."""
    M, N, K = 7, 5, 9

    W = np.ones((M, K), dtype=np.float32)
    A = np.ones((K, N), dtype=np.float32)

    O_ref = mma_reference(W, A)

    W_f16_bits = fp32_to_fp16_bits(W)
    A_f16_bits = fp32_to_fp16_bits(A)

    return {
        "case_id": case_id,
        "description": f"Edge tiles: M={M}, N={N}, K={K} (non-multiple-of-PE)",
        "M": M, "N": N, "K": K,
        "has_bias": False,
        "W_hex": [f"0x{int(w):04x}" for w in W_f16_bits],
        "A_hex": [f"0x{int(a):04x}" for a in A_f16_bits],
        "O_ref": O_ref.flatten().tolist(),
        "O_ref_hex": [f"0x{int(s):08x}" for s in O_ref.view(np.uint32).flatten()],
        "tolerance": 0.15  # Larger tolerance for edge tiles
    }


def generate_bias_case(case_id: str) -> dict:
    """Bias test: zero W/A, sequential bias → output = bias."""
    M, N, K = 8, 8, 8

    W = np.zeros((M, K), dtype=np.float32)
    A = np.zeros((K, N), dtype=np.float32)

    # Bias: sequential values 0..(M*N-1)
    bias_vals = np.arange(M * N, dtype=np.float32).reshape(M, N)
    O_ref = bias_vals.copy()  # Since W@A = 0

    W_f16_bits = fp32_to_fp16_bits(W)
    A_f16_bits = fp32_to_fp16_bits(A)
    bias_f16_bits = fp32_to_fp16_bits(bias_vals)

    return {
        "case_id": case_id,
        "description": f"Bias-only: zero W/A, bias={0}..{M*N-1}",
        "M": M, "N": N, "K": K,
        "has_bias": True,
        "W_hex": [f"0x{int(w):04x}" for w in W_f16_bits],
        "A_hex": [f"0x{int(a):04x}" for a in A_f16_bits],
        "bias_hex": [f"0x{int(b):04x}" for b in bias_f16_bits],
        "O_ref": O_ref.flatten().tolist(),
        "O_ref_hex": [f"0x{int(s):08x}" for s in O_ref.view(np.uint32).flatten()],
        "tolerance": 0.01
    }


def generate_large_case(case_id: str) -> dict:
    """Large matrix: 64×64×64, all-0.25."""
    M, N, K = 64, 64, 64

    W = np.full((M, K), 0.25, dtype=np.float32)
    A = np.full((K, N), 0.25, dtype=np.float32)

    O_ref = mma_reference(W, A)

    W_f16_bits = fp32_to_fp16_bits(W)
    A_f16_bits = fp32_to_fp16_bits(A)

    return {
        "case_id": case_id,
        "description": f"Large: M={M}, N={N}, K={K}, all-0.25 → all-4.0",
        "M": M, "N": N, "K": K,
        "has_bias": False,
        "W_hex": [f"0x{int(w):04x}" for w in W_f16_bits],
        "A_hex": [f"0x{int(a):04x}" for a in A_f16_bits],
        "O_ref": O_ref.flatten().tolist(),
        "O_ref_hex": [f"0x{int(s):08x}" for s in O_ref.view(np.uint32).flatten()],
        "tolerance": 0.2  # Larger matrices accumulate more error
    }


def save_case(case: dict) -> str:
    """Save a test case to a JSON file. Returns the file path."""
    filename = f"{case['case_id']}.json"
    filepath = REF_DIR / filename
    with open(filepath, "w") as f:
        json.dump(case, f, indent=2)
    return str(filepath)


def generate_index(cases: List[str]) -> None:
    """Generate an index file listing all golden reference cases."""
    index = {
        "description": "TU CModel Golden Reference Index",
        "version": "1.0",
        "num_cases": len(cases),
        "cases": cases
    }
    with open(REF_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TU CModel Golden Reference Generator"
    )
    parser.add_argument(
        "--cases", type=str, default="all",
        choices=["all", "identity", "random", "known", "edge", "bias", "large"],
        help="Which test cases to generate (default: all)"
    )
    parser.add_argument(
        "--num-random", type=int, default=50,
        help="Number of random test cases to generate (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    args = parser.parse_args()

    generated = []
    case_type = args.cases

    # Identity cases
    if case_type in ("all", "identity"):
        identity_configs = [
            (16, 16, 16), (32, 32, 32), (8, 8, 16),
            (32, 16, 16), (4, 8, 16), (48, 48, 48),
            (20, 20, 20),
        ]
        for i, (M, N, K) in enumerate(identity_configs):
            case = generate_identity_case(M, N, K, f"identity_{M}x{N}x{K}")
            path = save_case(case)
            generated.append(f"identity_{M}x{N}x{K}")
            print(f"  Generated: {path}")

    # Known value case
    if case_type in ("all", "known"):
        case = generate_known_value_case("known_16x8x16")
        path = save_case(case)
        generated.append("known_16x8x16")
        print(f"  Generated: {path}")

    # Edge tile case
    if case_type in ("all", "edge"):
        case = generate_edge_case("edge_7x5x9")
        path = save_case(case)
        generated.append("edge_7x5x9")
        print(f"  Generated: {path}")

    # Bias case
    if case_type in ("all", "bias"):
        case = generate_bias_case("bias_8x8x8")
        path = save_case(case)
        generated.append("bias_8x8x8")
        print(f"  Generated: {path}")

    # Large case
    if case_type in ("all", "large"):
        case = generate_large_case("large_64x64x64")
        path = save_case(case)
        generated.append("large_64x64x64")
        print(f"  Generated: {path}")

    # Random cases
    if case_type in ("all", "random"):
        rng = np.random.RandomState(args.seed)
        dims = [
            (16, 16, 16), (8, 16, 32), (32, 8, 16),
            (4, 4, 8), (10, 10, 10), (15, 15, 15),
            (31, 17, 23), (7, 11, 13),
        ]
        for i in range(args.num_random):
            M, N, K = dims[i % len(dims)]
            # Vary the seed
            seed = args.seed + i * 137 + i * i
            case = generate_random_case(seed, M, N, K, f"random_{i:04d}")
            path = save_case(case)
            generated.append(f"random_{i:04d}")
        print(f"  Generated: {args.num_random} random cases")

    # Write index
    generate_index(generated)
    print(f"\n  Total: {len(generated)} golden reference cases generated")
    print(f"  Index: {REF_DIR / 'index.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
