#!/usr/bin/env python3
"""
Reproducible benchmark runner for IMCS diploma experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imcs.benchmarking import (
    default_algorithms,
    default_block_sizes,
    default_compression_ratios,
    run_block_benchmark_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible IMCS benchmark studies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "examples" / "output",
        help="Directory where benchmark outputs will be saved.",
    )
    parser.add_argument(
        "--study-name",
        default="block_study",
        help="Subdirectory name under examples/output/benchmarks/ for this run.",
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=list(default_block_sizes()),
        help="Block sizes to evaluate.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=list(default_compression_ratios()),
        help="Compression ratios to evaluate.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(default_algorithms()),
        help="Algorithms to evaluate.",
    )
    parser.add_argument(
        "--basis",
        choices=("dct", "wavelet"),
        default="dct",
        help="Sparsity basis for the benchmark.",
    )
    parser.add_argument(
        "--matrix",
        choices=("gaussian", "bernoulli", "random"),
        default="gaussian",
        help="Measurement matrix type.",
    )
    parser.add_argument(
        "--measurement-mode",
        choices=("shared", "per_block"),
        default="shared",
        help="Phi strategy for blocked measurements.",
    )
    args = parser.parse_args()

    benchmark_dir = run_block_benchmark_study(
        output_dir=args.output_dir,
        study_name=args.study_name,
        block_sizes=args.block_sizes,
        compression_ratios=args.ratios,
        algorithms=args.algorithms,
        basis=args.basis,
        matrix_type=args.matrix,
        measurement_mode=args.measurement_mode,
        verbose=True,
    )
    print(f"Benchmark saved to: {benchmark_dir}")


if __name__ == "__main__":
    main()
