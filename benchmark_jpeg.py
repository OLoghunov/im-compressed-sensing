#!/usr/bin/env python3
"""
Baseline JPEG benchmark (Pillow) для сравнения с IMCS: PSNR/SSIM, BPP, время код/декод.

Результаты: examples/output/benchmarks/<study>/summary.csv, jpeg_psnr_vs_bpp.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imcs.baseline_codecs import run_jpeg_baseline_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run JPEG baseline benchmark (same test cases as IMCS block study).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "examples" / "output",
        help="Root output directory (benchmarks go under benchmarks/<study>).",
    )
    parser.add_argument(
        "--study-name",
        default="jpeg_baseline",
        help="Subdirectory name under examples/output/benchmarks/.",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        help="JPEG quality values (Pillow).",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Load images as RGB (default: grayscale like IMCS synthetic runs).",
    )
    args = parser.parse_args()

    out = run_jpeg_baseline_study(
        output_dir=args.output_dir,
        study_name=args.study_name,
        qualities=tuple(args.qualities),
        color=args.color,
        verbose=True,
    )
    print(f"JPEG benchmark saved to: {out}")


if __name__ == "__main__":
    main()
