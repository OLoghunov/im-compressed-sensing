#!/usr/bin/env python3
"""
Thesis benchmark: JPEG/WebP/JPEG2000 baselines vs several IMCS variants.

Examples:
  python benchmark_rd_compare.py --internal-cases --preset smoke
  python benchmark_rd_compare.py --dataset-dir ~/datasets/kodak --dataset-name kodak --preset thesis --color
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mplconfig"),
)

from imcs.baseline_codecs import CodecSpec
from imcs.benchmarking import default_benchmark_cases
from imcs.rd_compare import (
    IMCSVariant,
    cases_from_directory,
    default_imcs_variants,
    fast_imcs_variants,
    quantized_imcs_variants,
    run_rd_comparison_study,
    smoke_imcs_variants,
)
from imcs.rd_plots import generate_all_figures_and_tables


def _codec_specs_from_args(args: argparse.Namespace) -> tuple[CodecSpec, ...]:
    specs: list[CodecSpec] = []
    requested = set(args.codecs)
    if "jpeg" in requested:
        specs.append(
            CodecSpec(
                name="jpeg",
                label="JPEG",
                param_name="quality",
                param_values=tuple(float(q) for q in args.baseline_qualities),
                pil_format="JPEG",
            )
        )
    if "webp" in requested:
        specs.append(
            CodecSpec(
                name="webp",
                label="WebP",
                param_name="quality",
                param_values=tuple(float(q) for q in args.baseline_qualities),
                pil_format="WEBP",
            )
        )
    if "jpeg2000" in requested:
        specs.append(
            CodecSpec(
                name="jpeg2000",
                label="JPEG2000",
                param_name="quality",
                param_values=tuple(float(q) for q in args.jpeg2000_qualities),
                pil_format="JPEG2000",
            )
        )
    return tuple(specs)


def _imcs_variants_from_args(args: argparse.Namespace) -> tuple[IMCSVariant, ...]:
    if args.preset == "smoke":
        return smoke_imcs_variants(use_color=args.color)
    if args.preset == "thesis-fast":
        return fast_imcs_variants(use_color=args.color)
    if args.preset == "quantized":
        return quantized_imcs_variants(use_color=args.color)
    if args.preset == "thesis":
        return default_imcs_variants(use_color=args.color)

    color_mode = args.color_mode if args.color else "gray"
    return (
        IMCSVariant(
            variant_id=(
                f"imcs_{args.algorithm}_{args.basis}_{args.measurement_mode}_"
                f"{'mean_' if not args.no_block_mean_residual else ''}"
                f"{f'lf{args.low_frequency_coeffs}_' if args.low_frequency_coeffs else ''}"
                f"{args.measurement_dtype}_b{args.block_size}"
            ),
            label=(
                f"IMCS {args.algorithm.upper()} {args.basis.upper()} "
                f"{args.measurement_mode} "
                f"{'+ mean ' if not args.no_block_mean_residual else ''}"
                f"{f'+ LF{args.low_frequency_coeffs} ' if args.low_frequency_coeffs else ''}"
                f"{args.measurement_dtype}"
            ),
            block_edge=args.block_size,
            algorithm=args.algorithm,
            basis=args.basis,
            matrix_type=args.matrix,
            measurement_mode=args.measurement_mode,
            measurement_dtype=args.measurement_dtype,
            block_mean_residual=not args.no_block_mean_residual,
            low_frequency_coeffs=args.low_frequency_coeffs,
            color_mode=color_mode,
            chroma_ratio_scale=args.chroma_scale,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready RD and timing figures for JPEG/WebP/JPEG2000 vs IMCS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "examples" / "output",
        help="Root output directory. Results go to benchmarks/<study-name>.",
    )
    parser.add_argument("--study-name", default="rd_compare", help="Benchmark subdirectory name.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Image folder, non-recursive.")
    parser.add_argument("--dataset-name", default="custom", help="Dataset label written to CSV.")
    parser.add_argument("--internal-cases", action="store_true", help="Use examples/input test cases.")
    parser.add_argument(
        "--preset",
        choices=("smoke", "thesis-fast", "thesis", "quantized", "custom"),
        default="thesis",
        help=(
            "smoke: one IMCS variant; thesis-fast: OMP shared/per-block only; "
            "thesis: OMP + FISTA variants; quantized: float64/int16/int8 sweep; "
            "custom: use single-variant args."
        ),
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=("jpeg", "webp", "jpeg2000"),
        default=["jpeg", "webp", "jpeg2000"],
        help="Baseline codecs to try. Unsupported Pillow encoders are skipped gracefully.",
    )
    parser.add_argument(
        "--baseline-qualities",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
        help="Quality grid for JPEG and WebP.",
    )
    parser.add_argument(
        "--jpeg2000-qualities",
        nargs="+",
        type=int,
        default=[25, 30, 35, 40, 45, 50],
        help="JPEG2000 PSNR quality-layer grid for Pillow.",
    )
    parser.add_argument(
        "--imcs-ratios",
        nargs="+",
        type=float,
        default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        help="IMCS compression_ratio grid. Low values keep bpp comparable to classical codecs.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeated timed runs per point.")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--algorithm", choices=("omp", "ista", "fista", "sa"), default="omp")
    parser.add_argument("--basis", choices=("dct", "wavelet"), default="dct")
    parser.add_argument("--matrix", choices=("gaussian", "bernoulli", "random"), default="gaussian")
    parser.add_argument("--measurement-mode", choices=("shared", "per_block"), default="shared")
    parser.add_argument(
        "--measurement-dtype",
        choices=("float64", "int16", "int8"),
        default="float64",
        help="IMCS measurement payload dtype for --preset custom.",
    )
    parser.add_argument(
        "--no-block-mean-residual",
        action="store_true",
        help="Disable block-mean residual coding for --preset custom.",
    )
    parser.add_argument(
        "--low-frequency-coeffs",
        type=int,
        default=0,
        help="Number of explicit low-frequency DCT AC coefficients for --preset custom.",
    )
    parser.add_argument("--color", action="store_true", help="Use RGB/YCbCr color pipeline.")
    parser.add_argument("--color-mode", choices=("rgb", "ycbcr"), default="ycbcr")
    parser.add_argument("--chroma-scale", type=float, default=0.5)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="Parallel worker processes for IMCS block decoding. Use 1 to disable.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clear benchmarks/<study-name> before the run.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-facets", type=int, default=8)
    parser.add_argument("--bar-baseline-quality", type=int, default=70)
    parser.add_argument("--bar-jpeg2000-quality", type=int, default=40)
    parser.add_argument("--bar-imcs-ratio", type=float, default=0.08)
    args = parser.parse_args()

    if args.internal_cases:
        cases = default_benchmark_cases()
        dataset_label = "internal"
    elif args.dataset_dir is not None:
        root = args.dataset_dir.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        cases = cases_from_directory(root, args.dataset_name)
        dataset_label = args.dataset_name
        if not cases:
            raise SystemExit(f"No images found in {root}")
    else:
        cases = default_benchmark_cases()
        dataset_label = "internal"

    codec_specs = _codec_specs_from_args(args)
    imcs_variants = _imcs_variants_from_args(args)

    out = run_rd_comparison_study(
        args.output_dir,
        study_name=args.study_name,
        cases=cases,
        dataset_label=dataset_label,
        jpeg_qualities=tuple(args.baseline_qualities),
        imcs_ratios=tuple(args.imcs_ratios),
        codec_specs=codec_specs,
        imcs_variants=imcs_variants,
        repeats=args.repeats,
        use_color=args.color,
        color_mode=args.color_mode,
        chroma_ratio_scale=args.chroma_scale,
        parallel_block_workers=None if args.parallel_workers <= 1 else int(args.parallel_workers),
        clean_output=not args.no_clean,
        verbose=True,
    )

    if not args.no_plots:
        print("Generating thesis figures and tables...", flush=True)
        generate_all_figures_and_tables(
            out,
            max_facets=args.max_facets,
            baseline_quality_ref=args.bar_baseline_quality,
            jpeg2000_quality_ref=args.bar_jpeg2000_quality,
            imcs_ratio_ref=args.bar_imcs_ratio,
        )
        print("Figures and tables generated.", flush=True)

    print(f"RD compare done: {out}")
    print(f"  - {out / 'rd_long.csv'}")
    print(f"  - {out / 'rd_config.json'}")
    if not args.no_plots:
        print("  - fig_5_7_psnr_bpp_by_image.png")
        print("  - fig_5_8_mean_psnr_bpp_ci.png")
        print("  - fig_5_9_encode_decode_time_ci.png")
        print("  - fig_5_10_quality_time_tradeoff.png")
        print("  - tables_for_thesis.md")


if __name__ == "__main__":
    main()
