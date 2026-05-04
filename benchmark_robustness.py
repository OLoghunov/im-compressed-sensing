#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mplconfig"),
)

from imcs.rd_compare import cases_from_directory
from imcs.robustness_compare import RobustnessConfig, run_robustness_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark robustness under partial data loss: Hybrid IMCS vs JPEG.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "examples" / "output",
    )
    parser.add_argument("--study-name", default="robustness_kodak_fast")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="kodak")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--losses",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 0.75],
        help="Fractions of lost enhancement/stream data.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--imcs-ratio", type=float, default=0.08)
    parser.add_argument("--measurement-dtype", choices=("int8", "int16", "float64"), default="int8")
    parser.add_argument("--low-frequency-coeffs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    root = args.dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    cases = cases_from_directory(root, args.dataset_name)
    if args.max_images > 0:
        cases = cases[: args.max_images]
    if not cases:
        raise SystemExit(f"No images found in {root}")

    config = RobustnessConfig(
        block_edge=int(args.block_size),
        compression_ratio=float(args.imcs_ratio),
        measurement_dtype=args.measurement_dtype,
        low_frequency_coeffs=int(args.low_frequency_coeffs),
        jpeg_quality=int(args.jpeg_quality),
    )

    out = run_robustness_study(
        args.output_dir,
        study_name=args.study_name,
        cases=cases,
        dataset_label=args.dataset_name,
        losses=tuple(float(x) for x in args.losses),
        config=config,
        clean_output=not args.no_clean,
        verbose=True,
    )
    print(f"Robustness benchmark done: {out}")
    print(f"  - {out / 'robustness_long.csv'}")
    print("  - fig_robustness_psnr_loss.png")
    print("  - fig_robustness_decodable_loss.png")


if __name__ == "__main__":
    main()
