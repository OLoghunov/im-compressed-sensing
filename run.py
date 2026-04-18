#!/usr/bin/env python3
"""
Единая точка входа IMCS.

  python run.py              — графический интерфейс (Qt по умолчанию)
  python run.py файл.png     — один прогон в консоли с параметрами по умолчанию
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imcs.pipeline import (
    DEFAULT_ALGORITHM,
    DEFAULT_BLOCK_EDGE,
    DEFAULT_BASIS,
    DEFAULT_CHROMA_RATIO_SCALE,
    DEFAULT_COLOR_MODE,
    DEFAULT_KEYFRAME_INTERVAL,
    DEFAULT_MATRIX_TYPE,
    DEFAULT_MEASUREMENT_MODE,
    DEFAULT_RATIO,
    DEFAULT_SEQUENCE_STRATEGY,
    run_color_image,
    run_image,
    run_sequence,
    run_signal,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IMCS: кодирование и декодирование. Без аргументов открывается окно.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Файл изображения или 1D сигнала (.png, .npy, …). Если не указан — GUI.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=f"Compression ratio (по умолчанию {DEFAULT_RATIO})",
    )
    parser.add_argument(
        "--algorithm",
        choices=("omp", "ista", "fista", "sa"),
        default=DEFAULT_ALGORITHM,
        help=f"Алгоритм восстановления (по умолчанию {DEFAULT_ALGORITHM})",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=DEFAULT_BLOCK_EDGE,
        metavar="N",
        help=(
            f"Размер блока 2D (по умолчанию {DEFAULT_BLOCK_EDGE}; "
            "0 — полный кадр только на малых кадрах)"
        ),
    )
    parser.add_argument(
        "--basis",
        choices=("dct", "wavelet"),
        default=DEFAULT_BASIS,
        help=f"Базис разреженности (по умолчанию {DEFAULT_BASIS})",
    )
    parser.add_argument(
        "--matrix",
        choices=("gaussian", "bernoulli", "random"),
        default=DEFAULT_MATRIX_TYPE,
        help=f"Тип измерительной матрицы (по умолчанию {DEFAULT_MATRIX_TYPE})",
    )
    parser.add_argument(
        "--measurement-mode",
        choices=("shared", "per_block"),
        default=DEFAULT_MEASUREMENT_MODE,
        help=f"Стратегия Phi для блоков (по умолчанию {DEFAULT_MEASUREMENT_MODE})",
    )
    parser.add_argument(
        "--color-mode",
        choices=("gray", "rgb", "ycbcr"),
        default=DEFAULT_COLOR_MODE,
        help=f"Режим обработки изображения (по умолчанию {DEFAULT_COLOR_MODE})",
    )
    parser.add_argument(
        "--chroma-ratio-scale",
        type=float,
        default=DEFAULT_CHROMA_RATIO_SCALE,
        help="Множитель ratio для Cb/Cr в режиме ycbcr",
    )
    parser.add_argument(
        "--sequence-strategy",
        choices=("independent", "residual"),
        default=DEFAULT_SEQUENCE_STRATEGY,
        help=f"Стратегия для папки кадров (по умолчанию {DEFAULT_SEQUENCE_STRATEGY})",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=DEFAULT_KEYFRAME_INTERVAL,
        help=(
            "Интервал ключевых кадров для residual-режима "
            f"(по умолчанию {DEFAULT_KEYFRAME_INTERVAL})"
        ),
    )
    args = parser.parse_args()

    if args.path is None:
        try:
            from imcs.gui_qt import run_app as run_qt_app

            run_qt_app()
        except Exception as exc:
            print(f"Qt GUI недоступен, используется legacy GUI: {exc}", file=sys.stderr)
            from imcs.gui import run_app as run_legacy_app

            run_legacy_app()
        return

    path = Path(args.path)
    if not path.exists():
        print(f"Путь не найден: {path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(__file__).resolve().parent / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if path.is_dir():
        run_sequence(
            path,
            output_dir,
            compression_ratio=args.ratio,
            algorithm=args.algorithm,
            block_edge=args.block,
            force_full_frame=False,
            verbose=True,
            basis=args.basis,
            matrix_type=args.matrix,
            measurement_mode=args.measurement_mode,
            color_mode=args.color_mode,
            sequence_strategy=args.sequence_strategy,
            chroma_ratio_scale=args.chroma_ratio_scale,
            keyframe_interval=args.keyframe_interval,
        )
    elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        if args.color_mode == "gray":
            run_image(
                path,
                output_dir,
                compression_ratio=args.ratio,
                algorithm=args.algorithm,
                block_edge=args.block,
                force_full_frame=False,
                visualize_convergence=False,
                verbose=True,
                basis=args.basis,
                matrix_type=args.matrix,
                measurement_mode=args.measurement_mode,
            )
        else:
            run_color_image(
                path,
                output_dir,
                compression_ratio=args.ratio,
                algorithm=args.algorithm,
                block_edge=args.block,
                force_full_frame=False,
                verbose=True,
                basis=args.basis,
                matrix_type=args.matrix,
                measurement_mode=args.measurement_mode,
                color_mode=args.color_mode,
                chroma_ratio_scale=args.chroma_ratio_scale,
            )
    elif suffix in [".npy", ".txt"]:
        run_signal(
            path,
            output_dir,
            compression_ratio=args.ratio,
            algorithm=args.algorithm,
            verbose=True,
            basis=args.basis,
            matrix_type=args.matrix,
        )
    else:
        print(f"Неподдерживаемый формат: {suffix}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
