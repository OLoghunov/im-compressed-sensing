"""
Joint RD benchmark for baseline codecs and IMCS variants.

The output is a long-form CSV suitable for thesis plots:
one row = one image, method variant, parameter point and repeat.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from imcs.baseline_codecs import CodecSpec, benchmark_codec_once, default_codec_specs
from imcs.benchmarking import BenchmarkCase, default_benchmark_cases
from imcs.cli import load_image
from imcs.pipeline import (
    DEFAULT_BASIS,
    DEFAULT_CHROMA_RATIO_SCALE,
    DEFAULT_MATRIX_TYPE,
    DEFAULT_MEASUREMENT_MODE,
    ColorImageRunResult,
    ImageRunResult,
    run_color_image,
    run_image,
)
from imcs.rd_common import bpp_from_compressed_size


@dataclass(frozen=True)
class IMCSVariant:
    variant_id: str
    label: str
    block_edge: int = 8
    algorithm: str = "omp"
    basis: str = DEFAULT_BASIS
    matrix_type: str = DEFAULT_MATRIX_TYPE
    measurement_mode: str = DEFAULT_MEASUREMENT_MODE
    measurement_dtype: str = "float64"
    block_mean_residual: bool = True
    low_frequency_coeffs: int = 0
    color_mode: str = "ycbcr"
    chroma_ratio_scale: float = DEFAULT_CHROMA_RATIO_SCALE


def default_imcs_variants(*, use_color: bool = False) -> tuple[IMCSVariant, ...]:
    color_mode = "ycbcr" if use_color else "gray"
    return (
        IMCSVariant(
            variant_id="imcs_omp_dct_shared_mean_b8",
            label="IMCS OMP DCT shared + mean",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="shared",
            color_mode=color_mode,
        ),
        IMCSVariant(
            variant_id="imcs_omp_dct_perblock_mean_b8",
            label="IMCS OMP DCT per-block + mean",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="per_block",
            color_mode=color_mode,
        ),
        IMCSVariant(
            variant_id="imcs_fista_dct_shared_mean_b8",
            label="IMCS FISTA DCT shared + mean",
            block_edge=8,
            algorithm="fista",
            basis="dct",
            measurement_mode="shared",
            color_mode=color_mode,
        ),
    )


def smoke_imcs_variants(*, use_color: bool = False) -> tuple[IMCSVariant, ...]:
    color_mode = "ycbcr" if use_color else "gray"
    return (
        IMCSVariant(
            variant_id="imcs_omp_dct_shared_mean_b8",
            label="IMCS OMP DCT shared + mean",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="shared",
            color_mode=color_mode,
        ),
    )


def fast_imcs_variants(*, use_color: bool = False) -> tuple[IMCSVariant, ...]:
    color_mode = "ycbcr" if use_color else "gray"
    return (
        IMCSVariant(
            variant_id="imcs_omp_dct_shared_mean_b8",
            label="IMCS OMP DCT shared + mean",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="shared",
            color_mode=color_mode,
        ),
        IMCSVariant(
            variant_id="imcs_omp_dct_perblock_mean_b8",
            label="IMCS OMP DCT per-block + mean",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="per_block",
            color_mode=color_mode,
        ),
    )


def quantized_imcs_variants(*, use_color: bool = False) -> tuple[IMCSVariant, ...]:
    color_mode = "ycbcr" if use_color else "gray"
    return (
        IMCSVariant(
            variant_id="imcs_omp_dct_shared_mean_lf4_float64_b8",
            label="IMCS OMP DCT shared + mean + LF4 float64",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="shared",
            measurement_dtype="float64",
            low_frequency_coeffs=4,
            color_mode=color_mode,
        ),
        IMCSVariant(
            variant_id="imcs_omp_dct_shared_mean_lf4_int16_b8",
            label="IMCS OMP DCT shared + mean + LF4 int16",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="shared",
            measurement_dtype="int16",
            low_frequency_coeffs=4,
            color_mode=color_mode,
        ),
        IMCSVariant(
            variant_id="imcs_omp_dct_shared_mean_lf4_int8_b8",
            label="IMCS OMP DCT shared + mean + LF4 int8",
            block_edge=8,
            algorithm="omp",
            basis="dct",
            measurement_mode="shared",
            measurement_dtype="int8",
            low_frequency_coeffs=4,
            color_mode=color_mode,
        ),
    )


def rd_long_fieldnames() -> list[str]:
    return [
        "dataset",
        "case_key",
        "input_file",
        "width",
        "height",
        "channels",
        "method",
        "method_family",
        "variant_id",
        "variant_label",
        "repeat_index",
        "param_name",
        "param_value",
        "bpp",
        "psnr_db",
        "ssim",
        "mae",
        "encode_time_s",
        "decode_time_s",
        "compressed_size_bytes",
        "imcs_block_size",
        "imcs_algorithm",
        "imcs_basis",
        "imcs_matrix_type",
        "imcs_measurement_mode",
        "imcs_measurement_dtype",
        "imcs_block_mean_residual",
        "imcs_low_frequency_coeffs",
        "imcs_color_mode",
        "imcs_chroma_ratio_scale",
    ]


def cases_from_directory(
    root: Path,
    dataset_name: str,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> list[BenchmarkCase]:
    """Return all images from a non-recursive directory scan."""

    seen: set[Path] = set()
    paths: list[Path] = []
    for ext in extensions:
        for candidate in sorted(root.glob(f"*{ext}")) + sorted(root.glob(f"*{ext.upper()}")):
            resolved = candidate.resolve()
            if candidate.is_file() and resolved not in seen:
                seen.add(resolved)
                paths.append(candidate)

    used_keys: dict[str, int] = {}
    cases: list[BenchmarkCase] = []
    for path in paths:
        key = path.stem
        if key in used_keys:
            used_keys[key] += 1
            key = f"{path.stem}_{used_keys[key]}"
        else:
            used_keys[key] = 0
        cases.append(
            BenchmarkCase(
                key=key,
                path=path,
                category=dataset_name,
                description=path.name,
            )
        )
    return cases


def _imcs_compressed_total_bytes(result: ImageRunResult | ColorImageRunResult) -> int:
    if isinstance(result, ImageRunResult):
        return int(len(result.compressed))
    return int(sum(fp.stat().st_size for fp in result.output_subdir.glob("compressed_*.imcs")))


def _safe_metric(value: float) -> float:
    return round(float(value), 6) if math.isfinite(float(value)) else 99.0


def _row_common(
    *,
    dataset: str,
    case: BenchmarkCase,
    original: Any,
    method: str,
    method_family: str,
    variant_id: str,
    variant_label: str,
    repeat_index: int,
    param_name: str,
    param_value: str,
    bpp: float,
    psnr: float,
    ssim: float,
    mae: float,
    t_enc: float,
    t_dec: float,
    cbytes: int,
    imcs_meta: dict[str, str],
) -> dict[str, Any]:
    h, w = original.shape[:2]
    channels = original.shape[2] if original.ndim == 3 else 1
    return {
        "dataset": dataset,
        "case_key": case.key,
        "input_file": case.path.name,
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "method": method,
        "method_family": method_family,
        "variant_id": variant_id,
        "variant_label": variant_label,
        "repeat_index": int(repeat_index),
        "param_name": param_name,
        "param_value": param_value,
        "bpp": round(float(bpp), 6),
        "psnr_db": _safe_metric(float(psnr)),
        "ssim": round(float(ssim), 6),
        "mae": round(float(mae), 6),
        "encode_time_s": round(float(t_enc), 6),
        "decode_time_s": round(float(t_dec), 6),
        "compressed_size_bytes": int(cbytes),
        **imcs_meta,
    }


def _empty_imcs_meta() -> dict[str, str]:
    return {
        "imcs_block_size": "",
        "imcs_algorithm": "",
        "imcs_basis": "",
        "imcs_matrix_type": "",
        "imcs_measurement_mode": "",
        "imcs_measurement_dtype": "",
        "imcs_block_mean_residual": "",
        "imcs_low_frequency_coeffs": "",
        "imcs_color_mode": "",
        "imcs_chroma_ratio_scale": "",
    }


def _imcs_meta(variant: IMCSVariant, *, use_color: bool) -> dict[str, str]:
    return {
        "imcs_block_size": str(variant.block_edge),
        "imcs_algorithm": variant.algorithm,
        "imcs_basis": variant.basis,
        "imcs_matrix_type": variant.matrix_type,
        "imcs_measurement_mode": variant.measurement_mode,
        "imcs_measurement_dtype": variant.measurement_dtype,
        "imcs_block_mean_residual": str(variant.block_mean_residual),
        "imcs_low_frequency_coeffs": str(variant.low_frequency_coeffs),
        "imcs_color_mode": variant.color_mode if use_color else "gray",
        "imcs_chroma_ratio_scale": str(variant.chroma_ratio_scale if use_color else ""),
    }


def _variant_output_name(
    case: BenchmarkCase,
    variant: IMCSVariant,
    ratio: float,
    repeat_index: int,
) -> str:
    ratio_str = f"{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"rd_{case.key}_{variant.variant_id}_r{ratio_str}_rep{repeat_index}"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} c"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} мин {sec:04.1f} c"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} ч {int(minutes):02d} мин"


def run_rd_comparison_study(
    output_dir: Path,
    study_name: str = "rd_compare",
    cases: Sequence[BenchmarkCase] | None = None,
    dataset_label: str = "internal",
    jpeg_qualities: Sequence[int] | None = None,
    imcs_ratios: Sequence[float] | None = None,
    *,
    codec_specs: Sequence[CodecSpec] | None = None,
    imcs_variants: Sequence[IMCSVariant] | None = None,
    repeats: int = 3,
    block_edge: int = 8,
    algorithm: str = "omp",
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
    measurement_mode: str = DEFAULT_MEASUREMENT_MODE,
    use_color: bool = False,
    color_mode: str = "ycbcr",
    chroma_ratio_scale: float = DEFAULT_CHROMA_RATIO_SCALE,
    force_full_frame: bool = False,
    parallel_block_workers: int | None = None,
    clean_output: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Run JPEG/WebP/JPEG2000 and one or more IMCS variants on the same images.

    If imcs_variants is not provided, the legacy single-variant arguments are
    used to preserve old callers.
    """

    benchmark_dir = output_dir / "benchmarks" / study_name
    if clean_output and benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    cases = list(default_benchmark_cases() if cases is None else cases)
    qualities = tuple(
        jpeg_qualities
        if jpeg_qualities is not None
        else (10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
    )
    ratios = tuple(
        imcs_ratios if imcs_ratios is not None else (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
    )
    specs = tuple(
        default_codec_specs(include_webp=True, include_jpeg2000=True)
        if codec_specs is None
        else codec_specs
    )
    if imcs_variants is None:
        imcs_variants = (
            IMCSVariant(
                variant_id=f"imcs_{algorithm}_{basis}_{measurement_mode}_b{block_edge}",
                label=f"IMCS {algorithm.upper()} {basis.upper()} {measurement_mode}",
                block_edge=block_edge,
                algorithm=algorithm,
                basis=basis,
                matrix_type=matrix_type,
                measurement_mode=measurement_mode,
                measurement_dtype="float64",
                block_mean_residual=True,
                low_frequency_coeffs=0,
                color_mode=color_mode,
                chroma_ratio_scale=chroma_ratio_scale,
            ),
        )
    variants = tuple(imcs_variants)
    repeats = max(1, int(repeats))

    rows: list[dict[str, Any]] = []
    codec_failures: set[tuple[str, str]] = set()
    codec_points_per_case = sum(len(spec.param_values) * repeats for spec in specs)
    imcs_points_per_case = len(variants) * len(ratios) * repeats
    total_points = len(cases) * (codec_points_per_case + imcs_points_per_case)
    point_index = 0
    started_at = time.perf_counter()

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    def log_point_start(
        *,
        case_index: int,
        case: BenchmarkCase,
        label: str,
        param_name: str,
        param_value: float,
        repeat_index: int,
    ) -> float:
        nonlocal point_index
        point_index += 1
        elapsed = time.perf_counter() - started_at
        avg = elapsed / max(point_index - 1, 1)
        eta = avg * max(total_points - point_index + 1, 0)
        log(
            f"[{point_index}/{total_points}] image {case_index}/{len(cases)} "
            f"{case.key}: {label}, {param_name}={param_value}, "
            f"repeat {repeat_index + 1}/{repeats} | elapsed={_format_duration(elapsed)}, "
            f"ETA={_format_duration(eta)}"
        )
        return time.perf_counter()

    def log_point_done(
        *,
        step_started_at: float,
        bpp: float,
        psnr: float,
        t_enc: float,
        t_dec: float,
    ) -> None:
        step_elapsed = time.perf_counter() - step_started_at
        elapsed = time.perf_counter() - started_at
        avg = elapsed / max(point_index, 1)
        eta = avg * max(total_points - point_index, 0)
        log(
            f"    done in {_format_duration(step_elapsed)} | "
            f"bpp={bpp:.4f}, PSNR={min(psnr, 99.0):.2f} dB, "
            f"enc={t_enc:.4f}s, dec={t_dec:.4f}s | ETA={_format_duration(eta)}"
        )

    log("=== RD benchmark started ===")
    log(f"Output: {benchmark_dir}")
    log(f"Dataset: {dataset_label}; images={len(cases)}; repeats={repeats}")
    log("Baseline codecs: " + ", ".join(spec.label for spec in specs))
    log("IMCS variants: " + ", ".join(variant.label for variant in variants))
    log(f"IMCS parallel workers: {parallel_block_workers or 1}")
    log(f"Total benchmark points: {total_points}")

    for case_index, case in enumerate(cases, start=1):
        if not case.path.is_file():
            log(f"Skip missing: {case.path}")
            continue

        original = load_image(case.path, verbose=False, preserve_color=use_color)
        if original is None:
            continue
        if use_color and original.ndim != 3:
            log(f"Skip (need RGB): {case.path}")
            continue

        h, w = original.shape[:2]
        baseline_meta = _empty_imcs_meta()
        channels = original.shape[2] if original.ndim == 3 else 1
        log(
            f"\n--- Image {case_index}/{len(cases)}: {case.key} "
            f"({case.path.name}), size={w}x{h}, channels={channels} ---"
        )

        for spec in specs:
            values = spec.param_values
            for value in values:
                for repeat_index in range(repeats):
                    step_started_at = log_point_start(
                        case_index=case_index,
                        case=case,
                        label=spec.label,
                        param_name=spec.param_name,
                        param_value=float(value),
                        repeat_index=repeat_index,
                    )
                    try:
                        stats = benchmark_codec_once(
                            original,
                            spec,
                            float(value),
                            warmup_roundtrips=1 if repeat_index == 0 else 0,
                        )
                    except Exception as exc:  # Pillow build may not support WEBP/JPEG2000.
                        key = (spec.name, str(exc))
                        if key not in codec_failures and verbose:
                            codec_failures.add(key)
                            log(f"Skip codec {spec.label}: {exc}")
                        break
                    rows.append(
                        _row_common(
                            dataset=dataset_label,
                            case=case,
                            original=original,
                            method=spec.name,
                            method_family="baseline",
                            variant_id=spec.name,
                            variant_label=spec.label,
                            repeat_index=repeat_index,
                            param_name=spec.param_name,
                            param_value=str(value),
                            bpp=float(stats["bpp"]),
                            psnr=float(stats["psnr_db"]),
                            ssim=float(stats["ssim"]),
                            mae=float(stats["mae"]),
                            t_enc=float(stats["encode_time_s"]),
                            t_dec=float(stats["decode_time_s"]),
                            cbytes=int(stats["compressed_size_bytes"]),
                            imcs_meta=baseline_meta,
                        )
                    )
                    log_point_done(
                        step_started_at=step_started_at,
                        bpp=float(stats["bpp"]),
                        psnr=float(stats["psnr_db"]),
                        t_enc=float(stats["encode_time_s"]),
                        t_dec=float(stats["decode_time_s"]),
                    )

        for variant in variants:
            meta = _imcs_meta(variant, use_color=use_color)
            for ratio in ratios:
                for repeat_index in range(repeats):
                    step_started_at = log_point_start(
                        case_index=case_index,
                        case=case,
                        label=variant.label,
                        param_name="compression_ratio",
                        param_value=float(ratio),
                        repeat_index=repeat_index,
                    )
                    out_name = _variant_output_name(case, variant, float(ratio), repeat_index)
                    if use_color:
                        result = run_color_image(
                            case.path,
                            benchmark_dir,
                            compression_ratio=float(ratio),
                            algorithm=variant.algorithm,
                            block_edge=variant.block_edge,
                            force_full_frame=force_full_frame,
                            verbose=False,
                            basis=variant.basis,
                            matrix_type=variant.matrix_type,
                            measurement_mode=variant.measurement_mode,
                            measurement_dtype=variant.measurement_dtype,
                            block_mean_residual=variant.block_mean_residual,
                            low_frequency_coeffs=variant.low_frequency_coeffs,
                            color_mode=variant.color_mode,
                            chroma_ratio_scale=variant.chroma_ratio_scale,
                            output_name=out_name,
                            visualize_convergence=False,
                            parallel_block_workers=parallel_block_workers,
                        )
                    else:
                        result = run_image(
                            case.path,
                            benchmark_dir,
                            compression_ratio=float(ratio),
                            algorithm=variant.algorithm,
                            block_edge=variant.block_edge,
                            force_full_frame=force_full_frame,
                            visualize_convergence=False,
                            verbose=False,
                            basis=variant.basis,
                            matrix_type=variant.matrix_type,
                            measurement_mode=variant.measurement_mode,
                            measurement_dtype=variant.measurement_dtype,
                            block_mean_residual=variant.block_mean_residual,
                            low_frequency_coeffs=variant.low_frequency_coeffs,
                            output_name=out_name,
                            parallel_block_workers=parallel_block_workers,
                        )
                    if result is None:
                        continue
                    cbytes = _imcs_compressed_total_bytes(result)
                    bpp = bpp_from_compressed_size(cbytes, int(w), int(h))
                    rows.append(
                        _row_common(
                            dataset=dataset_label,
                            case=case,
                            original=original,
                            method="imcs",
                            method_family="imcs",
                            variant_id=variant.variant_id,
                            variant_label=variant.label,
                            repeat_index=repeat_index,
                            param_name="compression_ratio",
                            param_value=str(float(ratio)),
                            bpp=bpp,
                            psnr=float(result.metrics["psnr"]),
                            ssim=float(result.metrics["ssim"]),
                            mae=float(result.metrics["mae"]),
                            t_enc=float(result.t_encode),
                            t_dec=float(result.t_decode),
                            cbytes=cbytes,
                            imcs_meta=meta,
                        )
                    )
                    log_point_done(
                        step_started_at=step_started_at,
                        bpp=bpp,
                        psnr=float(result.metrics["psnr"]),
                        t_enc=float(result.t_encode),
                        t_dec=float(result.t_decode),
                    )

        log(f"--- Done image {case_index}/{len(cases)}: {case.key} ---")

    with open(benchmark_dir / "rd_long.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rd_long_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "dataset": dataset_label,
        "repeats": repeats,
        "codec_specs": [asdict(spec) for spec in specs],
        "jpeg_qualities": list(qualities),
        "imcs_ratios": list(ratios),
        "imcs_variants": [asdict(variant) for variant in variants],
        "use_color": use_color,
        "force_full_frame": force_full_frame,
        "parallel_block_workers": parallel_block_workers,
        "clean_output": clean_output,
        "cases": [{"key": case.key, "path": str(case.path)} for case in cases],
    }
    with open(benchmark_dir / "rd_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log(
        f"=== RD benchmark finished: {len(rows)} rows, "
        f"elapsed={_format_duration(time.perf_counter() - started_at)} ==="
    )
    return benchmark_dir
