"""
Единая логика кодирования/декодирования
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from imcs import IMCSDecoder, IMCSEncoder
from imcs.cli import (
    create_output_directory,
    interpret_psnr,
    load_image,
    load_signal,
    save_image,
    save_report,
    save_signal,
    visualize_signal_comparison,
)
from imcs.convergence import (
    clear_previous_convergence_plots,
    create_representative_block_convergence_plot,
)
from imcs.utils import calculate_compression_metrics

# Значения по умолчанию (не нужно помнить флаги CLI)
DEFAULT_RATIO = 0.5
DEFAULT_ALGORITHM = "omp"
DEFAULT_BLOCK_EDGE = 8
DEFAULT_SEED = 42
DEFAULT_BASIS = "dct"
DEFAULT_MATRIX_TYPE = "gaussian"
DEFAULT_MEASUREMENT_MODE = "shared"
DEFAULT_COLOR_MODE = "ycbcr"
DEFAULT_CHROMA_RATIO_SCALE = 0.5
DEFAULT_SEQUENCE_STRATEGY = "independent"
DEFAULT_KEYFRAME_INTERVAL = 5

FULL_FRAME_PIXELS_MAX = 32 * 32
IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]


@dataclass
class ImageRunResult:
    image_path: Path
    output_subdir: Path
    original: np.ndarray
    reconstructed: np.ndarray
    compressed: bytes
    metrics: dict
    quality: str
    t_encode: float
    t_decode: float
    algorithm: str
    compression_ratio: float
    basis: str
    matrix_type: str
    measurement_mode: str
    block_size: Optional[Tuple[int, int]]
    decode_profile: Optional[dict] = None


@dataclass
class SignalRunResult:
    signal_path: Path
    output_subdir: Path
    original: np.ndarray
    reconstructed: np.ndarray
    compressed: bytes
    metrics: dict
    quality: str
    t_encode: float
    t_decode: float
    algorithm: str
    compression_ratio: float
    basis: str
    matrix_type: str
    decode_profile: Optional[dict] = None


@dataclass
class ColorImageRunResult:
    image_path: Path
    output_subdir: Path
    original: np.ndarray
    reconstructed: np.ndarray
    metrics: dict
    quality: str
    t_encode: float
    t_decode: float
    algorithm: str
    compression_ratio: float
    basis: str
    matrix_type: str
    measurement_mode: str
    color_mode: str
    channel_ratios: tuple[float, float, float]
    decode_profile: Optional[dict] = None


@dataclass
class SequenceRunResult:
    sequence_path: Path
    output_subdir: Path
    frame_count: int
    average_metrics: dict
    avg_t_encode: float
    avg_t_decode: float
    decode_fps: float
    strategy: str
    color_mode: str
    frame_records: list[dict]


def _algorithm_map(name: str) -> str:
    return {
        "omp": "omp",
        "ista": "iterative_threshold",
        "fista": "fista",
        "sa": "simulated_annealing",
    }[name]


def _clamp_ratio(value: float) -> float:
    return float(min(max(value, 0.01), 0.99))


def _aggregate_decode_profiles(profiles: list[Optional[dict]]) -> Optional[dict]:
    merged: Optional[dict] = None
    for profile in profiles:
        if not profile:
            continue
        if merged is None:
            merged = dict(profile)
            continue
        for key, value in profile.items():
            if key == "max_block_reconstruction_s":
                merged[key] = max(float(merged.get(key, 0.0)), float(value))
            elif key == "parallel_workers":
                merged[key] = max(int(merged.get(key, 1)), int(value))
            elif key == "blocks_total":
                merged[key] = int(merged.get(key, 0)) + int(value)
            else:
                merged[key] = float(merged.get(key, 0.0)) + float(value)
    if merged is None:
        return None
    blocks_total = int(merged.get("blocks_total", 0))
    reconstruction_s = float(merged.get("reconstruction_s", 0.0))
    merged["avg_block_reconstruction_s"] = (
        reconstruction_s / blocks_total if blocks_total > 0 else 0.0
    )
    return merged


def _profile_log_lines(profile: Optional[dict], indent: str = "  ") -> list[str]:
    if not profile:
        return []
    return [
        f"{indent}Профиль декодера:",
        f"{indent}  workers={int(profile.get('parallel_workers', 1))}",
        f"{indent}  blocks={int(profile.get('blocks_total', 0))}",
    ]


def _block_size_for_image(
    original: np.ndarray,
    block_edge: int,
    force_full_frame: bool,
    verbose: bool,
) -> Optional[Tuple[int, int]]:
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    block_size: Optional[Tuple[int, int]] = None if block_edge == 0 else (block_edge, block_edge)

    if (
        original.ndim == 2
        and block_size is None
        and original.size > FULL_FRAME_PIXELS_MAX
        and not force_full_frame
    ):
        block_size = (DEFAULT_BLOCK_EDGE, DEFAULT_BLOCK_EDGE)
        log(
            "  Полнокадровый режим на этом размере отключён (очень долгое декодирование). "
            f"Используются блоки {block_size[0]}×{block_size[1]}. "
            "Для полного кадра: force_full_frame=True\n"
        )

    return block_size


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return np.array(img.convert("YCbCr"), dtype=np.float64)


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.clip(ycbcr, 0, 255).astype(np.uint8), mode="YCbCr")
    return np.array(img.convert("RGB"), dtype=np.float64)


def _list_frame_files(sequence_path: Path) -> list[Path]:
    return sorted(
        path
        for path in sequence_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def encode_signal(
    original: np.ndarray,
    compression_ratio: float,
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
) -> tuple[float, bytes]:
    encoder = IMCSEncoder(
        compression_ratio=compression_ratio,
        seed=DEFAULT_SEED,
        sparsity_basis=basis,
        matrix_type=matrix_type,
    )
    t0 = time.time()
    compressed = encoder.encode(original)
    return time.time() - t0, compressed


def decode_signal(
    compressed: bytes,
    algorithm: str,
    collect_profile: bool = False,
    parallel_block_workers: Optional[int] = None,
    verbose: bool = False,
) -> tuple[float, np.ndarray, IMCSDecoder]:
    log_fn = (lambda message: print(message, flush=True)) if verbose else None
    decoder = IMCSDecoder(
        reconstruction_algorithm=_algorithm_map(algorithm),
        parallel_block_workers=parallel_block_workers,
        log_fn=log_fn,
    )
    t0 = time.perf_counter()
    try:
        reconstructed = decoder.decode(compressed, return_history=False, profile=collect_profile)
    except Exception:
        if parallel_block_workers is None:
            raise
        if verbose:
            print("[decode] parallel path failed, retrying in serial mode", flush=True)
        decoder = IMCSDecoder(
            reconstruction_algorithm=_algorithm_map(algorithm),
            log_fn=log_fn,
        )
        reconstructed = decoder.decode(compressed, return_history=False, profile=collect_profile)
    return time.perf_counter() - t0, reconstructed, decoder


def encode_image(
    original: np.ndarray,
    compression_ratio: float,
    block_size: Optional[Tuple[int, int]] = None,
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
    measurement_mode: str = DEFAULT_MEASUREMENT_MODE,
) -> tuple[float, bytes]:
    encoder = IMCSEncoder(
        compression_ratio=compression_ratio,
        seed=DEFAULT_SEED,
        sparsity_basis=basis,
        matrix_type=matrix_type,
        block_size=block_size,
        measurement_mode=measurement_mode,
    )
    t0 = time.time()
    compressed = encoder.encode(original)
    return time.time() - t0, compressed


def decode_image(
    compressed: bytes,
    algorithm: str,
    visualize: bool,
    collect_profile: bool = False,
    parallel_block_workers: Optional[int] = None,
    verbose: bool = False,
) -> tuple[float, np.ndarray, IMCSDecoder]:
    log_fn = (lambda message: print(message, flush=True)) if verbose else None
    decoder = IMCSDecoder(
        reconstruction_algorithm=_algorithm_map(algorithm),
        parallel_block_workers=parallel_block_workers,
        log_fn=log_fn,
    )
    t0 = time.perf_counter()
    try:
        reconstructed = decoder.decode(
            compressed, return_history=visualize, profile=collect_profile
        )
    except Exception:
        if parallel_block_workers is None:
            raise
        if verbose:
            print("[decode] parallel path failed, retrying in serial mode", flush=True)
        decoder = IMCSDecoder(
            reconstruction_algorithm=_algorithm_map(algorithm),
            log_fn=log_fn,
        )
        reconstructed = decoder.decode(
            compressed, return_history=visualize, profile=collect_profile
        )
    return time.perf_counter() - t0, reconstructed, decoder


def _process_grayscale_array(
    original: np.ndarray,
    compression_ratio: float,
    algorithm: str,
    block_edge: int,
    force_full_frame: bool,
    visualize_convergence: bool,
    verbose: bool,
    basis: str,
    matrix_type: str,
    measurement_mode: str,
    collect_decode_profile: bool = False,
    parallel_block_workers: Optional[int] = None,
) -> tuple[Optional[Tuple[int, int]], float, bytes, float, np.ndarray, IMCSDecoder, dict, str]:
    block_size = _block_size_for_image(original, block_edge, force_full_frame, verbose)
    t_encode, compressed = encode_image(
        original,
        compression_ratio,
        block_size=block_size,
        basis=basis,
        matrix_type=matrix_type,
        measurement_mode=measurement_mode,
    )
    t_decode, reconstructed, decoder = decode_image(
        compressed,
        algorithm,
        False,
        collect_profile=collect_decode_profile,
        parallel_block_workers=parallel_block_workers,
        verbose=verbose,
    )
    metrics = calculate_compression_metrics(original, reconstructed)
    quality = interpret_psnr(metrics["psnr"])
    return block_size, t_encode, compressed, t_decode, reconstructed, decoder, metrics, quality


def _process_color_array(
    original_rgb: np.ndarray,
    compression_ratio: float,
    algorithm: str,
    block_edge: int,
    force_full_frame: bool,
    verbose: bool,
    basis: str,
    matrix_type: str,
    measurement_mode: str,
    color_mode: str,
    chroma_ratio_scale: float,
    collect_decode_profile: bool = False,
    parallel_block_workers: Optional[int] = None,
) -> tuple[np.ndarray, list[bytes], float, float, dict, str, tuple[float, float, float], Optional[dict]]:
    if color_mode not in {"rgb", "ycbcr"}:
        raise ValueError("color_mode must be 'rgb' or 'ycbcr'")

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if color_mode == "rgb":
        working = original_rgb
        channel_names = ("R", "G", "B")
    else:
        log("Преобразование RGB -> YCbCr...")
        working = _rgb_to_ycbcr(original_rgb)
        channel_names = ("Y", "Cb", "Cr")
        log("  ✓ Преобразование завершено")

    channel_ratios = (
        _clamp_ratio(compression_ratio),
        _clamp_ratio(compression_ratio * chroma_ratio_scale),
        _clamp_ratio(compression_ratio * chroma_ratio_scale),
    )
    if color_mode == "rgb":
        ratio_value = _clamp_ratio(compression_ratio)
        channel_ratios = (ratio_value, ratio_value, ratio_value)

    channel_reconstructed: list[np.ndarray] = []
    channel_compressed: list[bytes] = []
    channel_profiles: list[Optional[dict]] = []
    total_encode = 0.0
    total_decode = 0.0

    for idx, channel_name in enumerate(channel_names):
        channel = working[..., idx]
        log(
            f"[Канал {channel_name}] ratio={channel_ratios[idx]:.2f}, "
            f"shape={channel.shape}, workers={parallel_block_workers or 1}"
        )
        (
            block_size,
            t_encode,
            compressed,
            t_decode,
            reconstructed,
            decoder,
            channel_metrics,
            _,
        ) = _process_grayscale_array(
            channel,
            channel_ratios[idx],
            algorithm,
            block_edge,
            force_full_frame,
            False,
            verbose,
            basis,
            matrix_type,
            measurement_mode,
            collect_decode_profile=collect_decode_profile,
            parallel_block_workers=parallel_block_workers,
        )
        channel_reconstructed.append(reconstructed)
        channel_compressed.append(compressed)
        channel_profiles.append(decoder.last_profile)
        total_encode += t_encode
        total_decode += t_decode
        bs_note = f"{block_size[0]}×{block_size[1]}" if block_size else "полный кадр"
        log(
            f"  ✓ Канал {channel_name}: block={bs_note}, encoded={len(compressed)} байт, "
            f"encode={t_encode:.3f} c, decode={t_decode:.3f} c, "
            f"PSNR={channel_metrics['psnr']:.2f} dB, SSIM={channel_metrics['ssim']:.4f}"
        )
        for line in _profile_log_lines(decoder.last_profile, indent="    "):
            log(line)

    reconstructed_working = np.stack(channel_reconstructed, axis=2)
    if color_mode == "rgb":
        reconstructed_rgb = reconstructed_working
    else:
        log("Преобразование YCbCr -> RGB...")
        reconstructed_rgb = _ycbcr_to_rgb(reconstructed_working)
        log("  ✓ Обратное преобразование завершено")

    metrics = calculate_compression_metrics(original_rgb, reconstructed_rgb)
    quality = interpret_psnr(metrics["psnr"])
    return (
        reconstructed_rgb,
        channel_compressed,
        total_encode,
        total_decode,
        metrics,
        quality,
        channel_ratios,
        _aggregate_decode_profiles(channel_profiles),
    )


def _save_image_results(
    output_subdir: Path,
    image_path: Path,
    original: np.ndarray,
    reconstructed: np.ndarray,
    compressed: bytes,
    algorithm: str,
    compression_ratio: float,
    t_encode: float,
    t_decode: float,
    metrics: dict,
    quality: str,
    extra_fields: Optional[dict] = None,
) -> None:
    with open(output_subdir / "compressed.imcs", "wb") as f:
        f.write(compressed)
    save_image(original, output_subdir / "original.png")
    save_image(reconstructed, output_subdir / f"reconstructed_{algorithm}.png")
    save_report(
        output_subdir / "report.txt",
        image_path,
        original.shape,
        compression_ratio,
        algorithm,
        t_encode,
        t_decode,
        original.nbytes,
        len(compressed),
        metrics,
        quality,
        extra_fields=extra_fields,
    )


def run_image(
    image_path: Path,
    output_dir: Path,
    compression_ratio: float = DEFAULT_RATIO,
    algorithm: str = DEFAULT_ALGORITHM,
    block_edge: int = DEFAULT_BLOCK_EDGE,
    force_full_frame: bool = False,
    visualize_convergence: bool = False,
    verbose: bool = True,
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
    measurement_mode: str = DEFAULT_MEASUREMENT_MODE,
    output_name: Optional[str] = None,
    collect_decode_profile: bool = False,
    parallel_block_workers: Optional[int] = None,
) -> Optional[ImageRunResult]:
    """
    Кодирует и декодирует одно grayscale-изображение. block_edge=0 — полный кадр
    (на крупных кадрах принудительно 8×8, если не force_full_frame).
    """

    def log(*a, **k):
        if verbose:
            print(*a, **k)

    log("\n" + "=" * 80)
    log(f"Обработка: {image_path.name}")
    log("=" * 80 + "\n")

    original = load_image(image_path, verbose=verbose)
    if original is None:
        return None
    requested_workers = parallel_block_workers or 1
    log(
        "Конфигурация: "
        f"ratio={compression_ratio}, algorithm={algorithm.upper()}, basis={basis}, "
        f"matrix={matrix_type}, phi={measurement_mode}, requested_workers={requested_workers}"
    )

    block_size, t_encode, compressed, t_decode, reconstructed, decoder, metrics, quality = (
        _process_grayscale_array(
            original,
            compression_ratio,
            algorithm,
            block_edge,
            force_full_frame,
            visualize_convergence,
            verbose,
            basis,
            matrix_type,
            measurement_mode,
            collect_decode_profile=collect_decode_profile,
            parallel_block_workers=parallel_block_workers,
        )
    )

    log(f"  Размер: {original.shape}")
    log(f"  Значения: [{original.min():.0f}, {original.max():.0f}], среднее: {original.mean():.1f}")
    log()
    bs_note = f"{block_size[0]}×{block_size[1]}" if block_size else "полное изображение"
    if block_size:
        block_rows = int(math.ceil(original.shape[0] / block_size[0]))
        block_cols = int(math.ceil(original.shape[1] / block_size[1]))
        log(f"  Сетка блоков: {block_rows}×{block_cols} ({block_rows * block_cols} блоков)")
    else:
        log("  Режим обработки: полный кадр без разбиения")
    log(
        "Кодирование "
        f"(compression_ratio={compression_ratio}, блоки: {bs_note}, basis={basis}, "
        f"matrix={matrix_type}, phi={measurement_mode})..."
    )
    log(f"  ✓ Сжато за {t_encode:.3f} сек")
    log(f"  Размер: {len(compressed)} байт (было {original.nbytes} байт)")
    log(f"  Степень сжатия: {original.nbytes / len(compressed):.2f}x")
    log()
    log(f"Декодирование (алгоритм: {algorithm.upper()})...")
    log(f"  ✓ Восстановлено за {t_decode:.3f} сек")
    for line in _profile_log_lines(decoder.last_profile):
        log(line)
    log()
    log("Качество восстановления:")
    log(f"  MSE:   {metrics['mse']:.2f}")
    log(f"  PSNR:  {metrics['psnr']:.2f} dB")
    log(f"  SSIM:  {metrics['ssim']:.4f}")
    log(f"  MAE:   {metrics['mae']:.2f}")
    log(f"  Оценка качества: {quality}")
    log()

    output_subdir = create_output_directory(output_dir, output_name or image_path.stem)
    extra_fields = {
        "Basis": basis,
        "Matrix type": matrix_type,
        "Measurement mode": measurement_mode,
        "Block size": bs_note,
    }
    _save_image_results(
        output_subdir,
        image_path,
        original,
        reconstructed,
        compressed,
        algorithm,
        compression_ratio,
        t_encode,
        t_decode,
        metrics,
        quality,
        extra_fields=extra_fields,
    )

    if visualize_convergence:
        log("Создание визуализации сходимости по репрезентативному блоку...")
        clear_previous_convergence_plots(output_subdir)
        create_representative_block_convergence_plot(
            original,
            reconstructed,
            algorithm,
            output_subdir,
            basis=basis,
            matrix_type=matrix_type,
            measurement_mode=measurement_mode,
            compression_ratio=compression_ratio,
            block_size=block_size,
            seed=DEFAULT_SEED,
        )
        log("  ✓ Визуализация сохранена")
        log(f"  - convergence_{algorithm}.png")
        log()

    log(f"✓ Результаты сохранены в: {output_subdir}/")
    log("  - original.png")
    log(f"  - reconstructed_{algorithm}.png")
    log("  - compressed.imcs")
    log("  - report.txt")
    if visualize_convergence:
        log(f"  - convergence_{algorithm}.png")

    return ImageRunResult(
        image_path=image_path,
        output_subdir=output_subdir,
        original=original,
        reconstructed=reconstructed,
        compressed=compressed,
        metrics=metrics,
        quality=quality,
        t_encode=t_encode,
        t_decode=t_decode,
        algorithm=algorithm,
        compression_ratio=compression_ratio,
        basis=basis,
        matrix_type=matrix_type,
        measurement_mode=measurement_mode,
        block_size=block_size,
        decode_profile=decoder.last_profile,
    )


def run_color_image(
    image_path: Path,
    output_dir: Path,
    compression_ratio: float = DEFAULT_RATIO,
    algorithm: str = DEFAULT_ALGORITHM,
    block_edge: int = DEFAULT_BLOCK_EDGE,
    force_full_frame: bool = False,
    verbose: bool = True,
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
    measurement_mode: str = DEFAULT_MEASUREMENT_MODE,
    color_mode: str = "ycbcr",
    chroma_ratio_scale: float = DEFAULT_CHROMA_RATIO_SCALE,
    output_name: Optional[str] = None,
    visualize_convergence: bool = False,
    collect_decode_profile: bool = False,
    parallel_block_workers: Optional[int] = None,
) -> Optional[ColorImageRunResult]:
    def log(*a, **k):
        if verbose:
            print(*a, **k)

    log("\n" + "=" * 80)
    log(f"Обработка цветного изображения: {image_path.name}")
    log("=" * 80 + "\n")

    original = load_image(image_path, verbose=verbose, preserve_color=True)
    if original is None:
        return None
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("run_color_image expects an RGB image")
    requested_workers = parallel_block_workers or 1
    block_size = _block_size_for_image(original[..., 0], block_edge, force_full_frame, False)
    log(
        "Конфигурация: "
        f"ratio={compression_ratio}, algorithm={algorithm.upper()}, basis={basis}, "
        f"matrix={matrix_type}, phi={measurement_mode}, color_mode={color_mode.upper()}, "
        f"requested_workers={requested_workers}"
    )
    log(
        "Коэффициенты каналов: "
        f"base={compression_ratio:.2f}, chroma_scale={chroma_ratio_scale:.2f}"
    )

    (
        reconstructed,
        channel_compressed,
        t_encode,
        t_decode,
        metrics,
        quality,
        channel_ratios,
        decode_profile,
    ) = _process_color_array(
        original,
        compression_ratio,
        algorithm,
        block_edge,
        force_full_frame,
        verbose,
        basis,
        matrix_type,
        measurement_mode,
        color_mode,
        chroma_ratio_scale,
        collect_decode_profile=collect_decode_profile,
        parallel_block_workers=parallel_block_workers,
    )

    default_name = f"{image_path.stem}_{color_mode}"
    output_subdir = create_output_directory(output_dir, output_name or default_name)
    save_image(original, output_subdir / "original.png")
    save_image(reconstructed, output_subdir / f"reconstructed_{algorithm}.png")

    channel_labels = ("R", "G", "B") if color_mode == "rgb" else ("Y", "Cb", "Cr")
    log(f"  Размер: {original.shape}")
    log(
        "  Каналы: "
        + ", ".join(
            f"{label}(ratio={ratio:.2f}, bytes={len(compressed)})"
            for label, ratio, compressed in zip(channel_labels, channel_ratios, channel_compressed)
        )
    )
    log(
        f"  Суммарный размер сжатых данных: {sum(len(chunk) for chunk in channel_compressed)} байт "
        f"(было {original.nbytes} байт)"
    )
    log(f"  Суммарное кодирование: {t_encode:.3f} сек")
    log(f"  Суммарное декодирование: {t_decode:.3f} сек")
    for line in _profile_log_lines(decode_profile):
        log(line)
    for label, compressed in zip(channel_labels, channel_compressed):
        with open(output_subdir / f"compressed_{label.lower()}.imcs", "wb") as f:
            f.write(compressed)

    save_report(
        output_subdir / "report.txt",
        image_path,
        original.shape,
        compression_ratio,
        algorithm,
        t_encode,
        t_decode,
        int(original.nbytes),
        int(sum(len(chunk) for chunk in channel_compressed)),
        metrics,
        quality,
        extra_fields={
            "Color mode": color_mode,
            "Basis": basis,
            "Matrix type": matrix_type,
            "Measurement mode": measurement_mode,
            "Channel ratios": ", ".join(f"{ratio:.2f}" for ratio in channel_ratios),
        },
    )

    log(f"  Размер: {original.shape}")
    log(f"  Цветовой режим: {color_mode.upper()}")
    log(f"  PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
    log(f"  Кодирование: {t_encode:.3f} сек | декодирование: {t_decode:.3f} сек")
    if decode_profile:
        log(
            "  Профиль декодера: "
            f"recon={decode_profile['reconstruction_s']:.3f} c, "
            f"Phi={decode_profile['phi_build_s']:.3f} c, "
            f"A={decode_profile['sensing_matrix_build_s']:.3f} c, "
            f"workers={int(decode_profile['parallel_workers'])}"
        )
    if visualize_convergence:
        log("Создание визуализаций сходимости по репрезентативным блокам каналов...")
        clear_previous_convergence_plots(output_subdir)
        working_original = original if color_mode == "rgb" else _rgb_to_ycbcr(original)
        working_reconstructed = reconstructed if color_mode == "rgb" else _rgb_to_ycbcr(reconstructed)
        for idx, label in enumerate(channel_labels):
            create_representative_block_convergence_plot(
                working_original[..., idx],
                working_reconstructed[..., idx],
                algorithm,
                output_subdir,
                basis=basis,
                matrix_type=matrix_type,
                measurement_mode=measurement_mode,
                compression_ratio=channel_ratios[idx],
                block_size=block_size,
                seed=DEFAULT_SEED,
                channel_label=label,
            )
        log("  ✓ Визуализации сохранены")
        for label in channel_labels:
            log(f"  - convergence_{algorithm}_{label.lower()}.png")
    log(f"✓ Результаты сохранены в: {output_subdir}/")

    return ColorImageRunResult(
        image_path=image_path,
        output_subdir=output_subdir,
        original=original,
        reconstructed=reconstructed,
        metrics=metrics,
        quality=quality,
        t_encode=t_encode,
        t_decode=t_decode,
        algorithm=algorithm,
        compression_ratio=compression_ratio,
        basis=basis,
        matrix_type=matrix_type,
        measurement_mode=measurement_mode,
        color_mode=color_mode,
        channel_ratios=channel_ratios,
        decode_profile=decode_profile,
    )


def run_sequence(
    sequence_path: Path,
    output_dir: Path,
    compression_ratio: float = DEFAULT_RATIO,
    algorithm: str = DEFAULT_ALGORITHM,
    block_edge: int = DEFAULT_BLOCK_EDGE,
    force_full_frame: bool = False,
    verbose: bool = True,
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
    measurement_mode: str = DEFAULT_MEASUREMENT_MODE,
    color_mode: str = DEFAULT_COLOR_MODE,
    sequence_strategy: str = DEFAULT_SEQUENCE_STRATEGY,
    chroma_ratio_scale: float = DEFAULT_CHROMA_RATIO_SCALE,
    keyframe_interval: int = DEFAULT_KEYFRAME_INTERVAL,
) -> Optional[SequenceRunResult]:
    def log(*a, **k):
        if verbose:
            print(*a, **k)

    frame_paths = _list_frame_files(sequence_path)
    if not frame_paths:
        if verbose:
            print(f"❌ В папке нет поддерживаемых кадров: {sequence_path}")
        return None

    output_subdir = create_output_directory(output_dir, f"{sequence_path.stem}_{sequence_strategy}")
    frames_dir = output_subdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    encode_times: list[float] = []
    decode_times: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    reference_gray: Optional[np.ndarray] = None

    log("\n" + "=" * 80)
    log(f"Обработка последовательности: {sequence_path.name}")
    log("=" * 80 + "\n")

    for idx, frame_path in enumerate(frame_paths):
        frame_name = f"{idx:04d}_{frame_path.stem}"
        frame_output_dir = frames_dir / frame_name
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        if color_mode != "gray" and sequence_strategy != "independent":
            raise ValueError("Residual sequence mode is currently supported only for grayscale")

        if color_mode == "gray":
            current = load_image(frame_path, verbose=False)
            if current is None:
                continue

            is_keyframe = idx == 0 or sequence_strategy == "independent" or (
                idx % keyframe_interval == 0
            )
            if reference_gray is not None and current.shape != reference_gray.shape:
                is_keyframe = True

            if is_keyframe or reference_gray is None:
                result = run_image(
                    frame_path,
                    frames_dir,
                    compression_ratio=compression_ratio,
                    algorithm=algorithm,
                    block_edge=block_edge,
                    force_full_frame=force_full_frame,
                    visualize_convergence=False,
                    verbose=False,
                    basis=basis,
                    matrix_type=matrix_type,
                    measurement_mode=measurement_mode,
                )
                if result is None:
                    continue
                reconstructed = result.reconstructed
                compressed_size = len(result.compressed)
                t_encode = result.t_encode
                t_decode = result.t_decode
                metrics = result.metrics
                reference_gray = reconstructed
            else:
                residual = current - reference_gray
                (
                    _,
                    t_encode,
                    compressed,
                    t_decode,
                    reconstructed_residual,
                    _,
                    _,
                    _,
                ) = _process_grayscale_array(
                    residual,
                    compression_ratio,
                    algorithm,
                    block_edge,
                    force_full_frame,
                    False,
                    False,
                    basis,
                    matrix_type,
                    measurement_mode,
                )
                reconstructed = np.clip(reference_gray + reconstructed_residual, 0, 255)
                metrics = calculate_compression_metrics(current, reconstructed)
                compressed_size = len(compressed)
                reference_gray = reconstructed
                save_image(current, frame_output_dir / "original.png")
                save_image(reconstructed, frame_output_dir / f"reconstructed_{algorithm}.png")
                with open(frame_output_dir / "compressed_residual.imcs", "wb") as f:
                    f.write(compressed)
                save_report(
                    frame_output_dir / "report.txt",
                    frame_path,
                    current.shape,
                    compression_ratio,
                    algorithm,
                    t_encode,
                    t_decode,
                    current.nbytes,
                    compressed_size,
                    metrics,
                    interpret_psnr(metrics["psnr"]),
                    extra_fields={
                        "Sequence strategy": sequence_strategy,
                        "Frame type": "residual",
                        "Basis": basis,
                        "Matrix type": matrix_type,
                        "Measurement mode": measurement_mode,
                    },
                )
        else:
            color_result = run_color_image(
                frame_path,
                frames_dir,
                compression_ratio=compression_ratio,
                algorithm=algorithm,
                block_edge=block_edge,
                force_full_frame=force_full_frame,
                verbose=False,
                basis=basis,
                matrix_type=matrix_type,
                measurement_mode=measurement_mode,
                color_mode=color_mode,
                chroma_ratio_scale=chroma_ratio_scale,
            )
            if color_result is None:
                continue
            reconstructed = color_result.reconstructed
            compressed_size = int(
                sum(
                    path.stat().st_size
                    for path in color_result.output_subdir.glob("compressed_*.imcs")
                )
            )
            t_encode = color_result.t_encode
            t_decode = color_result.t_decode
            metrics = color_result.metrics

        record = {
            "frame": frame_path.name,
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "t_encode": float(t_encode),
            "t_decode": float(t_decode),
            "compressed_size": int(compressed_size),
        }
        records.append(record)
        encode_times.append(float(t_encode))
        decode_times.append(float(t_decode))
        psnr_values.append(float(metrics["psnr"]))
        ssim_values.append(float(metrics["ssim"]))

    summary_path = output_subdir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame", "psnr", "ssim", "t_encode", "t_decode", "compressed_size"],
        )
        writer.writeheader()
        writer.writerows(records)

    average_metrics = {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }
    avg_t_encode = float(np.mean(encode_times))
    avg_t_decode = float(np.mean(decode_times))
    decode_fps = 0.0 if avg_t_decode <= 0 else 1.0 / avg_t_decode

    report_path = output_subdir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЁТ О ПОСЛЕДОВАТЕЛЬНОСТИ IMCS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Папка: {sequence_path}\n")
        f.write(f"Кадров: {len(records)}\n")
        f.write(f"Стратегия: {sequence_strategy}\n")
        f.write(f"Цветовой режим: {color_mode}\n")
        f.write(f"Basis: {basis}\n")
        f.write(f"Matrix type: {matrix_type}\n")
        f.write(f"Measurement mode: {measurement_mode}\n\n")
        f.write(f"Средний PSNR: {average_metrics['psnr']:.2f} dB\n")
        f.write(f"Средний SSIM: {average_metrics['ssim']:.4f}\n")
        f.write(f"Среднее время кодирования: {avg_t_encode:.3f} сек\n")
        f.write(f"Среднее время декодирования: {avg_t_decode:.3f} сек\n")
        f.write(f"Оценочная скорость декодирования: {decode_fps:.2f} FPS\n")

    log(f"✓ Последовательность обработана: {len(records)} кадров")
    log(
        f"  Средний PSNR: {average_metrics['psnr']:.2f} dB | "
        f"средний SSIM: {average_metrics['ssim']:.4f}"
    )
    log(f"  Оценочная скорость декодирования: {decode_fps:.2f} FPS")
    log(f"✓ Результаты сохранены в: {output_subdir}/")

    return SequenceRunResult(
        sequence_path=sequence_path,
        output_subdir=output_subdir,
        frame_count=len(records),
        average_metrics=average_metrics,
        avg_t_encode=avg_t_encode,
        avg_t_decode=avg_t_decode,
        decode_fps=decode_fps,
        strategy=sequence_strategy,
        color_mode=color_mode,
        frame_records=records,
    )


def run_signal(
    signal_path: Path,
    output_dir: Path,
    compression_ratio: float = DEFAULT_RATIO,
    algorithm: str = DEFAULT_ALGORITHM,
    verbose: bool = True,
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
    collect_decode_profile: bool = False,
    parallel_block_workers: Optional[int] = None,
) -> Optional[SignalRunResult]:
    def log(*a, **k):
        if verbose:
            print(*a, **k)

    log("\n" + "=" * 80)
    log(f"Обработка 1D сигнала: {signal_path.name}")
    log("=" * 80 + "\n")

    original = load_signal(signal_path, verbose=verbose)
    if original is None:
        return None
    requested_workers = parallel_block_workers or 1
    log(
        "Конфигурация: "
        f"ratio={compression_ratio}, algorithm={algorithm.upper()}, basis={basis}, "
        f"matrix={matrix_type}, requested_workers={requested_workers}"
    )

    log(f"  Размер: {len(original)} отсчётов")
    log(
        f"  Значения: [{original.min():.2f}, {original.max():.2f}], "
        f"среднее: {original.mean():.2f}"
    )
    log()

    log(
        f"Кодирование (compression_ratio={compression_ratio}, basis={basis}, "
        f"matrix={matrix_type})..."
    )
    t_encode, compressed = encode_signal(
        original,
        compression_ratio,
        basis=basis,
        matrix_type=matrix_type,
    )
    log(f"  ✓ Сжато за {t_encode:.3f} сек")
    log(f"  Размер: {len(compressed)} байт (было {original.nbytes} байт)")
    log(f"  Степень сжатия: {original.nbytes / len(compressed):.2f}x")
    log()

    log(f"Декодирование (алгоритм: {algorithm.upper()})...")
    t_decode, reconstructed, decoder = decode_signal(
        compressed,
        algorithm,
        collect_profile=collect_decode_profile,
        parallel_block_workers=parallel_block_workers,
        verbose=verbose,
    )
    log(f"  ✓ Восстановлено за {t_decode:.3f} сек")
    for line in _profile_log_lines(decoder.last_profile):
        log(line)
    log()

    metrics = calculate_compression_metrics(original, reconstructed)
    quality = interpret_psnr(metrics["psnr"])

    log("Качество восстановления:")
    log(f"  MSE:   {metrics['mse']:.2f}")
    log(f"  PSNR:  {metrics['psnr']:.2f} dB")
    log(f"  SSIM:  {metrics['ssim']:.4f}")
    log(f"  MAE:   {metrics['mae']:.2f}")
    log(f"  Оценка качества: {quality}")
    log()

    output_subdir = create_output_directory(output_dir, signal_path.stem)

    with open(output_subdir / "compressed.imcs", "wb") as f:
        f.write(compressed)
    save_signal(original, output_subdir / "original.npy")
    save_signal(reconstructed, output_subdir / f"reconstructed_{algorithm}.npy")

    visualize_signal_comparison(
        original,
        reconstructed,
        output_subdir / f"comparison_{algorithm}.png",
        verbose=verbose,
    )

    save_report(
        output_subdir / "report.txt",
        signal_path,
        (len(original),),
        compression_ratio,
        algorithm,
        t_encode,
        t_decode,
        original.nbytes,
        len(compressed),
        metrics,
        quality,
        extra_fields={
            "Basis": basis,
            "Matrix type": matrix_type,
        },
    )

    log(f"✓ Результаты сохранены в: {output_subdir}/")
    log("  - original.npy")
    log(f"  - reconstructed_{algorithm}.npy")
    log("  - compressed.imcs")
    log("  - report.txt")
    log(f"  - comparison_{algorithm}.png")

    return SignalRunResult(
        signal_path=signal_path,
        output_subdir=output_subdir,
        original=original,
        reconstructed=reconstructed,
        compressed=compressed,
        metrics=metrics,
        quality=quality,
        t_encode=t_encode,
        t_decode=t_decode,
        algorithm=algorithm,
        compression_ratio=compression_ratio,
        basis=basis,
        matrix_type=matrix_type,
        decode_profile=decoder.last_profile,
    )
