from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .utils import create_basis_matrix, dct2

MAX_IMAGE_DISPLAY_EDGE = 768
MAX_IMAGE_ANALYSIS_EDGE = 256
MAX_SIGNAL_DISPLAY_POINTS = 4096
MAX_SIGNAL_ANALYSIS_POINTS = 2048
MAX_CURVE_POINTS = 2048
HISTOGRAM_BINS = 64


@dataclass
class AnalyticsPayload:
    mode: str
    basis: str
    source_label: str
    analysis_note: str
    original_display: np.ndarray
    reconstructed_display: np.ndarray
    error_display: np.ndarray
    histogram_edges: np.ndarray
    histogram_counts: np.ndarray
    coeff_display_original: np.ndarray
    coeff_display_reconstructed: np.ndarray
    sorted_coeff_original: np.ndarray
    sorted_coeff_reconstructed: np.ndarray
    convergence_images: list[np.ndarray]
    convergence_titles: list[str]


def build_analytics_payload(result: Any) -> AnalyticsPayload:
    basis = str(getattr(result, "basis", "dct"))
    color_mode = str(getattr(result, "color_mode", "gray"))
    original = np.asarray(result.original, dtype=np.float64)
    reconstructed = np.asarray(result.reconstructed, dtype=np.float64)
    convergence_images, convergence_titles = _load_convergence_images(
        getattr(result, "output_subdir", None)
    )
    working_original, working_reconstructed, basis_source_label = _select_analysis_arrays(
        original, reconstructed, color_mode
    )
    if working_original.ndim == 1:
        return _build_signal_payload(
            working_original,
            working_reconstructed,
            basis=basis,
            source_label=basis_source_label,
            convergence_images=convergence_images,
            convergence_titles=convergence_titles,
        )
    return _build_image_payload(
        original,
        reconstructed,
        working_original,
        working_reconstructed,
        basis=basis,
        basis_source_label=basis_source_label,
        convergence_images=convergence_images,
        convergence_titles=convergence_titles,
    )


def _select_analysis_arrays(
    original: np.ndarray, reconstructed: np.ndarray, color_mode: str
) -> tuple[np.ndarray, np.ndarray, str]:
    if original.ndim == 1:
        return original.ravel(), reconstructed.ravel(), "1D signal"
    if original.ndim == 2:
        return original, reconstructed, "Grayscale image"

    original_u8 = np.clip(original, 0, 255).astype(np.uint8)
    reconstructed_u8 = np.clip(reconstructed, 0, 255).astype(np.uint8)
    original_rgb = Image.fromarray(original_u8, mode="RGB")
    reconstructed_rgb = Image.fromarray(reconstructed_u8, mode="RGB")

    if color_mode == "ycbcr":
        original_y = np.array(original_rgb.convert("YCbCr"), dtype=np.float64)[..., 0]
        reconstructed_y = np.array(reconstructed_rgb.convert("YCbCr"), dtype=np.float64)[..., 0]
        return original_y, reconstructed_y, "Y channel (YCbCr)"

    original_gray = np.array(original_rgb.convert("L"), dtype=np.float64)
    reconstructed_gray = np.array(reconstructed_rgb.convert("L"), dtype=np.float64)
    return original_gray, reconstructed_gray, "Grayscale projection from RGB"


def _build_signal_payload(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    basis: str,
    source_label: str,
    convergence_images: list[np.ndarray],
    convergence_titles: list[str],
) -> AnalyticsPayload:
    signal_original, note = _prepare_signal_for_basis(original, basis)
    signal_reconstructed, _ = _prepare_signal_for_basis(reconstructed, basis)
    coeff_original = _signal_coefficients(signal_original, basis)
    coeff_reconstructed = _signal_coefficients(signal_reconstructed, basis)
    error_signal = np.abs(original - reconstructed)
    hist_counts, hist_edges = np.histogram(error_signal, bins=HISTOGRAM_BINS)
    return AnalyticsPayload(
        mode="signal",
        basis=basis,
        source_label=source_label,
        analysis_note=note,
        original_display=_downsample_signal(original, MAX_SIGNAL_DISPLAY_POINTS),
        reconstructed_display=_downsample_signal(reconstructed, MAX_SIGNAL_DISPLAY_POINTS),
        error_display=_downsample_signal(error_signal, MAX_SIGNAL_DISPLAY_POINTS),
        histogram_edges=hist_edges.astype(np.float32),
        histogram_counts=hist_counts.astype(np.float32),
        coeff_display_original=_downsample_signal(coeff_original, MAX_SIGNAL_DISPLAY_POINTS),
        coeff_display_reconstructed=_downsample_signal(
            coeff_reconstructed, MAX_SIGNAL_DISPLAY_POINTS
        ),
        sorted_coeff_original=_sorted_magnitudes(coeff_original, MAX_CURVE_POINTS),
        sorted_coeff_reconstructed=_sorted_magnitudes(
            coeff_reconstructed, MAX_CURVE_POINTS
        ),
        convergence_images=convergence_images,
        convergence_titles=convergence_titles,
    )


def _build_image_payload(
    original_display_source: np.ndarray,
    reconstructed_display_source: np.ndarray,
    original_analysis: np.ndarray,
    reconstructed_analysis: np.ndarray,
    *,
    basis: str,
    basis_source_label: str,
    convergence_images: list[np.ndarray],
    convergence_titles: list[str],
) -> AnalyticsPayload:
    tile_original, tile_reconstructed, note = _prepare_image_tile(
        original_analysis, reconstructed_analysis, basis
    )
    coeff_original = _image_coefficients(tile_original, basis)
    coeff_reconstructed = _image_coefficients(tile_reconstructed, basis)
    error_map = _image_error_map(original_display_source, reconstructed_display_source)
    source_label = "Color image" if original_display_source.ndim == 3 else basis_source_label
    analysis_note = (
        f"Basis analysis: {basis_source_label}; {note}"
        if original_display_source.ndim == 3
        else note
    )
    hist_counts, hist_edges = np.histogram(error_map, bins=HISTOGRAM_BINS)
    return AnalyticsPayload(
        mode="image",
        basis=basis,
        source_label=source_label,
        analysis_note=analysis_note,
        original_display=_downsample_image(original_display_source, MAX_IMAGE_DISPLAY_EDGE),
        reconstructed_display=_downsample_image(reconstructed_display_source, MAX_IMAGE_DISPLAY_EDGE),
        error_display=_downsample_image(error_map, MAX_IMAGE_DISPLAY_EDGE),
        histogram_edges=hist_edges.astype(np.float32),
        histogram_counts=hist_counts.astype(np.float32),
        coeff_display_original=np.log1p(np.abs(coeff_original)).astype(np.float32),
        coeff_display_reconstructed=np.log1p(np.abs(coeff_reconstructed)).astype(np.float32),
        sorted_coeff_original=_sorted_magnitudes(coeff_original, MAX_CURVE_POINTS),
        sorted_coeff_reconstructed=_sorted_magnitudes(
            coeff_reconstructed, MAX_CURVE_POINTS
        ),
        convergence_images=convergence_images,
        convergence_titles=convergence_titles,
    )


def _load_convergence_images(output_subdir: Any) -> tuple[list[np.ndarray], list[str]]:
    if output_subdir is None:
        return [], []
    output_path = Path(output_subdir)
    if not output_path.exists():
        return [], []

    images: list[np.ndarray] = []
    titles: list[str] = []
    for path in sorted(output_path.glob("convergence_*.png")):
        try:
            image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        stem = path.stem
        suffix = stem.split("_")[-1].upper()
        if suffix in {"Y", "CB", "CR", "R", "G", "B"}:
            titles.append(f"Сходимость {suffix}")
        else:
            titles.append("Сходимость")
        images.append(image)
    return images, titles


def _prepare_signal_for_basis(signal: np.ndarray, basis: str) -> tuple[np.ndarray, str]:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    length = signal.size
    if basis == "wavelet":
        target_len = _largest_power_of_two(min(length, MAX_SIGNAL_ANALYSIS_POINTS))
        if target_len <= 0:
            target_len = 1
        if length == target_len:
            return signal, f"Full signal, n={target_len}"
        resampled = _resample_1d(signal, target_len)
        return resampled, f"Resampled for Haar basis, n={target_len}"

    if length <= MAX_SIGNAL_ANALYSIS_POINTS:
        return signal, f"Full signal, n={length}"

    resampled = _resample_1d(signal, MAX_SIGNAL_ANALYSIS_POINTS)
    return resampled, f"Resampled for analysis, n={MAX_SIGNAL_ANALYSIS_POINTS}"


def _prepare_image_tile(
    original: np.ndarray, reconstructed: np.ndarray, basis: str
) -> tuple[np.ndarray, np.ndarray, str]:
    h, w = original.shape
    if basis == "wavelet":
        tile_h = _largest_power_of_two(min(h, MAX_IMAGE_ANALYSIS_EDGE))
        tile_w = _largest_power_of_two(min(w, MAX_IMAGE_ANALYSIS_EDGE))
    else:
        tile_h = min(h, MAX_IMAGE_ANALYSIS_EDGE)
        tile_w = min(w, MAX_IMAGE_ANALYSIS_EDGE)

    tile_h = max(1, tile_h)
    tile_w = max(1, tile_w)
    start_h = max((h - tile_h) // 2, 0)
    start_w = max((w - tile_w) // 2, 0)
    stop_h = start_h + tile_h
    stop_w = start_w + tile_w
    note = f"Central tile {tile_h}x{tile_w}"
    if tile_h == h and tile_w == w:
        note = f"Full frame {tile_h}x{tile_w}"
    return (
        original[start_h:stop_h, start_w:stop_w],
        reconstructed[start_h:stop_h, start_w:stop_w],
        note,
    )


def _signal_coefficients(signal: np.ndarray, basis: str) -> np.ndarray:
    psi = create_basis_matrix(signal.size, basis)
    return psi @ signal


def _image_coefficients(image: np.ndarray, basis: str) -> np.ndarray:
    if basis == "dct":
        return dct2(image)
    psi_row = create_basis_matrix(image.shape[0], basis)
    psi_col = create_basis_matrix(image.shape[1], basis)
    return psi_row @ image @ psi_col.T


def _sorted_magnitudes(array: np.ndarray, max_points: int) -> np.ndarray:
    sorted_values = np.sort(np.abs(np.asarray(array, dtype=np.float64)).ravel())[::-1]
    if sorted_values.size <= max_points:
        return sorted_values.astype(np.float32)
    indices = np.linspace(0, sorted_values.size - 1, max_points, dtype=np.int64)
    return sorted_values[indices].astype(np.float32)


def _downsample_signal(signal: np.ndarray, max_points: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size <= max_points:
        return signal.astype(np.float32)
    indices = np.linspace(0, signal.size - 1, max_points, dtype=np.int64)
    return signal[indices].astype(np.float32)


def _downsample_image(image: np.ndarray, max_edge: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape[:2]
    step = int(max(np.ceil(h / max_edge), np.ceil(w / max_edge), 1))
    return image[::step, ::step].astype(np.float32)


def _image_error_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    error = np.abs(np.asarray(original, dtype=np.float64) - np.asarray(reconstructed, dtype=np.float64))
    if error.ndim == 3:
        return error.mean(axis=2)
    return error


def _largest_power_of_two(value: int) -> int:
    if value < 1:
        return 0
    return 1 << (int(value).bit_length() - 1)


def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.size == target_len:
        return signal.astype(np.float64, copy=True)
    source_x = np.linspace(0.0, 1.0, signal.size, dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    return np.interp(target_x, source_x, signal).astype(np.float64)
