from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageFile

from imcs.benchmarking import BenchmarkCase
from imcs.cli import load_image
from imcs.decoder import IMCSDecoder
from imcs.encoder import IMCSEncoder
from imcs.utils import (
    calculate_compression_metrics,
    create_2d_basis_matrix,
    generate_measurement_matrix,
    idct2,
    omp,
    zigzag_indices,
)


@dataclass(frozen=True)
class RobustnessConfig:
    block_edge: int = 8
    compression_ratio: float = 0.08
    measurement_dtype: str = "int8"
    low_frequency_coeffs: int = 4
    jpeg_quality: int = 70
    seed: int = 42


def robustness_fieldnames() -> list[str]:
    return [
        "dataset",
        "case_key",
        "input_file",
        "width",
        "height",
        "method",
        "loss_fraction",
        "received_fraction",
        "decoded",
        "bpp_received",
        "psnr_db",
        "ssim",
        "mae",
        "encode_time_s",
        "decode_time_s",
        "encoded_size_bytes",
        "received_size_bytes",
        "notes",
    ]


def _image_to_pil_gray(original: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(original, 0, 255).astype(np.uint8), mode="L")


def _encode_jpeg(original: np.ndarray, quality: int) -> tuple[bytes, float]:
    buf = BytesIO()
    t0 = time.perf_counter()
    _image_to_pil_gray(original).save(
        buf,
        format="JPEG",
        quality=int(quality),
        optimize=False,
        progressive=False,
    )
    return buf.getvalue(), time.perf_counter() - t0


def _decode_jpeg_strict(blob: bytes) -> tuple[np.ndarray | None, float, str]:
    if not blob:
        return None, 0.0, "empty stream"
    t0 = time.perf_counter()
    try:
        img = Image.open(BytesIO(blob))
        arr = np.array(img.convert("L"), dtype=np.float64)
        return arr, time.perf_counter() - t0, ""
    except Exception as exc:
        return None, time.perf_counter() - t0, str(exc)


def _decode_jpeg_tolerant(blob: bytes) -> tuple[np.ndarray | None, float, str]:
    if not blob:
        return None, 0.0, "empty stream"
    old = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    t0 = time.perf_counter()
    try:
        img = Image.open(BytesIO(blob))
        arr = np.array(img.convert("L"), dtype=np.float64)
        return arr, time.perf_counter() - t0, ""
    except Exception as exc:
        return None, time.perf_counter() - t0, str(exc)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old


def _truncate_stream(blob: bytes, received_fraction: float) -> bytes:
    keep = int(round(len(blob) * max(0.0, min(1.0, received_fraction))))
    return blob[:keep]


def _imcs_received_size(encoded_size: int, measurement_bytes: int, keep_fraction: float) -> int:
    keep_measurements = int(round(measurement_bytes * max(0.0, min(1.0, keep_fraction))))
    return int(encoded_size - measurement_bytes + keep_measurements)


def _measurement_rows(m: int, keep_fraction: float, *, seed: int) -> np.ndarray:
    if keep_fraction >= 1.0:
        return np.arange(m, dtype=int)
    keep = int(math.floor(float(m) * max(0.0, min(1.0, keep_fraction))))
    if keep <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(m, size=keep, replace=False)).astype(int)


def decode_imcs_with_measurement_loss(
    compressed: bytes,
    *,
    keep_fraction: float,
    reconstruction_algorithm: str = "omp",
    mask_seed: int = 2026,
) -> np.ndarray:
    decoder = IMCSDecoder(reconstruction_algorithm=reconstruction_algorithm)
    metadata, measurements = decoder._deserialize(compressed)
    if not metadata["is_2d"] or not metadata.get("block_h", 0):
        raise ValueError("Robustness experiment currently expects blocked 2D IMCS data")

    pad_h, pad_w = metadata["original_shape"]
    content_h, content_w = metadata["content_shape"]
    bh = int(metadata["block_h"])
    bw = int(metadata["block_w"])
    m_per_block = int(metadata["m_row"])
    n_blocks = int(metadata["m_col"])
    b_pixels = bh * bw
    n_blocks_h = pad_h // bh
    n_blocks_w = pad_w // bw
    y_blocks = measurements.reshape(n_blocks, m_per_block)

    Psi_2d = create_2d_basis_matrix(bh, bw, metadata["sparsity_basis"])
    low_frequency_count = int(metadata.get("low_frequency_coeffs", 0))
    low_frequency_positions = zigzag_indices(bh, bw)[:low_frequency_count]
    low_frequency_coefficients = metadata.get("low_frequency_coefficients")
    block_means = metadata.get("block_means")

    shared_phi = None
    if metadata.get("measurement_mode", "shared") == "shared" and m_per_block > 0:
        shared_phi = generate_measurement_matrix(
            m_per_block,
            b_pixels,
            metadata["matrix_type"],
            metadata["seed"],
        )

    X_pad = np.zeros((pad_h, pad_w), dtype=np.float64)
    for idx in range(n_blocks):
        bi = idx // n_blocks_w
        bj = idx % n_blocks_w

        block = np.zeros((bh, bw), dtype=np.float64)
        if low_frequency_coefficients is not None and low_frequency_positions:
            coeff_matrix = np.zeros((bh, bw), dtype=np.float64)
            for pos, value in zip(low_frequency_positions, low_frequency_coefficients[idx]):
                coeff_matrix[pos] = value
            block += idct2(coeff_matrix)
        if block_means is not None:
            block += float(block_means[idx])

        rows = _measurement_rows(m_per_block, keep_fraction, seed=mask_seed + idx)
        if rows.size > 0:
            if shared_phi is None:
                phi_seed = metadata["seed"] + idx
                Phi = generate_measurement_matrix(
                    m_per_block,
                    b_pixels,
                    metadata["matrix_type"],
                    phi_seed,
                )
            else:
                Phi = shared_phi
            Phi_keep = Phi[rows]
            y_keep = y_blocks[idx, rows]
            A_keep = Phi_keep @ Psi_2d.T
            sparsity = max(min(int(rows.size) // 2, b_pixels // 4), 1)
            if reconstruction_algorithm != "omp":
                raise ValueError("Robustness loss decoder currently supports OMP only")
            s = omp(y_keep, A_keep, sparsity)
            block += (Psi_2d.T @ s).reshape(bh, bw, order="F")

        X_pad[bi * bh : (bi + 1) * bh, bj * bw : (bj + 1) * bw] = block

    return X_pad[:content_h, :content_w]


def _metric_row(
    *,
    dataset: str,
    case: BenchmarkCase,
    original: np.ndarray,
    method: str,
    loss_fraction: float,
    received_fraction: float,
    decoded: bool,
    reconstructed: np.ndarray | None,
    encode_time_s: float,
    decode_time_s: float,
    encoded_size_bytes: int,
    received_size_bytes: int,
    notes: str = "",
) -> dict[str, Any]:
    h, w = original.shape[:2]
    if decoded and reconstructed is not None:
        metrics = calculate_compression_metrics(original, reconstructed)
        psnr = float(metrics["psnr"])
        ssim = float(metrics["ssim"])
        mae = float(metrics["mae"])
    else:
        psnr = 0.0
        ssim = 0.0
        mae = 255.0
    return {
        "dataset": dataset,
        "case_key": case.key,
        "input_file": case.path.name,
        "width": int(w),
        "height": int(h),
        "method": method,
        "loss_fraction": float(loss_fraction),
        "received_fraction": float(received_fraction),
        "decoded": int(bool(decoded)),
        "bpp_received": 8.0 * float(received_size_bytes) / float(h * w),
        "psnr_db": psnr,
        "ssim": ssim,
        "mae": mae,
        "encode_time_s": float(encode_time_s),
        "decode_time_s": float(decode_time_s),
        "encoded_size_bytes": int(encoded_size_bytes),
        "received_size_bytes": int(received_size_bytes),
        "notes": notes,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=robustness_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _group_mean(rows: Sequence[dict[str, Any]], method: str, loss: float, key: str) -> float:
    vals = [float(r[key]) for r in rows if r["method"] == method and float(r["loss_fraction"]) == loss]
    return float(np.mean(vals)) if vals else float("nan")


def plot_robustness(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    losses = sorted({float(row["loss_fraction"]) for row in rows})
    colors = ("#2f6fed", "#12b76a", "#f97316", "#a855f7", "#ef4444")

    fig = Figure(figsize=(9.0, 5.2), facecolor="#ffffff")
    ax = fig.add_subplot(1, 1, 1, facecolor="#ffffff")
    for idx, method in enumerate(methods):
        y = [_group_mean(rows, method, loss, "psnr_db") for loss in losses]
        ax.plot(
            [100.0 * loss for loss in losses],
            y,
            marker="o",
            linewidth=1.6,
            color=colors[idx % len(colors)],
            label=method,
        )
    ax.set_title("Устойчивость к потере данных", fontsize=12)
    ax.set_xlabel("Потеря данных, %")
    ax.set_ylabel("PSNR, дБ (0 = поток не декодирован)")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "fig_robustness_psnr_loss.png", dpi=190, bbox_inches="tight")
    fig.clear()

    fig = Figure(figsize=(9.0, 5.2), facecolor="#ffffff")
    ax = fig.add_subplot(1, 1, 1, facecolor="#ffffff")
    for idx, method in enumerate(methods):
        y = [_group_mean(rows, method, loss, "decoded") for loss in losses]
        ax.plot(
            [100.0 * loss for loss in losses],
            y,
            marker="o",
            linewidth=1.6,
            color=colors[idx % len(colors)],
            label=method,
        )
    ax.set_title("Доля успешно декодированных изображений", fontsize=12)
    ax.set_xlabel("Потеря данных, %")
    ax.set_ylabel("Доля декодированных изображений")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "fig_robustness_decodable_loss.png", dpi=190, bbox_inches="tight")
    fig.clear()


def run_robustness_study(
    output_dir: Path,
    *,
    study_name: str,
    cases: Sequence[BenchmarkCase],
    dataset_label: str,
    losses: Sequence[float],
    config: RobustnessConfig,
    clean_output: bool = True,
    verbose: bool = True,
) -> Path:
    study_dir = output_dir / "benchmarks" / study_name
    if clean_output and study_dir.exists():
        import shutil

        shutil.rmtree(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    total = len(cases) * len(losses) * 3
    done = 0
    started = time.perf_counter()
    log("=== Robustness benchmark started ===")
    log(f"Output: {study_dir}")
    log(f"Images: {len(cases)}; losses={', '.join(str(x) for x in losses)}")

    for case_idx, case in enumerate(cases, start=1):
        original = load_image(case.path, verbose=False, preserve_color=False)
        if original is None:
            continue
        h, w = original.shape[:2]
        log(f"\n--- Image {case_idx}/{len(cases)}: {case.key}, size={w}x{h} ---")

        jpeg_blob, jpeg_t_enc = _encode_jpeg(original, config.jpeg_quality)

        encoder = IMCSEncoder(
            compression_ratio=config.compression_ratio,
            seed=config.seed,
            sparsity_basis="dct",
            matrix_type="gaussian",
            block_size=(config.block_edge, config.block_edge),
            measurement_mode="shared",
            measurement_dtype=config.measurement_dtype,
            block_mean_residual=True,
            low_frequency_coeffs=config.low_frequency_coeffs,
        )
        t0 = time.perf_counter()
        imcs_blob = encoder.encode(original)
        imcs_t_enc = time.perf_counter() - t0
        imcs_decoder = IMCSDecoder()
        metadata, _ = imcs_decoder._deserialize(imcs_blob)
        measurement_bytes = int(metadata.get("measurement_data_length", 0))

        for loss in losses:
            received_fraction = 1.0 - float(loss)

            done += 1
            log(
                f"[{done}/{total}] {case.key}: Hybrid IMCS, loss={100*loss:.0f}% "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
            t1 = time.perf_counter()
            recon = decode_imcs_with_measurement_loss(
                imcs_blob,
                keep_fraction=received_fraction,
                mask_seed=config.seed + 10_000,
            )
            t_dec = time.perf_counter() - t1
            received_size = _imcs_received_size(len(imcs_blob), measurement_bytes, received_fraction)
            rows.append(
                _metric_row(
                    dataset=dataset_label,
                    case=case,
                    original=original,
                    method="Hybrid IMCS",
                    loss_fraction=float(loss),
                    received_fraction=received_fraction,
                    decoded=True,
                    reconstructed=recon,
                    encode_time_s=imcs_t_enc,
                    decode_time_s=t_dec,
                    encoded_size_bytes=len(imcs_blob),
                    received_size_bytes=received_size,
                    notes=f"measurement_bytes={measurement_bytes}",
                )
            )

            for method, decoder_fn in (
                ("JPEG strict", _decode_jpeg_strict),
                ("JPEG tolerant", _decode_jpeg_tolerant),
            ):
                done += 1
                log(f"[{done}/{total}] {case.key}: {method}, loss={100*loss:.0f}%")
                damaged = _truncate_stream(jpeg_blob, received_fraction)
                recon_jpeg, t_dec_jpeg, note = decoder_fn(damaged)
                decoded = recon_jpeg is not None and recon_jpeg.shape == original.shape
                rows.append(
                    _metric_row(
                        dataset=dataset_label,
                        case=case,
                        original=original,
                        method=method,
                        loss_fraction=float(loss),
                        received_fraction=received_fraction,
                        decoded=decoded,
                        reconstructed=recon_jpeg if decoded else None,
                        encode_time_s=jpeg_t_enc,
                        decode_time_s=t_dec_jpeg,
                        encoded_size_bytes=len(jpeg_blob),
                        received_size_bytes=len(damaged),
                        notes=note,
                    )
                )

    _write_csv(study_dir / "robustness_long.csv", rows)
    plot_robustness(rows, study_dir)
    log(f"=== Robustness benchmark finished: {len(rows)} rows ===")
    log(f"Output: {study_dir}")
    return study_dir
