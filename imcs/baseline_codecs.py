"""
Baseline image codecs for RD comparison against IMCS.

The module keeps all codecs on the same measurement scale:
PSNR/SSIM/MAE are calculated against the same input array, while bpp is
8 * compressed_stream_bytes / (height * width).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from imcs.benchmarking import BenchmarkCase, default_benchmark_cases
from imcs.cli import load_image
from imcs.utils import calculate_compression_metrics


@dataclass(frozen=True)
class CodecSpec:
    name: str
    label: str
    param_name: str
    param_values: tuple[float, ...]
    pil_format: str


def default_codec_specs(
    *,
    include_webp: bool = True,
    include_jpeg2000: bool = True,
) -> tuple[CodecSpec, ...]:
    specs = [
        CodecSpec(
            name="jpeg",
            label="JPEG",
            param_name="quality",
            param_values=(10, 20, 30, 40, 50, 60, 70, 80, 90, 95),
            pil_format="JPEG",
        )
    ]
    if include_webp:
        specs.append(
            CodecSpec(
                name="webp",
                label="WebP",
                param_name="quality",
                param_values=(10, 20, 30, 40, 50, 60, 70, 80, 90, 95),
                pil_format="WEBP",
            )
        )
    if include_jpeg2000:
        specs.append(
            CodecSpec(
                name="jpeg2000",
                label="JPEG2000",
                param_name="quality",
                # Pillow's JPEG2000 encoder accepts this as PSNR quality layers.
                param_values=(25, 30, 35, 40, 45, 50),
                pil_format="JPEG2000",
            )
        )
    return tuple(specs)


def jpeg_baseline_fieldnames() -> list[str]:
    return [
        "case_key",
        "case_category",
        "input_file",
        "width",
        "height",
        "channels",
        "codec",
        "codec_label",
        "param_name",
        "param_value",
        "encode_time_s",
        "decode_time_s",
        "compressed_size_bytes",
        "bpp",
        "psnr_db",
        "ssim",
        "mae",
    ]


def _array_to_pil_u8(arr: np.ndarray) -> Image.Image:
    u8 = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 255.0).astype(np.uint8)
    if u8.ndim == 2:
        return Image.fromarray(u8, mode="L")
    if u8.ndim == 3 and u8.shape[2] == 3:
        return Image.fromarray(u8, mode="RGB")
    raise ValueError(f"Baseline codecs support 2D grayscale or HxWx3 RGB, got {u8.shape}")


def _pil_to_float_array(img: Image.Image, *, want_color: bool) -> np.ndarray:
    if want_color:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.float64)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.float64)


def _codec_save_options(spec: CodecSpec, param_value: float) -> dict[str, Any]:
    if spec.name == "jpeg":
        return {
            "quality": int(round(param_value)),
            "optimize": False,
            "progressive": False,
        }
    if spec.name == "webp":
        return {
            "quality": int(round(param_value)),
            "method": 4,
            "lossless": False,
        }
    if spec.name == "jpeg2000":
        return {
            "quality_mode": "dB",
            "quality_layers": [float(param_value)],
            "irreversible": True,
        }
    raise ValueError(f"Unsupported codec: {spec.name}")


def _encode_decode_once(
    src_pil: Image.Image,
    spec: CodecSpec,
    param_value: float,
) -> tuple[bytes, Image.Image, float, float]:
    options = _codec_save_options(spec, param_value)
    buf = BytesIO()
    t0 = time.perf_counter()
    src_pil.save(buf, format=spec.pil_format, **options)
    t_enc = time.perf_counter() - t0
    blob = buf.getvalue()

    t1 = time.perf_counter()
    decoded = Image.open(BytesIO(blob)).copy()
    t_dec = time.perf_counter() - t1
    return blob, decoded, t_enc, t_dec


def benchmark_codec_once(
    original: np.ndarray,
    spec: CodecSpec,
    param_value: float,
    *,
    warmup_roundtrips: int = 2,
) -> dict[str, Any]:
    """Encode/decode one codec parameter point fully in memory."""

    want_color = original.ndim == 3
    src_pil = _array_to_pil_u8(original)

    for _ in range(max(0, warmup_roundtrips)):
        _encode_decode_once(src_pil, spec, param_value)

    blob, decoded, t_enc, t_dec = _encode_decode_once(src_pil, spec, param_value)
    recon = _pil_to_float_array(decoded, want_color=want_color)

    if recon.shape != original.shape:
        raise RuntimeError(f"Shape mismatch after {spec.label}: {recon.shape} vs {original.shape}")

    metrics = calculate_compression_metrics(original, recon)
    h, w = original.shape[:2]
    bpp = 8.0 * float(len(blob)) / float(h * w)

    return {
        "encode_time_s": float(t_enc),
        "decode_time_s": float(t_dec),
        "compressed_size_bytes": int(len(blob)),
        "bpp": float(bpp),
        "psnr_db": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "mae": float(metrics["mae"]),
    }


def benchmark_jpeg_once(
    original: np.ndarray,
    quality: int,
    *,
    warmup_roundtrips: int = 2,
) -> dict[str, Any]:
    """Backward-compatible wrapper used by older benchmark scripts."""

    jpeg = default_codec_specs(include_webp=False, include_jpeg2000=False)[0]
    return benchmark_codec_once(
        original,
        jpeg,
        float(quality),
        warmup_roundtrips=warmup_roundtrips,
    )


def run_jpeg_baseline_study(
    output_dir: Path,
    study_name: str = "jpeg_baseline",
    cases: Sequence[BenchmarkCase] | None = None,
    qualities: Sequence[int] | None = None,
    color: bool = False,
    verbose: bool = True,
) -> Path:
    """Legacy JPEG-only benchmark used by benchmark_jpeg.py."""

    benchmark_dir = output_dir / "benchmarks" / study_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    cases = list(default_benchmark_cases() if cases is None else cases)
    quality_values = tuple(
        (10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
        if qualities is None
        else tuple(int(q) for q in qualities)
    )
    jpeg_spec = default_codec_specs(include_webp=False, include_jpeg2000=False)[0]

    rows: list[dict[str, object]] = []
    for case in cases:
        if not case.path.is_file():
            if verbose:
                print(f"Skip missing file: {case.path}")
            continue
        original = load_image(case.path, verbose=False, preserve_color=color)
        if original is None:
            continue
        h, w = original.shape[:2]
        ch = 3 if original.ndim == 3 else 1

        for q in quality_values:
            stats = benchmark_codec_once(original, jpeg_spec, float(q), warmup_roundtrips=2)
            rows.append(
                {
                    "case_key": case.key,
                    "case_category": case.category,
                    "input_file": case.path.name,
                    "width": int(w),
                    "height": int(h),
                    "channels": int(ch),
                    "codec": jpeg_spec.name,
                    "codec_label": jpeg_spec.label,
                    "param_name": jpeg_spec.param_name,
                    "param_value": int(q),
                    "encode_time_s": round(stats["encode_time_s"], 6),
                    "decode_time_s": round(stats["decode_time_s"], 6),
                    "compressed_size_bytes": int(stats["compressed_size_bytes"]),
                    "bpp": round(stats["bpp"], 6),
                    "psnr_db": round(stats["psnr_db"], 6),
                    "ssim": round(stats["ssim"], 6),
                    "mae": round(stats["mae"], 6),
                }
            )
            if verbose:
                print(
                    f"  {case.key} JPEG q={q}: PSNR={stats['psnr_db']:.2f} dB, "
                    f"BPP={stats['bpp']:.3f}, enc={stats['encode_time_s']:.4f}s, "
                    f"dec={stats['decode_time_s']:.4f}s"
                )

    _write_jpeg_outputs(benchmark_dir, rows, cases, quality_values, color)
    return benchmark_dir


def _write_jpeg_outputs(
    benchmark_dir: Path,
    rows: list[dict[str, object]],
    cases: Sequence[BenchmarkCase],
    qualities: tuple[int, ...],
    color: bool,
) -> None:
    with open(benchmark_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=jpeg_baseline_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "codec": "jpeg",
        "qualities": list(qualities),
        "color_mode": "rgb" if color else "grayscale",
        "cases": [
            {
                "key": c.key,
                "path": str(c.path),
                "category": c.category,
                "description": c.description,
            }
            for c in cases
        ],
    }
    with open(benchmark_dir / "benchmark_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _write_jpeg_rd_plot(benchmark_dir / "jpeg_psnr_vs_bpp.png", rows)
    _write_jpeg_summary_md(benchmark_dir / "summary.md", rows)


def _write_jpeg_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# JPEG baseline (Pillow)\n\n")
        f.write("Колонки: " + ", ".join(jpeg_baseline_fieldnames()) + "\n\n")
        if rows:
            mean_psnr = float(np.mean([float(r["psnr_db"]) for r in rows]))
            f.write(f"- mean PSNR: {mean_psnr:.2f} dB\n")


def _write_jpeg_rd_plot(output_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_key"]), []).append(row)

    fig = Figure(figsize=(10, 6), facecolor="#ffffff")
    ax = fig.add_subplot(1, 1, 1, facecolor="#ffffff")
    for key in sorted(by_case.keys()):
        pts = sorted(by_case[key], key=lambda x: float(x["param_value"]))
        bpp = [float(p["bpp"]) for p in pts]
        psnr = [min(float(p["psnr_db"]), 99.0) for p in pts]
        ax.plot(bpp, psnr, marker="o", linewidth=1.5, label=key)

    ax.set_title("JPEG: PSNR от битрейта", fontsize=12)
    ax.set_xlabel("Бит на пиксель (bpp)", fontsize=10)
    ax.set_ylabel("PSNR, дБ", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
