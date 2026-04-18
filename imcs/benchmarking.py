from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from matplotlib.figure import Figure

from imcs.pipeline import (
    DEFAULT_BASIS,
    DEFAULT_MATRIX_TYPE,
    DEFAULT_MEASUREMENT_MODE,
    run_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "examples" / "input"


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    path: Path
    category: str
    description: str


def default_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            key="smooth_gradient",
            path=INPUT_DIR / "02_gradient_h.png",
            category="smooth",
            description="Smooth horizontal gradient",
        ),
        BenchmarkCase(
            key="edge_rich_cross",
            path=INPUT_DIR / "10_cross.png",
            category="edge-rich",
            description="Geometric cross with sharp transitions",
        ),
        BenchmarkCase(
            key="textured_random",
            path=INPUT_DIR / "18_texture_random_32x32.png",
            category="textured",
            description="Random texture with weak sparsity in DCT",
        ),
        BenchmarkCase(
            key="large_radial",
            path=INPUT_DIR / "20_radial_gradient_128x128.png",
            category="large",
            description="Larger synthetic image for scalability checks",
        ),
    ]


def default_block_sizes() -> tuple[int, ...]:
    return (8, 16, 32)


def default_compression_ratios() -> tuple[float, ...]:
    return (0.3, 0.5, 0.7)


def default_algorithms() -> tuple[str, ...]:
    return ("omp", "ista")


def benchmark_fieldnames() -> list[str]:
    return [
        "case_key",
        "case_category",
        "input_file",
        "width",
        "height",
        "block_size",
        "compression_ratio",
        "algorithm",
        "basis",
        "matrix_type",
        "measurement_mode",
        "encode_time_s",
        "decode_time_s",
        "decode_reconstruction_s",
        "decode_phi_build_s",
        "decode_sensing_matrix_build_s",
        "decode_inverse_transform_s",
        "decode_parallel_workers",
        "psnr_db",
        "ssim",
        "mae",
        "compressed_size_bytes",
        "output_dir",
    ]


def run_block_benchmark_study(
    output_dir: Path,
    study_name: str = "block_study",
    cases: Sequence[BenchmarkCase] | None = None,
    block_sizes: Sequence[int] | None = None,
    compression_ratios: Sequence[float] | None = None,
    algorithms: Sequence[str] | None = None,
    basis: str = DEFAULT_BASIS,
    matrix_type: str = DEFAULT_MATRIX_TYPE,
    measurement_mode: str = DEFAULT_MEASUREMENT_MODE,
    collect_decode_profile: bool = False,
    parallel_block_workers: int | None = None,
    verbose: bool = True,
) -> Path:
    benchmark_dir = output_dir / "benchmarks" / study_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    cases = list(default_benchmark_cases() if cases is None else cases)
    block_sizes = tuple(default_block_sizes() if block_sizes is None else block_sizes)
    compression_ratios = tuple(
        default_compression_ratios() if compression_ratios is None else compression_ratios
    )
    algorithms = tuple(default_algorithms() if algorithms is None else algorithms)

    rows: list[dict[str, object]] = []

    for case in cases:
        for block_size in block_sizes:
            for ratio in compression_ratios:
                for algorithm in algorithms:
                    ratio_str = str(ratio).replace(".", "p")
                    output_name = (
                        f"{case.path.stem}_b{block_size}_r{ratio_str}_{algorithm}_"
                        f"{basis}_{matrix_type}_{measurement_mode}"
                    )
                    result = run_image(
                        case.path,
                        benchmark_dir,
                        compression_ratio=float(ratio),
                        algorithm=algorithm,
                        block_edge=int(block_size),
                        force_full_frame=False,
                        visualize_convergence=False,
                        verbose=verbose,
                        basis=basis,
                        matrix_type=matrix_type,
                        measurement_mode=measurement_mode,
                        output_name=output_name,
                        collect_decode_profile=collect_decode_profile,
                        parallel_block_workers=parallel_block_workers,
                    )
                    if result is None:
                        continue

                    height, width = result.original.shape[:2]
                    rows.append(
                        {
                            "case_key": case.key,
                            "case_category": case.category,
                            "input_file": case.path.name,
                            "width": int(width),
                            "height": int(height),
                            "block_size": int(block_size),
                            "compression_ratio": float(ratio),
                            "algorithm": algorithm,
                            "basis": basis,
                            "matrix_type": matrix_type,
                            "measurement_mode": measurement_mode,
                            "encode_time_s": round(result.t_encode, 6),
                            "decode_time_s": round(result.t_decode, 6),
                            "decode_reconstruction_s": round(
                                float((result.decode_profile or {}).get("reconstruction_s", 0.0)), 6
                            ),
                            "decode_phi_build_s": round(
                                float((result.decode_profile or {}).get("phi_build_s", 0.0)), 6
                            ),
                            "decode_sensing_matrix_build_s": round(
                                float(
                                    (result.decode_profile or {}).get(
                                        "sensing_matrix_build_s", 0.0
                                    )
                                ),
                                6,
                            ),
                            "decode_inverse_transform_s": round(
                                float(
                                    (result.decode_profile or {}).get(
                                        "inverse_transform_s", 0.0
                                    )
                                ),
                                6,
                            ),
                            "decode_parallel_workers": int(
                                (result.decode_profile or {}).get("parallel_workers", 1)
                            ),
                            "psnr_db": round(result.metrics["psnr"], 6),
                            "ssim": round(result.metrics["ssim"], 6),
                            "mae": round(result.metrics["mae"], 6),
                            "compressed_size_bytes": int(len(result.compressed)),
                            "output_dir": str(result.output_subdir),
                        }
                    )

    _write_benchmark_outputs(
        benchmark_dir,
        rows,
        cases=cases,
        block_sizes=block_sizes,
        compression_ratios=compression_ratios,
        algorithms=algorithms,
        basis=basis,
        matrix_type=matrix_type,
        measurement_mode=measurement_mode,
    )
    return benchmark_dir


def _write_benchmark_outputs(
    benchmark_dir: Path,
    rows: list[dict[str, object]],
    *,
    cases: Sequence[BenchmarkCase],
    block_sizes: Sequence[int],
    compression_ratios: Sequence[float],
    algorithms: Sequence[str],
    basis: str,
    matrix_type: str,
    measurement_mode: str,
) -> None:
    csv_path = benchmark_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=benchmark_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "cases": [
            {
                "key": case.key,
                "path": str(case.path),
                "category": case.category,
                "description": case.description,
            }
            for case in cases
        ],
        "block_sizes": list(block_sizes),
        "compression_ratios": list(compression_ratios),
        "algorithms": list(algorithms),
        "basis": basis,
        "matrix_type": matrix_type,
        "measurement_mode": measurement_mode,
    }
    with open(benchmark_dir / "benchmark_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _write_summary_markdown(benchmark_dir / "summary.md", rows)
    _write_summary_plot(benchmark_dir / "summary_plot.png", rows, block_sizes, algorithms)


def _write_summary_markdown(summary_path: Path, rows: list[dict[str, object]]) -> None:
    grouped = _group_average_rows(rows, ("block_size", "algorithm"))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# IMCS Benchmark Summary\n\n")
        f.write("## Result schema\n\n")
        f.write(", ".join(benchmark_fieldnames()) + "\n\n")
        f.write("## Average by block size and algorithm\n\n")
        f.write("| Block | Algorithm | Avg PSNR (dB) | Avg SSIM | Avg Decode (s) |\n")
        f.write("|-------|-----------|---------------|----------|----------------|\n")
        for row in grouped:
            f.write(
                f"| {row['block_size']} | {row['algorithm']} | {row['psnr_db']:.2f} | "
                f"{row['ssim']:.4f} | {row['decode_time_s']:.4f} |\n"
            )


def _write_summary_plot(
    output_path: Path,
    rows: list[dict[str, object]],
    block_sizes: Sequence[int],
    algorithms: Sequence[str],
) -> None:
    averages = _group_average_rows(rows, ("block_size", "algorithm"))
    average_map = {
        (int(row["block_size"]), str(row["algorithm"])): row for row in averages
    }

    fig = Figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for algorithm in algorithms:
        psnr_values = [
            float(average_map[(block_size, algorithm)]["psnr_db"])
            for block_size in block_sizes
            if (block_size, algorithm) in average_map
        ]
        decode_values = [
            float(average_map[(block_size, algorithm)]["decode_time_s"])
            for block_size in block_sizes
            if (block_size, algorithm) in average_map
        ]
        valid_blocks = [
            block_size for block_size in block_sizes if (block_size, algorithm) in average_map
        ]
        ax1.plot(valid_blocks, psnr_values, marker="o", label=algorithm.upper())
        ax2.plot(valid_blocks, decode_values, marker="o", label=algorithm.upper())

    ax1.set_title("Average PSNR by block size")
    ax1.set_xlabel("Block size")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Average decode time by block size")
    ax2.set_xlabel("Block size")
    ax2.set_ylabel("Decode time (s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")


def _group_average_rows(
    rows: Iterable[dict[str, object]],
    keys: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)

    result: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        result_row: dict[str, object] = {name: value for name, value in zip(keys, key)}
        result_row["psnr_db"] = float(np.mean([float(row["psnr_db"]) for row in group_rows]))
        result_row["ssim"] = float(np.mean([float(row["ssim"]) for row in group_rows]))
        result_row["decode_time_s"] = float(
            np.mean([float(row["decode_time_s"]) for row in group_rows])
        )
        result.append(result_row)
    return result
