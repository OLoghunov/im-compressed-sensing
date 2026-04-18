from __future__ import annotations

import os
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
from scipy.fftpack import dct

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imcs.utils import (
    create_2d_basis_matrix,
    fista,
    generate_measurement_matrix,
    ista,
    omp,
    simulated_annealing,
)


def create_representative_block_convergence_plot(
    original: np.ndarray,
    reconstructed: np.ndarray,
    algorithm: str,
    output_subdir: Path,
    *,
    basis: str,
    matrix_type: str,
    measurement_mode: str,
    compression_ratio: float,
    block_size: tuple[int, int] | None,
    seed: int,
    channel_label: str | None = None,
) -> None:
    block, block_idx, block_note = _select_representative_block(
        original, reconstructed, block_size
    )
    bh, bw = block.shape
    b_pixels = bh * bw
    m_per_block = max(1, int(b_pixels * compression_ratio))
    phi_seed = seed if measurement_mode == "shared" else seed + block_idx
    phi = generate_measurement_matrix(m_per_block, b_pixels, matrix_type, phi_seed)
    psi_2d = create_2d_basis_matrix(bh, bw, basis)
    a_matrix = phi @ psi_2d.T
    x_vec = block.reshape(-1, order="F")
    y = phi @ x_vec
    s_true = psi_2d @ x_vec

    adaptive_lambda = 10.0 * (b_pixels / 1000.0)
    sparsity = max(min(m_per_block // 2, b_pixels // 4), 1)
    algo_key = algorithm.lower()
    if algo_key == "omp":
        _, history, residuals = omp(y, a_matrix, sparsity, return_history=True)
    elif algo_key == "ista":
        _, history, residuals = ista(
            y, a_matrix, adaptive_lambda, max_iter=1000, return_history=True
        )
    elif algo_key == "fista":
        _, history, residuals = fista(
            y, a_matrix, adaptive_lambda, max_iter=1000, return_history=True
        )
    elif algo_key == "sa":
        _, history, residuals = simulated_annealing(
            y, a_matrix, adaptive_lambda, max_iter=1000, return_history=True
        )
    else:
        raise ValueError(f"Unsupported algorithm for convergence plot: {algorithm}")

    suffix = f"_{channel_label.lower()}" if channel_label else ""
    filename_prefix = f"convergence_{algorithm}{suffix}"
    label = f"{algorithm.upper()} [{block_note}]"
    plot_convergence_path(
        history,
        residuals,
        s_true,
        a_matrix,
        y,
        adaptive_lambda,
        label,
        output_dir=str(output_subdir),
        filename_prefix=filename_prefix,
    )


def _select_representative_block(
    original: np.ndarray,
    reconstructed: np.ndarray,
    block_size: tuple[int, int] | None,
) -> tuple[np.ndarray, int, str]:
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if block_size is None:
        return original, 0, "full frame"

    bh, bw = block_size
    h, w = original.shape
    n_blocks_h = (h + bh - 1) // bh
    n_blocks_w = (w + bw - 1) // bw

    best_idx = 0
    best_bi = 0
    best_bj = 0
    best_score = -1.0
    idx = 0
    for bi in range(n_blocks_h):
        for bj in range(n_blocks_w):
            block_orig = original[
                bi * bh : min((bi + 1) * bh, h),
                bj * bw : min((bj + 1) * bw, w),
            ]
            block_recon = reconstructed[
                bi * bh : min((bi + 1) * bh, h),
                bj * bw : min((bj + 1) * bw, w),
            ]
            score = float(np.mean(np.abs(block_orig - block_recon)))
            if score > best_score:
                best_score = score
                best_idx = idx
                best_bi = bi
                best_bj = bj
            idx += 1

    selected = np.zeros((bh, bw), dtype=np.float64)
    block = original[
        best_bi * bh : min((best_bi + 1) * bh, h),
        best_bj * bw : min((best_bj + 1) * bw, w),
    ]
    selected[: block.shape[0], : block.shape[1]] = block
    return selected, best_idx, f"block {best_idx}"


def plot_convergence_path(
    history: List[np.ndarray],
    residuals: List[float],
    s_true: np.ndarray,
    a_matrix: np.ndarray,
    y: np.ndarray,
    lambda_param: float,
    algorithm: str,
    output_dir: str = "output",
    filename_prefix: str = "",
) -> None:
    if len(history) < 2:
        print(f"   Недостаточно итераций для визуализации {algorithm}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    ax1.semilogy(residuals, "b-", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Итерация", fontsize=12)
    ax1.set_ylabel("||y - A·s||₂ (log scale)", fontsize=12)
    ax1.set_title(f"{algorithm}: Сходимость", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    sparsities: list = [np.sum(np.abs(s) > 1e-6) for s in history]
    ax2.plot(sparsities, "g-", linewidth=2)
    ax2.set_xlabel("Итерация", fontsize=12)
    ax2.set_ylabel("Количество ненулевых коэффициентов", fontsize=12)
    ax2.set_title(f"{algorithm}: Эволюция sparsity", fontsize=14)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    s_final = history[-1]
    nonzero_indices = np.where(np.abs(s_final) > 1e-10)[0]

    if len(nonzero_indices) >= 2:
        variations = []
        for idx in nonzero_indices:
            values = [s[idx] for s in history]
            variation = np.max(values) - np.min(values)
            variations.append((idx, variation))

        variations.sort(key=lambda x: x[1], reverse=True)
        idx1, idx2 = variations[0][0], variations[1][0]

        traj_s1 = [s[idx1] for s in history]
        traj_s2 = [s[idx2] for s in history]

        s1_min, s1_max = min(traj_s1) - 2, max(traj_s1) + 2
        s2_min, s2_max = min(traj_s2) - 2, max(traj_s2) + 2

        s1_grid = np.linspace(s1_min, s1_max, 80)
        s2_grid = np.linspace(s2_min, s2_max, 80)
        s1_mesh, s2_mesh = np.meshgrid(s1_grid, s2_grid)

        z = np.zeros_like(s1_mesh)
        s_test = np.zeros_like(s_final)
        for i in range(s1_mesh.shape[0]):
            for j in range(s1_mesh.shape[1]):
                s_test[idx1] = s1_mesh[i, j]
                s_test[idx2] = s2_mesh[i, j]
                residual_norm = np.linalg.norm(y - a_matrix @ s_test)
                l1_norm = np.linalg.norm(s_test, 1)
                z[i, j] = residual_norm**2 + lambda_param * l1_norm

        contourf = ax3.contourf(s1_mesh, s2_mesh, z, levels=25, cmap="viridis", alpha=0.6)
        ax3.contour(s1_mesh, s2_mesh, z, levels=10, colors="black", alpha=0.15, linewidths=0.5)

        step = max(1, len(traj_s1) // 30)
        traj_s1_plot = traj_s1[::step]
        traj_s2_plot = traj_s2[::step]

        ax3.plot(traj_s1, traj_s2, "r-", linewidth=2, alpha=0.8, zorder=10)
        ax3.scatter(
            traj_s1_plot,
            traj_s2_plot,
            c=range(len(traj_s1_plot)),
            cmap="coolwarm",
            s=30,
            edgecolors="black",
            linewidth=0.5,
            zorder=11,
        )
        ax3.plot(
            traj_s1[0],
            traj_s2[0],
            "o",
            color="lime",
            markersize=10,
            label="Старт",
            zorder=12,
            markeredgecolor="black",
        )
        ax3.plot(
            traj_s1[-1],
            traj_s2[-1],
            "s",
            color="red",
            markersize=8,
            label="Финиш",
            zorder=12,
            markeredgecolor="black",
        )

        plt.colorbar(contourf, ax=ax3, label="Функция потерь", fraction=0.046, pad=0.04)

        ax3.set_xlabel(f"s[{idx1}]", fontsize=12)
        ax3.set_ylabel(f"s[{idx2}]", fontsize=12)
        ax3.set_title(f"{algorithm}: Ландшафт функции потерь", fontsize=14)
        ax3.legend(fontsize=10, loc="best")
        ax3.grid(True, alpha=0.2)
    else:
        ax3.text(
            0.5,
            0.5,
            "Недостаточно ненулевых\nкоэффициентов",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title(f"{algorithm}: Путь к минимуму", fontsize=14)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{filename_prefix}.png" if filename_prefix else f"{algorithm.lower()}_convergence.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"   Сохранено: {output_path}")

