import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from scipy.fftpack import dct


def create_convergence_plot_from_decoder(
    decoder,
    compressed: bytes,
    original: np.ndarray,
    algorithm: str,
    output_subdir: Path,
):
    """
    Создает визуализацию сходимости из декодера.

    Args:
        decoder: IMCSDecoder с сохраненной историей
        compressed: Сжатые данные
        original: Исходное изображение
        algorithm: Алгоритм восстановления
        output_subdir: Директория для сохранения
    """
    from imcs.utils import generate_measurement_matrix

    history = decoder.last_history
    residuals = decoder.last_residuals
    metadata, measurements = decoder._deserialize(compressed)

    if metadata["is_2d"]:
        n_row, n_col = metadata["original_shape"]
        m_total = metadata["m_row"]
        n_total = n_row * n_col

        Phi = generate_measurement_matrix(
            m_total, n_total, metadata["matrix_type"], metadata["seed"]
        )

        def create_dct_matrix(n):
            return dct(np.eye(n), axis=0, norm="ortho")

        Psi_row = create_dct_matrix(n_row)
        Psi_col = create_dct_matrix(n_col)
        Psi_2d = np.kron(Psi_col, Psi_row)

        A = Phi @ Psi_2d.T
        y = measurements

        original_flat = original.flatten()
        Psi = dct(np.eye(n_row * n_col), norm="ortho")
        s_true = Psi @ original_flat

        plot_convergence_path(
            history,
            residuals,
            s_true,
            A,
            y,
            decoder.lambda_param,
            algorithm.upper(),
            output_dir=str(output_subdir),
            filename_prefix=f"convergence_{algorithm}",
        )


def plot_convergence_path(
    history: List[np.ndarray],
    residuals: List[float],
    s_true: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    lambda_param: float,
    algorithm: str,
    output_dir: str = "visualization/output",
    filename_prefix: str = "",
):
    """
    Визуализирует путь сходимости алгоритма на ландшафте функции потерь.

    Args:
        history: Список состояний s на каждой итерации
        residuals: Список значений residual на каждой итерации
        s_true: Истинный разреженный вектор (для определения значимых компонент)
        A: Sensing matrix (для вычисления функции потерь)
        y: Вектор измерений (для вычисления функции потерь)
        lambda_param: Параметр регуляризации
        algorithm: Название алгоритма ('OMP', 'ISTA', 'SA')
        output_dir: Папка для сохранения
        filename_prefix: Префикс для имени файла (например, имя изображения)
    """
    if len(history) < 2:
        print(f"   Недостаточно итераций для визуализации {algorithm}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. График сходимости residual
    ax1 = axes[0]
    ax1.semilogy(residuals, "b-", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Итерация", fontsize=12)
    ax1.set_ylabel("||y - A·s||₂ (log scale)", fontsize=12)
    ax1.set_title(f"{algorithm}: Сходимость", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 2. Эволюция sparsity
    ax2 = axes[1]
    sparsities: list = [np.sum(np.abs(s) > 1e-6) for s in history]
    ax2.plot(sparsities, "g-", linewidth=2)
    ax2.set_xlabel("Итерация", fontsize=12)
    ax2.set_ylabel("Количество ненулевых коэффициентов", fontsize=12)
    ax2.set_title(f"{algorithm}: Эволюция sparsity", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. 2D путь к минимуму
    ax3 = axes[2]

    # Находим два самых значимых компонента
    s_final = history[-1]
    nonzero_indices = np.where(np.abs(s_final) > 1e-10)[0]

    if len(nonzero_indices) >= 2:
        # Берем два компонента с максимальной амплитудой изменения
        # "Изменчивость" = размах (range) = max(s[i]) - min(s[i]) по всем итерациям
        # Это показывает, какие коэффициенты активно меняются во время работы алгоритма
        variations = []
        for idx in nonzero_indices:
            values = [s[idx] for s in history]
            variation = np.max(values) - np.min(values)  # Размах изменения
            variations.append((idx, variation))

        variations.sort(key=lambda x: x[1], reverse=True)
        idx1, idx2 = variations[0][0], variations[1][0]

        # Траектория алгоритма в пространстве (s[idx1], s[idx2])
        traj_s1 = [s[idx1] for s in history]
        traj_s2 = [s[idx2] for s in history]

        # Границы для сетки
        s1_min, s1_max = min(traj_s1) - 2, max(traj_s1) + 2
        s2_min, s2_max = min(traj_s2) - 2, max(traj_s2) + 2

        # Создаем сетку для вычисления функции потерь
        # На каждой точке сетки вычисляем L(s), где:
        # s = [0, ..., 0, s[idx1], 0, ..., 0, s[idx2], 0, ..., 0]
        s1_grid = np.linspace(s1_min, s1_max, 80)
        s2_grid = np.linspace(s2_min, s2_max, 80)
        S1, S2 = np.meshgrid(s1_grid, s2_grid)

        # Вычисляем функцию потерь на сетке
        Z = np.zeros_like(S1)
        s_test = np.zeros_like(s_final)
        for i in range(S1.shape[0]):
            for j in range(S1.shape[1]):
                s_test[idx1] = S1[i, j]
                s_test[idx2] = S2[i, j]
                # Функция потерь: L(s) = ||y - A·s||₂² + λ||s||₁
                residual_norm = np.linalg.norm(y - A @ s_test)
                l1_norm = np.linalg.norm(s_test, 1)
                Z[i, j] = residual_norm**2 + lambda_param * l1_norm

        # Рисуем контуры функции потерь (ландшафт)
        contourf = ax3.contourf(S1, S2, Z, levels=25, cmap="viridis", alpha=0.6)
        ax3.contour(S1, S2, Z, levels=10, colors="black", alpha=0.15, linewidths=0.5)

        # Рисуем траекторию алгоритма поверх контуров
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

        # Colorbar для функции потерь
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

    # Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    if filename_prefix:
        filename = f"{filename_prefix}.png"
    else:
        filename = f"{algorithm.lower()}_convergence.png"

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"   Сохранено: {output_path}")
