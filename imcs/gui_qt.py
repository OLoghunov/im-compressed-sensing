from __future__ import annotations

import contextlib
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Optional

import numpy as np
from PIL import Image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from imcs.analytics import AnalyticsPayload, build_analytics_payload
from imcs.pipeline import (
    DEFAULT_ALGORITHM,
    DEFAULT_BASIS,
    DEFAULT_BLOCK_EDGE,
    DEFAULT_CHROMA_RATIO_SCALE,
    DEFAULT_COLOR_MODE,
    DEFAULT_MATRIX_TYPE,
    DEFAULT_MEASUREMENT_DTYPE,
    DEFAULT_MEASUREMENT_MODE,
    DEFAULT_RATIO,
    run_color_image,
    run_image,
    run_signal,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "examples" / "input"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
_SIGNAL_SUFFIXES = {".npy", ".txt"}


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def _is_signal_path(path: Path) -> bool:
    return path.suffix.lower() in _SIGNAL_SUFFIXES


class _QueueLogWriter:
    def __init__(self, result_queue) -> None:
        self.result_queue = result_queue
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.result_queue.put({"status": "log", "message": line})
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.result_queue.put({"status": "log", "message": self._buffer})
        self._buffer = ""


def _run_processing_task(task: dict, result_queue) -> None:
    log_writer = _QueueLogWriter(result_queue)
    try:
        task_type = task["task_type"]
        kwargs = dict(task["kwargs"])
        kwargs["verbose"] = True
        with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
            if task_type == "image":
                result = run_image(**kwargs)
            elif task_type == "color_image":
                result = run_color_image(**kwargs)
            elif task_type == "signal":
                result = run_signal(**kwargs)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        log_writer.flush()
        result_queue.put({"status": "ok", "task_type": task_type, "result": result})
    except Exception as exc:
        log_writer.flush()
        result_queue.put({"status": "error", "message": str(exc)})


def _run_analytics_task(task: dict, result_queue) -> None:
    try:
        payload = build_analytics_payload(task["result"])
        result_queue.put({"status": "ok", "payload": payload})
    except Exception as exc:
        result_queue.put({"status": "error", "message": str(exc)})


class PreviewCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(8.8, 5.6), dpi=100, facecolor="#f5f7fb")
        super().__init__(self.figure)
        self.ax_input = self.figure.add_subplot(121, facecolor="#ffffff")
        self.ax_result = self.figure.add_subplot(122, facecolor="#ffffff")
        self.figure.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, wspace=0.08)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.show_empty()

    def _style_axis(self, ax, title: str) -> None:
        ax.clear()
        ax.set_title(title, fontsize=11, pad=10, color="#202734")
        ax.tick_params(colors="#667085", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#d0d7e2")

    def _draw_placeholder(self, ax, title: str, text: str) -> None:
        self._style_axis(ax, title)
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color="#7a8599",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_image(self, ax, array: np.ndarray, title: str) -> None:
        self._style_axis(ax, title)
        if array.ndim == 3:
            ax.imshow(np.clip(array, 0, 255).astype(np.uint8))
        else:
            ax.imshow(np.clip(array, 0, 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_signal(self, ax, signal: np.ndarray, title: str) -> None:
        self._style_axis(ax, title)
        ax.plot(signal, color="#2f6fed", lw=1.4)
        ax.grid(True, alpha=0.35, color="#d8deea")

    def show_empty(self) -> None:
        self._draw_placeholder(self.ax_input, "Исходные данные", "Выберите файл")
        self._draw_placeholder(self.ax_result, "После IMCS", "Запустите обработку")
        self.draw_idle()

    def show_input_path(self, path: Path) -> None:
        if _is_image_path(path):
            img = Image.open(path)
            if img.mode in {"RGB", "RGBA"}:
                arr = np.array(img.convert("RGB"), dtype=np.uint8)
            else:
                arr = np.array(img.convert("L"), dtype=np.uint8)
            self._draw_image(self.ax_input, arr, "Исходное")
            self._draw_placeholder(self.ax_result, "После IMCS", "Нажмите «Запустить IMCS»")
        elif _is_signal_path(path):
            if path.suffix.lower() == ".npy":
                signal = np.load(path)
            else:
                signal = np.loadtxt(path)
            self._draw_signal(self.ax_input, np.asarray(signal, dtype=np.float64).ravel(), "Исходный сигнал")
            self._draw_placeholder(self.ax_result, "После IMCS", "Нажмите «Запустить IMCS»")
        else:
            self._draw_placeholder(self.ax_input, "Исходные данные", "Неподдерживаемый формат")
            self._draw_placeholder(self.ax_result, "После IMCS", "Нажмите «Запустить IMCS»")
        self.draw_idle()

    def show_result(self, original: np.ndarray, reconstructed: np.ndarray) -> None:
        if original.ndim == 1:
            self._draw_signal(self.ax_input, original, "Исходный сигнал")
            self._draw_signal(self.ax_result, reconstructed, "После IMCS")
        else:
            self._draw_image(self.ax_input, original, "Исходное")
            self._draw_image(self.ax_result, reconstructed, "После IMCS")
        self.draw_idle()


class AnalyticsCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(11.0, 7.8), dpi=100, facecolor="#ffffff")
        super().__init__(self.figure)
        self.setStyleSheet("background: #ffffff;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.show_empty()

    def _style_axis(self, ax, title: str) -> None:
        ax.clear()
        ax.set_title(title, fontsize=11, pad=10, color="#202734")
        ax.tick_params(colors="#667085", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#d0d7e2")

    def _set_axis_labels(self, ax, xlabel: str = "", ylabel: str = "") -> None:
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color="#475467")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color="#475467")

    def _hide_axis_frame(self, ax) -> None:
        for spine in ax.spines.values():
            spine.set_visible(False)

    def show_empty(self, text: str = "Запустите IMCS, чтобы построить аналитику") -> None:
        self._resize_canvas(height_inches=7.8)
        self.figure.clear()
        self.figure.set_facecolor("#ffffff")
        ax = self.figure.add_subplot(111, facecolor="#ffffff")
        self._style_axis(ax, "Аналитика")
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color="#7a8599",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.draw_idle()

    def show_loading(self) -> None:
        self.show_empty("Графики рассчитываются...")

    def show_payload(self, payload: AnalyticsPayload) -> None:
        if payload.mode == "signal":
            self._draw_signal_payload(payload)
        else:
            self._draw_image_payload(payload)
        self.draw_idle()

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()

    def export_png(self, path: Path) -> None:
        self.figure.savefig(path, dpi=200, bbox_inches="tight")

    def export_panels(self, output_dir: Path, payload: AnalyticsPayload) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if payload.mode == "signal":
            return self._export_signal_panels(output_dir, payload)
        return self._export_image_panels(output_dir, payload)

    def _draw_signal_payload(self, payload: AnalyticsPayload) -> None:
        convergence_count = len(payload.convergence_images)
        height_inches = 7.6 + (2.8 * convergence_count)
        self._resize_canvas(height_inches=height_inches)
        self.figure.clear()
        self.figure.set_facecolor("#ffffff")
        height_ratios = [1.0, 1.0] + ([1.2] * convergence_count)
        grid = self.figure.add_gridspec(
            2 + convergence_count,
            2,
            hspace=0.24,
            wspace=0.22,
            height_ratios=height_ratios,
        )
        ax_overlay = self.figure.add_subplot(grid[0, 0], facecolor="#ffffff")
        ax_error = self.figure.add_subplot(grid[0, 1], facecolor="#ffffff")
        ax_coeff = self.figure.add_subplot(grid[1, 0], facecolor="#ffffff")
        ax_sorted = self.figure.add_subplot(grid[1, 1], facecolor="#ffffff")

        self._style_axis(ax_overlay, "Исходный и восстановленный сигнал")
        ax_overlay.plot(payload.original_display, color="#2f6fed", lw=1.3, label="Исходный")
        ax_overlay.plot(
            payload.reconstructed_display, color="#12b76a", lw=1.1, alpha=0.85, label="После IMCS"
        )
        ax_overlay.grid(True, alpha=0.35, color="#d8deea")
        ax_overlay.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax_overlay, "Отсчёт", "Амплитуда")

        self._style_axis(ax_error, "Абсолютная ошибка")
        ax_error.plot(payload.error_display, color="#ef4444", lw=1.1)
        ax_error.grid(True, alpha=0.35, color="#d8deea")
        self._set_axis_labels(ax_error, "Отсчёт", "Ошибка")

        self._style_axis(ax_coeff, f"Коэффициенты в базисе {payload.basis.upper()}")
        ax_coeff.plot(
            payload.coeff_display_original,
            color="#2f6fed",
            lw=1.1,
            alpha=0.9,
            label="Исходный",
        )
        ax_coeff.plot(
            payload.coeff_display_reconstructed,
            color="#12b76a",
            lw=1.0,
            alpha=0.85,
            label="После IMCS",
        )
        ax_coeff.grid(True, alpha=0.35, color="#d8deea")
        ax_coeff.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax_coeff, "Индекс коэффициента", "Значение")

        self._style_axis(ax_sorted, "Модули коэффициентов")
        ax_sorted.semilogy(payload.sorted_coeff_original, color="#2f6fed", lw=1.2, label="Исходный")
        ax_sorted.semilogy(
            payload.sorted_coeff_reconstructed,
            color="#12b76a",
            lw=1.1,
            alpha=0.85,
            label="После IMCS",
        )
        ax_sorted.grid(True, alpha=0.35, color="#d8deea")
        ax_sorted.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax_sorted, "Индекс после сортировки", "Модуль коэффициента")

        for idx, image in enumerate(payload.convergence_images):
            ax_conv = self.figure.add_subplot(grid[2 + idx, :], facecolor="#ffffff")
            title = payload.convergence_titles[idx] if idx < len(payload.convergence_titles) else "Сходимость"
            self._style_axis(ax_conv, title)
            ax_conv.imshow(image)
            ax_conv.set_xticks([])
            ax_conv.set_yticks([])
            self._hide_axis_frame(ax_conv)

        self.figure.suptitle(
            f"Аналитика: {payload.source_label} | {payload.analysis_note}",
            fontsize=12,
            color="#202734",
            y=0.972,
        )
        self.figure.subplots_adjust(left=0.085, right=0.98, top=0.935, bottom=0.06)

    def _draw_image_payload(self, payload: AnalyticsPayload) -> None:
        convergence_count = len(payload.convergence_images)
        height_inches = 7.8 + (3.0 * convergence_count)
        self._resize_canvas(height_inches=height_inches)
        self.figure.clear()
        self.figure.set_facecolor("#ffffff")
        height_ratios = [0.95, 1.0] + ([1.35] * convergence_count)
        grid = self.figure.add_gridspec(
            2 + convergence_count,
            3,
            hspace=0.24,
            wspace=0.22,
            height_ratios=height_ratios,
        )
        ax_original = self.figure.add_subplot(grid[0, 0], facecolor="#ffffff")
        ax_reconstructed = self.figure.add_subplot(grid[0, 1], facecolor="#ffffff")
        ax_error = self.figure.add_subplot(grid[0, 2], facecolor="#ffffff")
        ax_hist = self.figure.add_subplot(grid[1, 0], facecolor="#ffffff")
        ax_basis = self.figure.add_subplot(grid[1, 1], facecolor="#ffffff")
        ax_sorted = self.figure.add_subplot(grid[1, 2], facecolor="#ffffff")

        self._style_axis(ax_original, "Исходное")
        self._imshow_analysis_image(ax_original, payload.original_display)
        ax_original.set_xticks([])
        ax_original.set_yticks([])

        self._style_axis(ax_reconstructed, "После IMCS")
        self._imshow_analysis_image(ax_reconstructed, payload.reconstructed_display)
        ax_reconstructed.set_xticks([])
        ax_reconstructed.set_yticks([])

        self._style_axis(ax_error, "Heatmap ошибки")
        ax_error.imshow(payload.error_display, cmap="inferno")
        ax_error.set_xticks([])
        ax_error.set_yticks([])

        self._style_axis(ax_hist, "Гистограмма абсолютной ошибки")
        edges = payload.histogram_edges
        counts = payload.histogram_counts
        widths = np.diff(edges)
        ax_hist.bar(edges[:-1], counts, width=widths, align="edge", color="#2f6fed", alpha=0.8)
        ax_hist.grid(True, alpha=0.25, color="#d8deea")
        self._set_axis_labels(ax_hist, "Абсолютная ошибка", "Частота")

        self._style_axis(ax_basis, f"Heatmap коэффициентов ({payload.basis.upper()})")
        ax_basis.imshow(payload.coeff_display_original, cmap="magma")
        ax_basis.set_xticks([])
        ax_basis.set_yticks([])

        self._style_axis(ax_sorted, "Модули коэффициентов")
        ax_sorted.semilogy(payload.sorted_coeff_original, color="#2f6fed", lw=1.2, label="Исходное")
        ax_sorted.semilogy(
            payload.sorted_coeff_reconstructed,
            color="#12b76a",
            lw=1.1,
            alpha=0.85,
            label="После IMCS",
        )
        ax_sorted.grid(True, alpha=0.35, color="#d8deea")
        ax_sorted.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax_sorted, "Индекс после сортировки", "Модуль коэффициента")

        for idx, image in enumerate(payload.convergence_images):
            ax_conv = self.figure.add_subplot(grid[2 + idx, :], facecolor="#ffffff")
            title = payload.convergence_titles[idx] if idx < len(payload.convergence_titles) else "Сходимость"
            self._style_axis(ax_conv, title)
            ax_conv.imshow(image)
            ax_conv.set_xticks([])
            ax_conv.set_yticks([])
            self._hide_axis_frame(ax_conv)

        self.figure.suptitle(
            f"Аналитика: {payload.source_label} | {payload.analysis_note}",
            fontsize=12,
            color="#202734",
            y=0.972,
        )
        self.figure.subplots_adjust(left=0.085, right=0.98, top=0.935, bottom=0.06)

    def _imshow_analysis_image(self, ax, image: np.ndarray) -> None:
        if image.ndim == 3:
            ax.imshow(np.clip(image, 0, 255).astype(np.uint8))
            return
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    def _resize_canvas(self, *, height_inches: float) -> None:
        self.figure.set_size_inches(11.0, height_inches, forward=True)
        self.setMinimumHeight(int(height_inches * self.figure.dpi))

    def _create_export_figure(self, title: str, width: float = 5.6, height: float = 4.2):
        fig = Figure(figsize=(width, height), dpi=200, facecolor="#ffffff")
        ax = fig.add_subplot(111, facecolor="#ffffff")
        self._style_axis(ax, title)
        return fig, ax

    def _save_export_figure(self, fig: Figure, path: Path) -> Path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.clear()
        return path

    def _export_signal_panels(self, output_dir: Path, payload: AnalyticsPayload) -> list[Path]:
        exported: list[Path] = []

        fig, ax = self._create_export_figure("Исходный и восстановленный сигнал", 6.2, 4.0)
        ax.plot(payload.original_display, color="#2f6fed", lw=1.3, label="Исходный")
        ax.plot(payload.reconstructed_display, color="#12b76a", lw=1.1, alpha=0.85, label="После IMCS")
        ax.grid(True, alpha=0.35, color="#d8deea")
        ax.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax, "Отсчёт", "Амплитуда")
        exported.append(self._save_export_figure(fig, output_dir / "signal_overlay.png"))

        fig, ax = self._create_export_figure("Абсолютная ошибка", 6.2, 4.0)
        ax.plot(payload.error_display, color="#ef4444", lw=1.1)
        ax.grid(True, alpha=0.35, color="#d8deea")
        self._set_axis_labels(ax, "Отсчёт", "Ошибка")
        exported.append(self._save_export_figure(fig, output_dir / "signal_error.png"))

        fig, ax = self._create_export_figure(f"Коэффициенты в базисе {payload.basis.upper()}", 6.2, 4.0)
        ax.plot(payload.coeff_display_original, color="#2f6fed", lw=1.1, alpha=0.9, label="Исходный")
        ax.plot(payload.coeff_display_reconstructed, color="#12b76a", lw=1.0, alpha=0.85, label="После IMCS")
        ax.grid(True, alpha=0.35, color="#d8deea")
        ax.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax, "Индекс коэффициента", "Значение")
        exported.append(self._save_export_figure(fig, output_dir / "basis_coefficients.png"))

        fig, ax = self._create_export_figure("Модули коэффициентов", 6.2, 4.0)
        ax.semilogy(payload.sorted_coeff_original, color="#2f6fed", lw=1.2, label="Исходный")
        ax.semilogy(payload.sorted_coeff_reconstructed, color="#12b76a", lw=1.1, alpha=0.85, label="После IMCS")
        ax.grid(True, alpha=0.35, color="#d8deea")
        ax.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax, "Индекс после сортировки", "Модуль коэффициента")
        exported.append(self._save_export_figure(fig, output_dir / "sorted_coefficients.png"))

        for idx, image in enumerate(payload.convergence_images):
            fig, ax = self._create_export_figure(
                payload.convergence_titles[idx] if idx < len(payload.convergence_titles) else "Сходимость",
                6.0,
                4.5,
            )
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            self._hide_axis_frame(ax)
            suffix = payload.convergence_titles[idx].split()[-1].lower() if idx < len(payload.convergence_titles) else str(idx)
            exported.append(self._save_export_figure(fig, output_dir / f"convergence_{suffix}.png"))

        return exported

    def _export_image_panels(self, output_dir: Path, payload: AnalyticsPayload) -> list[Path]:
        exported: list[Path] = []

        fig, ax = self._create_export_figure("Исходное", 5.0, 4.5)
        self._imshow_analysis_image(ax, payload.original_display)
        ax.set_xticks([])
        ax.set_yticks([])
        exported.append(self._save_export_figure(fig, output_dir / "original.png"))

        fig, ax = self._create_export_figure("После IMCS", 5.0, 4.5)
        self._imshow_analysis_image(ax, payload.reconstructed_display)
        ax.set_xticks([])
        ax.set_yticks([])
        exported.append(self._save_export_figure(fig, output_dir / "reconstructed.png"))

        fig, ax = self._create_export_figure("Heatmap ошибки", 5.0, 4.5)
        ax.imshow(payload.error_display, cmap="inferno")
        ax.set_xticks([])
        ax.set_yticks([])
        exported.append(self._save_export_figure(fig, output_dir / "error_heatmap.png"))

        fig, ax = self._create_export_figure("Гистограмма абсолютной ошибки", 5.8, 4.0)
        edges = payload.histogram_edges
        counts = payload.histogram_counts
        widths = np.diff(edges)
        ax.bar(edges[:-1], counts, width=widths, align="edge", color="#2f6fed", alpha=0.8)
        ax.grid(True, alpha=0.25, color="#d8deea")
        self._set_axis_labels(ax, "Абсолютная ошибка", "Частота")
        exported.append(self._save_export_figure(fig, output_dir / "error_histogram.png"))

        fig, ax = self._create_export_figure(f"Heatmap коэффициентов ({payload.basis.upper()})", 5.0, 4.5)
        ax.imshow(payload.coeff_display_original, cmap="magma")
        ax.set_xticks([])
        ax.set_yticks([])
        exported.append(self._save_export_figure(fig, output_dir / "basis_heatmap.png"))

        fig, ax = self._create_export_figure("Модули коэффициентов", 5.8, 4.0)
        ax.semilogy(payload.sorted_coeff_original, color="#2f6fed", lw=1.2, label="Исходное")
        ax.semilogy(payload.sorted_coeff_reconstructed, color="#12b76a", lw=1.1, alpha=0.85, label="После IMCS")
        ax.grid(True, alpha=0.35, color="#d8deea")
        ax.legend(loc="upper right", fontsize=8)
        self._set_axis_labels(ax, "Индекс после сортировки", "Модуль коэффициента")
        exported.append(self._save_export_figure(fig, output_dir / "sorted_coefficients.png"))

        for idx, image in enumerate(payload.convergence_images):
            fig, ax = self._create_export_figure(
                payload.convergence_titles[idx] if idx < len(payload.convergence_titles) else "Сходимость",
                6.0,
                4.5,
            )
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            self._hide_axis_frame(ax)
            suffix = payload.convergence_titles[idx].split()[-1].lower() if idx < len(payload.convergence_titles) else str(idx)
            exported.append(self._save_export_figure(fig, output_dir / f"convergence_{suffix}.png"))

        return exported


class IMCSQtMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.selected_path: Optional[Path] = None
        self.worker_process: Optional[mp.Process] = None
        self.worker_queue: Optional[mp.Queue] = None
        self.analytics_process: Optional[mp.Process] = None
        self.analytics_queue: Optional[mp.Queue] = None
        self.analytics_payload: Optional[AnalyticsPayload] = None
        self.last_result = None
        self.active_task: Optional[dict] = None
        self._summary_log_started = False

        self.setWindowTitle("IMCS")
        self.resize(1460, 920)
        self.setMinimumSize(980, 680)
        self._apply_theme()
        self._build_ui()

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(30)
        self.poll_timer.timeout.connect(self._poll_worker)
        self.analytics_poll_timer = QTimer(self)
        self.analytics_poll_timer.setInterval(30)
        self.analytics_poll_timer.timeout.connect(self._poll_analytics_worker)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.output_dir_edit.setText(str(OUTPUT_DIR.resolve()))
        self._set_running_state(False)
        self.preview.show_empty()
        self.analytics_canvas.show_empty()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f3f6fb;
                color: #1f2937;
                font-size: 13px;
            }
            QLabel, QCheckBox {
                background: transparent;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: 700;
                color: #1f2937;
            }
            QLabel#subtitleLabel {
                color: #667085;
                font-size: 12px;
            }
            QFrame#topCard, QFrame#summaryCard {
                background: #ffffff;
                border: 1px solid #d8deea;
                border-radius: 16px;
            }
            QDockWidget {
                color: #1f2937;
            }
            QDockWidget::title {
                background: #ffffff;
                text-align: left;
                padding: 10px 14px;
                border-bottom: 1px solid #d8deea;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d8deea;
                border-radius: 14px;
                margin-top: 18px;
                padding-top: 16px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background: transparent;
            }
            QPushButton, QToolButton {
                background: #ffffff;
                border: 1px solid #cfd7e4;
                border-radius: 10px;
                padding: 9px 14px;
                color: #243041;
            }
            QPushButton:hover, QToolButton:hover {
                background: #eef3fb;
            }
            QToolButton#analyticsButton:disabled {
                background: #f2f4f7;
                border-color: #d8deea;
                color: #98a2b3;
            }
            QPushButton#primaryButton {
                background: #2f6fed;
                border-color: #2f6fed;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover {
                background: #4a83ef;
            }
            QPushButton#dangerButton {
                background: #fff3f4;
                border-color: #f0c6cc;
                color: #a33545;
            }
            QPushButton#dangerButton:hover {
                background: #ffe7ea;
            }
            QPushButton#dangerButton:disabled {
                background: #f2f4f7;
                border-color: #d8deea;
                color: #98a2b3;
            }
            QPushButton:checked {
                background: #dfeaff;
                border-color: #2f6fed;
                color: #1d4ed8;
                font-weight: 600;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #cfd7e4;
                border-radius: 10px;
                padding: 8px 10px;
                selection-background-color: #bcd2ff;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                border: 1px solid #cfd7e4;
                selection-background-color: #dfeaff;
            }
            QPlainTextEdit {
                padding: 10px;
            }
            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #d8deea;
                color: #667085;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QCheckBox {
                spacing: 8px;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        top_card = QFrame()
        top_card.setObjectName("topCard")
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(18, 16, 18, 16)
        top_layout.setSpacing(14)

        title_box = QVBoxLayout()
        title = QLabel("IMCS")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Image Compression using Compressed Sensing")
        subtitle.setObjectName("subtitleLabel")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top_layout.addLayout(title_box, 1)

        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("color: #667085;")
        self.file_label.setWordWrap(True)
        top_layout.addWidget(self.file_label, 2)

        self.btn_choose = QPushButton("Выбрать файл")
        self.btn_choose.clicked.connect(self._choose_file)
        top_layout.addWidget(self.btn_choose)

        self.btn_run = QPushButton("Запустить IMCS")
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self._start_processing)
        top_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.setObjectName("dangerButton")
        self.btn_stop.clicked.connect(self._stop_processing)
        top_layout.addWidget(self.btn_stop)

        self.btn_view_toggle = QToolButton()
        self.btn_view_toggle.setObjectName("analyticsButton")
        self.btn_view_toggle.setText("Аналитика")
        self.btn_view_toggle.setToolTip("Станет доступна после первого успешного запуска IMCS.")
        self.btn_view_toggle.clicked.connect(self._toggle_view)
        self.btn_view_toggle.setEnabled(False)
        top_layout.addWidget(self.btn_view_toggle)

        self.btn_settings = QToolButton()
        self.btn_settings.setText("Настройки ›")
        self.btn_settings.setCheckable(True)
        self.btn_settings.setChecked(True)
        self.btn_settings.toggled.connect(self._toggle_settings)
        top_layout.addWidget(self.btn_settings)

        root.addWidget(top_card)

        self.view_stack = QStackedWidget()

        run_page = QWidget()
        run_layout = QVBoxLayout(run_page)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(14)

        self.preview = PreviewCanvas()
        run_layout.addWidget(self.preview, 1)

        summary_card = QFrame()
        summary_card.setObjectName("summaryCard")
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(16, 16, 16, 16)
        summary_layout.setSpacing(10)

        summary_title = QLabel("Сводка запуска")
        summary_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937;")
        summary_layout.addWidget(summary_title)

        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMaximumBlockCount(400)
        self.summary_box.setPlainText(
            "Здесь появятся времена, метрики и параметры последнего запуска."
        )
        summary_layout.addWidget(self.summary_box)

        run_layout.addWidget(summary_card, 0)
        self.view_stack.addWidget(run_page)

        analytics_page = QWidget()
        analytics_layout = QVBoxLayout(analytics_page)
        analytics_layout.setContentsMargins(0, 0, 0, 0)
        analytics_layout.setSpacing(12)

        analytics_card = QFrame()
        analytics_card.setObjectName("summaryCard")
        analytics_card_layout = QVBoxLayout(analytics_card)
        analytics_card_layout.setContentsMargins(16, 16, 16, 16)
        analytics_card_layout.setSpacing(10)

        analytics_header = QHBoxLayout()
        analytics_header.setSpacing(10)
        analytics_title = QLabel("Аналитика запуска")
        analytics_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937;")
        self.btn_refresh_analytics = QPushButton("Пересчитать")
        self.btn_refresh_analytics.clicked.connect(self._start_analytics_processing)
        self.btn_refresh_analytics.setEnabled(False)
        self.btn_export_analytics = QPushButton("Экспорт PNG")
        self.btn_export_analytics.clicked.connect(self._export_analytics_png)
        self.btn_export_analytics.setEnabled(False)

        analytics_header.addWidget(analytics_title)
        analytics_header.addStretch(1)
        analytics_header.addWidget(self.btn_refresh_analytics)
        analytics_header.addWidget(self.btn_export_analytics)
        analytics_card_layout.addLayout(analytics_header)

        self.analytics_canvas = AnalyticsCanvas()
        self.analytics_scroll = QScrollArea()
        self.analytics_scroll.setWidgetResizable(True)
        self.analytics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.analytics_scroll.setWidget(self.analytics_canvas)
        analytics_card_layout.addWidget(self.analytics_scroll, 1)
        analytics_layout.addWidget(analytics_card, 1)
        self.view_stack.addWidget(analytics_page)

        root.addWidget(self.view_stack, 1)

        self.settings_dock = QDockWidget("Настройки стенда", self)
        self.settings_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.settings_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.settings_dock.setMinimumWidth(400)
        self.settings_dock.setMaximumWidth(560)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)
        self.settings_dock.setTitleBarWidget(QWidget())
        self.settings_dock.visibilityChanged.connect(self._on_settings_visibility_changed)
        self.resizeDocks([self.settings_dock], [500], Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_root = QWidget()
        settings_layout = QVBoxLayout(settings_root)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(12)

        settings_layout.addWidget(self._build_compression_group())
        settings_layout.addWidget(self._build_reconstruction_group())
        settings_layout.addWidget(self._build_color_group())
        settings_layout.addWidget(self._build_performance_group())
        settings_layout.addWidget(self._build_output_group())
        settings_layout.addStretch(1)

        scroll.setWidget(settings_root)
        self.settings_dock.setWidget(scroll)

        status = QStatusBar()
        status.showMessage("Готово к работе.")
        self.setStatusBar(status)

    def _build_compression_group(self) -> QGroupBox:
        group = QGroupBox("Сжатие и блоки")
        layout = QFormLayout(group)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setDecimals(2)
        self.ratio_spin.setRange(0.01, 0.99)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setValue(DEFAULT_RATIO)
        self.ratio_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.block_combo = QComboBox()
        self.block_combo.addItem("8 × 8", 8)
        self.block_combo.addItem("16 × 16", 16)
        self.block_combo.addItem("32 × 32", 32)
        self.block_combo.addItem("Полный кадр (0)", 0)
        self.block_combo.setCurrentIndex(0)
        self.block_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.measurement_combo = QComboBox()
        self.measurement_combo.addItem("shared", "shared")
        self.measurement_combo.addItem("per_block", "per_block")
        self.measurement_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.matrix_combo = QComboBox()
        self.matrix_combo.addItem("gaussian", "gaussian")
        self.matrix_combo.addItem("bernoulli", "bernoulli")
        self.matrix_combo.addItem("random", "random")
        self.matrix_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.measurement_dtype_combo = QComboBox()
        self.measurement_dtype_combo.addItem("float64", "float64")
        self.measurement_dtype_combo.addItem("int16", "int16")
        self.measurement_dtype_combo.addItem("int8", "int8")
        dtype_index = self.measurement_dtype_combo.findData("int8")
        if dtype_index < 0:
            dtype_index = self.measurement_dtype_combo.findData(DEFAULT_MEASUREMENT_DTYPE)
        self.measurement_dtype_combo.setCurrentIndex(max(dtype_index, 0))
        self.measurement_dtype_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.imcs_mode_group = QButtonGroup(self)
        self.imcs_mode_group.setExclusive(True)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self.btn_mode_cs_only = QPushButton("CS-only")
        self.btn_mode_mean = QPushButton("+ mean")
        self.btn_mode_hybrid = QPushButton("+ mean + LF4")
        for button, mode_id in (
            (self.btn_mode_cs_only, 0),
            (self.btn_mode_mean, 1),
            (self.btn_mode_hybrid, 2),
        ):
            button.setCheckable(True)
            self.imcs_mode_group.addButton(button, mode_id)
            mode_row.addWidget(button)

        self.imcs_mode_group.button(2).setChecked(True)

        layout.addRow("Коэффициент сжатия:", self.ratio_spin)
        layout.addRow("Размер блока:", self.block_combo)
        layout.addRow("Режим Phi:", self.measurement_combo)
        layout.addRow("Тип матрицы:", self.matrix_combo)
        layout.addRow("Хранение y:", self.measurement_dtype_combo)
        layout.addRow("Режим IMCS:", mode_row)
        return group

    def _build_reconstruction_group(self) -> QGroupBox:
        group = QGroupBox("Восстановление")
        layout = QFormLayout(group)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("OMP", "omp")
        self.algorithm_combo.addItem("ISTA", "ista")
        self.algorithm_combo.addItem("FISTA", "fista")
        self.algorithm_combo.addItem("SA", "sa")
        self.algorithm_combo.setCurrentIndex(0)
        self.algorithm_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.basis_combo = QComboBox()
        self.basis_combo.addItem("DCT", "dct")
        self.basis_combo.addItem("Wavelet (Haar)", "wavelet")
        self.basis_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addRow("Алгоритм:", self.algorithm_combo)
        layout.addRow("Базис:", self.basis_combo)
        return group

    def _build_color_group(self) -> QGroupBox:
        group = QGroupBox("Цвет")
        layout = QFormLayout(group)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)

        self.color_combo = QComboBox()
        self.color_combo.addItem("gray", "gray")
        self.color_combo.addItem("rgb", "rgb")
        self.color_combo.addItem("ycbcr", "ycbcr")
        self.color_combo.setCurrentIndex(2 if DEFAULT_COLOR_MODE == "ycbcr" else 0)
        self.color_combo.currentIndexChanged.connect(self._sync_mode_controls)
        self.color_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.chroma_spin = QDoubleSpinBox()
        self.chroma_spin.setDecimals(2)
        self.chroma_spin.setRange(0.05, 1.5)
        self.chroma_spin.setSingleStep(0.05)
        self.chroma_spin.setValue(DEFAULT_CHROMA_RATIO_SCALE)
        self.chroma_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addRow("Цветовой режим:", self.color_combo)
        layout.addRow("Множитель chroma:", self.chroma_spin)
        return group

    def _build_performance_group(self) -> QGroupBox:
        group = QGroupBox("Производительность")
        layout = QFormLayout(group)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)

        self.parallel_check = QCheckBox("Распараллеливать блоки")
        self.parallel_check.setChecked(True)
        self.parallel_check.toggled.connect(self._sync_mode_controls)

        self.parallel_combo = QComboBox()
        self.parallel_combo.addItem("2 процесса", 2)
        self.parallel_combo.addItem("4 процесса", 4)
        self.parallel_combo.addItem("8 процессов", 8)
        self.parallel_combo.setCurrentIndex(2)
        self.parallel_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addRow("", self.parallel_check)
        layout.addRow("Число процессов:", self.parallel_combo)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Вывод")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(8)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.btn_output_dir = QPushButton("Папка…")
        self.btn_output_dir.clicked.connect(self._choose_output_dir)
        self.btn_output_dir.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.output_dir_edit, 1)
        row.addWidget(self.btn_output_dir, 0)

        layout.addWidget(QLabel("Каталог результатов:"))
        layout.addLayout(row)
        return group

    def _toggle_settings(self, visible: bool) -> None:
        self.settings_dock.setVisible(visible)

    def _on_settings_visibility_changed(self, visible: bool) -> None:
        self.btn_settings.blockSignals(True)
        self.btn_settings.setChecked(visible)
        self.btn_settings.setText("Настройки ›" if visible else "Настройки ‹")
        self.btn_settings.blockSignals(False)

    def _choose_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение или сигнал",
            str(INPUT_DIR),
            "Изображения и сигналы (*.png *.jpg *.jpeg *.bmp *.tiff *.npy *.txt);;Все файлы (*.*)",
        )
        if not chosen:
            self.statusBar().showMessage("Выбор файла отменён.")
            return

        self.selected_path = Path(chosen)
        self.last_result = None
        self._set_view("run")
        self.preview.show_input_path(self.selected_path)
        self._sync_mode_controls()
        self._reset_analytics_state("Аналитика будет построена после запуска IMCS.")
        self.statusBar().showMessage(f"Предпросмотр: {self.selected_path.name}")

    def _choose_output_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Каталог для результатов",
            self.output_dir_edit.text() or str(OUTPUT_DIR),
        )
        if chosen:
            self.output_dir_edit.setText(chosen)

    def _current_block_edge(self) -> int:
        return int(self.block_combo.currentData())

    def _parallel_block_workers(self) -> Optional[int]:
        if not self.parallel_check.isChecked():
            return None
        return int(self.parallel_combo.currentData())

    def _current_imcs_mode_options(self) -> tuple[bool, int]:
        mode_id = self.imcs_mode_group.checkedId()
        if mode_id == 2:
            return True, 4
        if mode_id == 1:
            return True, 0
        return False, 0

    def _sync_mode_controls(self) -> None:
        path = self.selected_path
        is_signal = bool(path and _is_signal_path(path))
        is_image = bool(path and _is_image_path(path))
        if path is None:
            self.file_label.setText("Файл не выбран")
        elif is_signal:
            self.file_label.setText(f"Сигнал: {path.name}")
        elif is_image:
            self.file_label.setText(f"Изображение: {path.name}")
        else:
            self.file_label.setText(path.name)

        self.color_combo.setEnabled(not is_signal)
        self.chroma_spin.setEnabled(not is_signal and self.color_combo.currentData() == "ycbcr")
        self.block_combo.setEnabled(not is_signal)
        self.measurement_combo.setEnabled(not is_signal)
        self.btn_mode_cs_only.setEnabled(not is_signal)
        self.btn_mode_mean.setEnabled(not is_signal)
        self.btn_mode_hybrid.setEnabled(not is_signal)
        self.parallel_combo.setEnabled(self.parallel_check.isChecked())

    def _build_task(self) -> Optional[dict]:
        if self.selected_path is None:
            QMessageBox.warning(self, "IMCS", "Сначала выберите файл.")
            return None

        output_dir = Path(self.output_dir_edit.text().strip() or str(OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)

        algorithm = str(self.algorithm_combo.currentData())
        basis = str(self.basis_combo.currentData())
        matrix_type = str(self.matrix_combo.currentData())
        measurement_mode = str(self.measurement_combo.currentData())
        measurement_dtype = str(self.measurement_dtype_combo.currentData())
        block_mean_residual, low_frequency_coeffs = self._current_imcs_mode_options()
        parallel_block_workers = self._parallel_block_workers()

        if _is_signal_path(self.selected_path):
            return {
                "task_type": "signal",
                "kwargs": {
                    "signal_path": self.selected_path,
                    "output_dir": output_dir,
                    "compression_ratio": float(self.ratio_spin.value()),
                    "algorithm": algorithm,
                    "verbose": False,
                    "basis": basis,
                    "matrix_type": matrix_type,
                    "measurement_dtype": measurement_dtype,
                    "collect_decode_profile": True,
                    "parallel_block_workers": parallel_block_workers,
                },
            }

        if _is_image_path(self.selected_path):
            color_mode = str(self.color_combo.currentData())
            common_kwargs = {
                "image_path": self.selected_path,
                "output_dir": output_dir,
                "compression_ratio": float(self.ratio_spin.value()),
                "algorithm": algorithm,
                "block_edge": self._current_block_edge(),
                "force_full_frame": False,
                "verbose": False,
                "basis": basis,
                "matrix_type": matrix_type,
                "measurement_mode": measurement_mode,
                "measurement_dtype": measurement_dtype,
                "block_mean_residual": block_mean_residual,
                "low_frequency_coeffs": low_frequency_coeffs,
                "collect_decode_profile": True,
                "parallel_block_workers": parallel_block_workers,
                "visualize_convergence": True,
            }
            if color_mode == "gray":
                return {"task_type": "image", "kwargs": common_kwargs}
            common_kwargs["color_mode"] = color_mode
            common_kwargs["chroma_ratio_scale"] = float(self.chroma_spin.value())
            return {"task_type": "color_image", "kwargs": common_kwargs}

        QMessageBox.warning(self, "IMCS", "Неподдерживаемый формат файла.")
        return None

    def _set_running_state(self, is_running: bool) -> None:
        self.btn_run.setEnabled(not is_running)
        self.btn_choose.setEnabled(not is_running)
        self.btn_output_dir.setEnabled(not is_running)
        self.btn_stop.setEnabled(is_running)
        self.btn_view_toggle.setEnabled((not is_running) and self.last_result is not None)
        self.btn_export_analytics.setEnabled(
            (not is_running) and self.analytics_payload is not None
        )
        self.btn_refresh_analytics.setEnabled((not is_running) and self.last_result is not None)

    def _start_processing(self) -> None:
        if self.worker_process is not None and self.worker_process.is_alive():
            QMessageBox.information(self, "IMCS", "Обработка уже выполняется.")
            return

        task = self._build_task()
        if task is None:
            return

        self.last_result = None
        self._set_view("run")
        self._reset_analytics_state("Аналитика обновится после завершения нового запуска.")
        self._summary_log_started = True
        self.summary_box.setPlainText("Запуск обработки…\n")
        self.statusBar().showMessage("IMCS: выполняется обработка…")
        self.active_task = task
        ctx = mp.get_context("spawn")
        self.worker_queue = ctx.Queue()
        self.worker_process = ctx.Process(target=_run_processing_task, args=(task, self.worker_queue))
        self.worker_process.start()
        self._set_running_state(True)
        self.poll_timer.start()

    def _cleanup_worker(self) -> None:
        if self.worker_process is not None:
            if self.worker_process.is_alive():
                self.worker_process.join(timeout=0.1)
            self.worker_process = None
        if self.worker_queue is not None:
            try:
                self.worker_queue.close()
            except Exception:
                pass
            self.worker_queue = None

    def _cleanup_analytics_worker(self) -> None:
        if self.analytics_process is not None:
            if self.analytics_process.is_alive():
                self.analytics_process.join(timeout=0.1)
            self.analytics_process = None
        if self.analytics_queue is not None:
            try:
                self.analytics_queue.close()
            except Exception:
                pass
            self.analytics_queue = None

    def _set_view(self, name: str) -> None:
        is_analytics = name == "analytics"
        self.view_stack.setCurrentIndex(1 if is_analytics else 0)
        self.btn_view_toggle.setText("Запуск" if is_analytics else "Аналитика")

    def _toggle_view(self) -> None:
        if self.view_stack.currentIndex() == 0:
            if self.last_result is None:
                QMessageBox.information(self, "IMCS", "Сначала выполните запуск IMCS.")
                return
            self._set_view("analytics")
            if self.analytics_payload is None and (
                self.analytics_process is None or not self.analytics_process.is_alive()
            ):
                self._start_analytics_processing()
            return
        self._set_view("run")

    def _reset_analytics_state(self, message: str) -> None:
        self.analytics_poll_timer.stop()
        if self.analytics_process is not None and self.analytics_process.is_alive():
            self.analytics_process.terminate()
            self.analytics_process.join(timeout=1.0)
        self._cleanup_analytics_worker()
        self.analytics_payload = None
        self.analytics_canvas.show_empty(message)
        self.btn_export_analytics.setEnabled(False)
        self.btn_refresh_analytics.setEnabled(self.last_result is not None)
        self.btn_view_toggle.setEnabled(self.last_result is not None)

    def _start_analytics_processing(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "IMCS", "Нет результата для аналитики.")
            return
        self._reset_analytics_state("Рассчитываются графики аналитики...")
        self.analytics_canvas.show_loading()
        ctx = mp.get_context("spawn")
        self.analytics_queue = ctx.Queue()
        task = {"result": self.last_result}
        self.analytics_process = ctx.Process(target=_run_analytics_task, args=(task, self.analytics_queue))
        self.analytics_process.start()
        self.analytics_poll_timer.start()

    def _stop_processing(self) -> None:
        self.poll_timer.stop()
        if self.worker_process is not None and self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join(timeout=1.5)
        self._cleanup_worker()
        self.active_task = None
        self._set_running_state(False)
        self.statusBar().showMessage("Обработка остановлена.")
        self.summary_box.setPlainText("Обработка остановлена пользователем.")

    def _poll_analytics_worker(self) -> None:
        if self.analytics_queue is None:
            return
        try:
            message = self.analytics_queue.get_nowait()
        except Empty:
            return

        self.analytics_poll_timer.stop()
        if message["status"] == "error":
            self._cleanup_analytics_worker()
            self.analytics_canvas.show_empty(message["message"])
            self.statusBar().showMessage("Ошибка построения аналитики.")
            return

        self.analytics_payload = message["payload"]
        self._cleanup_analytics_worker()
        self.analytics_canvas.show_payload(self.analytics_payload)
        self.btn_export_analytics.setEnabled(True)
        self.btn_refresh_analytics.setEnabled(True)
        self.btn_view_toggle.setEnabled(True)
        self.statusBar().showMessage("Аналитические графики готовы.")

    def _poll_worker(self) -> None:
        if self.worker_queue is None:
            return
        log_messages: list[str] = []
        terminal_message = None
        while True:
            try:
                message = self.worker_queue.get_nowait()
            except Empty:
                break
            if message["status"] == "log":
                log_messages.append(message["message"])
                continue
            terminal_message = message
            break

        if log_messages:
            self._append_summary_log("\n".join(log_messages))

        if terminal_message is None:
            return
        message = terminal_message

        self.poll_timer.stop()
        self._set_running_state(False)

        if message["status"] == "error":
            self._cleanup_worker()
            QMessageBox.critical(self, "IMCS", message["message"])
            self.statusBar().showMessage("Ошибка обработки.")
            self.summary_box.setPlainText(message["message"])
            return

        task_type = message["task_type"]
        result = message["result"]
        self._cleanup_worker()
        if result is None:
            self.statusBar().showMessage("Запуск завершился без результата.")
            return

        self.last_result = result
        self.preview.show_result(result.original, result.reconstructed)
        self._append_summary_log("")
        self._append_summary_log("=== Итоговая сводка ===")
        self._append_summary_log(self._format_summary(task_type, result))
        self.btn_view_toggle.setEnabled(True)
        self.btn_refresh_analytics.setEnabled(True)
        self._start_analytics_processing()
        requested_workers = self._active_workers_text()
        self.statusBar().showMessage(
            f"Готово: код {result.t_encode:.3f} c | декод {result.t_decode:.3f} c | "
            f"workers {requested_workers} | {result.output_subdir}"
        )

    def _format_summary(self, task_type: str, result) -> str:
        metrics = getattr(result, "metrics", {})
        lines = [
            f"Тип задачи: {task_type}",
            f"Файл: {getattr(result, 'image_path', getattr(result, 'signal_path', Path('-')))}",
            f"Алгоритм: {getattr(result, 'algorithm', '-')}",
            f"Workers: {self._active_workers_text()}",
            f"Время кодирования: {getattr(result, 't_encode', 0.0):.3f} c",
            f"Время декодирования: {getattr(result, 't_decode', 0.0):.3f} c",
        ]
        if "psnr" in metrics:
            lines.extend(
                [
                    f"PSNR: {metrics['psnr']:.2f} dB",
                    f"SSIM: {metrics['ssim']:.4f}",
                    f"MAE: {metrics['mae']:.2f}",
                ]
            )
        if hasattr(result, "basis"):
            lines.append(f"Базис: {result.basis}")
        if hasattr(result, "matrix_type"):
            lines.append(f"Matrix type: {result.matrix_type}")
        if hasattr(result, "measurement_mode"):
            lines.append(f"Measurement mode: {result.measurement_mode}")
        if hasattr(result, "measurement_dtype"):
            lines.append(f"Measurement dtype: {result.measurement_dtype}")
        if hasattr(result, "block_mean_residual"):
            lines.append(f"Block mean residual: {result.block_mean_residual}")
        if hasattr(result, "low_frequency_coeffs"):
            lines.append(f"Low-frequency coeffs: {result.low_frequency_coeffs}")
        if hasattr(result, "block_size"):
            block_size = result.block_size
            lines.append(
                "Размер блока: "
                + (f"{block_size[0]}×{block_size[1]}" if block_size else "полный кадр")
            )
        if hasattr(result, "color_mode"):
            lines.append(f"Color mode: {result.color_mode}")
        profile = getattr(result, "decode_profile", None)
        if profile:
            lines.extend(
                [
                    "",
                    f"Блоков декодировано: {int(profile.get('blocks_total', 0))}",
                ]
            )
        lines.extend(["", f"Результаты: {result.output_subdir}"])
        return "\n".join(lines)

    def _active_workers_text(self) -> str:
        if self.active_task is None:
            return "авто"
        kwargs = self.active_task.get("kwargs", {})
        workers = kwargs.get("parallel_block_workers")
        return "авто" if workers is None else str(workers)

    def _append_summary_log(self, text: str) -> None:
        if not self._summary_log_started:
            self.summary_box.setPlainText("")
            self._summary_log_started = True
        self.summary_box.appendPlainText(text)
        scrollbar = self.summary_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _export_analytics_png(self) -> None:
        if self.last_result is None or self.analytics_payload is None:
            QMessageBox.information(self, "IMCS", "Сначала дождитесь построения аналитики.")
            return
        default_dir = Path(self.last_result.output_subdir) / "analytics_exports"
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Сохранить графики аналитики",
            str(default_dir),
        )
        if not chosen:
            self.statusBar().showMessage("Экспорт аналитики отменён.")
            return
        output_dir = Path(chosen)
        exported = self.analytics_canvas.export_panels(output_dir, self.analytics_payload)
        self.statusBar().showMessage(
            f"Аналитика сохранена: {len(exported)} PNG в {output_dir}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker_process is not None and self.worker_process.is_alive():
            self._stop_processing()
        if self.analytics_process is not None and self.analytics_process.is_alive():
            self.analytics_poll_timer.stop()
            self.analytics_process.terminate()
            self.analytics_process.join(timeout=1.0)
            self._cleanup_analytics_worker()
        super().closeEvent(event)


def run_app() -> None:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("IMCS")
    window = IMCSQtMainWindow()
    window.show()
    app.exec()
