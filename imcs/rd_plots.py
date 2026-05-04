"""
Thesis-ready plots and tables from rd_long.csv.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from matplotlib.figure import Figure


PLOT_COLORS = (
    "#2f6fed",
    "#12b76a",
    "#f97316",
    "#a855f7",
    "#ef4444",
    "#0ea5e9",
    "#64748b",
)


def read_rd_long_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _series_key(row: dict[str, str]) -> str:
    return row.get("variant_id") or row["method"]


def _series_label(row: dict[str, str]) -> str:
    return row.get("variant_label") or row["method"].upper()


def _param_float(row: dict[str, str]) -> float:
    return float(row["param_value"])


def _mean_ci(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0, 1
    ci95 = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(float(arr.size))
    return mean, ci95, int(arr.size)


def _format_mean_ci(values: Iterable[float], digits: int = 3) -> str:
    mean, ci95, n = _mean_ci(values)
    if n == 0:
        return "-"
    return f"{mean:.{digits}f} ± {ci95:.{digits}f}"


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, pad=10, color="#202734")
    ax.set_xlabel(xlabel, fontsize=9, color="#475467")
    ax.set_ylabel(ylabel, fontsize=9, color="#475467")
    ax.tick_params(colors="#667085", labelsize=8)
    ax.grid(True, alpha=0.28, color="#d8deea")
    for spine in ax.spines.values():
        spine.set_color("#d0d7e2")


def _save(fig: Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="#ffffff")
    fig.clear()


def _collapse_repeats(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["case_key"],
            _series_key(row),
            row["param_name"],
            row["param_value"],
        )
        grouped[key].append(row)

    collapsed: list[dict[str, str]] = []
    numeric_fields = (
        "bpp",
        "psnr_db",
        "ssim",
        "mae",
        "encode_time_s",
        "decode_time_s",
        "compressed_size_bytes",
    )
    for group_rows in grouped.values():
        base = dict(group_rows[0])
        for field in numeric_fields:
            values = np.array([_float(row, field) for row in group_rows], dtype=float)
            base[field] = f"{float(np.mean(values)):.9g}"
        base["repeat_index"] = "mean"
        collapsed.append(base)
    return collapsed


def _series_order(rows: Sequence[dict[str, str]]) -> list[str]:
    labels: dict[str, str] = {}
    for row in rows:
        labels[_series_key(row)] = _series_label(row)
    baseline = sorted(k for k in labels if not k.startswith("imcs"))
    imcs = sorted(k for k in labels if k.startswith("imcs"))
    return baseline + imcs


def _series_color_map(series_keys: Sequence[str]) -> dict[str, str]:
    return {key: PLOT_COLORS[idx % len(PLOT_COLORS)] for idx, key in enumerate(series_keys)}


def plot_rd_facets_overlay(
    rows: Sequence[dict[str, str]],
    output_path: Path,
    *,
    max_images: int = 8,
) -> None:
    """One panel per image, all available codecs and IMCS variants on PSNR-bpp axes."""

    collapsed = _collapse_repeats(rows)
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in collapsed:
        by_case[row["case_key"]].append(row)

    keys = sorted(by_case.keys())[:max_images]
    if not keys:
        return

    series_keys = _series_order(collapsed)
    colors = _series_color_map(series_keys)
    ncols = min(3, len(keys))
    nrows = int(math.ceil(len(keys) / ncols))
    fig = Figure(figsize=(4.2 * ncols, 3.6 * nrows), facecolor="#ffffff")

    for idx, case_key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, idx + 1, facecolor="#ffffff")
        pts = by_case[case_key]
        for series_key in series_keys:
            selected = [row for row in pts if _series_key(row) == series_key]
            if len(selected) < 1:
                continue
            selected = sorted(selected, key=lambda row: _float(row, "bpp"))
            bpp = np.array([_float(row, "bpp") for row in selected], dtype=float)
            psnr = np.array([min(_float(row, "psnr_db"), 99.0) for row in selected], dtype=float)
            marker = "s" if series_key.startswith("imcs") else "o"
            ax.plot(
                bpp,
                psnr,
                marker=marker,
                linewidth=1.35,
                markersize=3.6,
                color=colors[series_key],
                label=_series_label(selected[0]),
            )
        _style_axis(ax, case_key, "Бит на пиксель (bpp)", "PSNR, дБ")
        ax.legend(fontsize=6.7, loc="best", frameon=True)

    fig.suptitle("Зависимость качества восстановления от битрейта", fontsize=13, color="#202734")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, output_path)


def _interp_sorted_x(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    if xs.size == 0:
        return np.full_like(xq, np.nan, dtype=float)
    uniq_x: list[float] = []
    uniq_y: list[float] = []
    for i, xv in enumerate(xs):
        if i > 0 and abs(float(xv) - float(xs[i - 1])) < 1e-9:
            uniq_y[-1] = 0.5 * (uniq_y[-1] + float(ys[i]))
            continue
        uniq_x.append(float(xv))
        uniq_y.append(float(ys[i]))
    return np.interp(xq, np.array(uniq_x), np.array(uniq_y), left=np.nan, right=np.nan)


def plot_rd_mean_with_ci(
    rows: Sequence[dict[str, str]],
    output_path: Path,
    *,
    n_grid: int = 160,
) -> None:
    """Mean PSNR-bpp curve across images with 95% confidence interval."""

    collapsed = _collapse_repeats(rows)
    if not collapsed:
        return
    finite_bpp = np.array([_float(row, "bpp") for row in collapsed if _float(row, "bpp") > 0])
    if finite_bpp.size == 0:
        return
    bpp_min = max(0.01, float(np.nanmin(finite_bpp)) * 0.95)
    bpp_max = float(np.nanmax(finite_bpp)) * 1.05
    grid = np.linspace(bpp_min, bpp_max, n_grid)

    case_keys = sorted({row["case_key"] for row in collapsed})
    series_keys = _series_order(collapsed)
    colors = _series_color_map(series_keys)

    fig = Figure(figsize=(9.2, 5.6), facecolor="#ffffff")
    ax = fig.add_subplot(1, 1, 1, facecolor="#ffffff")
    has_series = False
    for series_key in series_keys:
        curves: list[np.ndarray] = []
        label = series_key
        for case_key in case_keys:
            pts = [
                row
                for row in collapsed
                if row["case_key"] == case_key and _series_key(row) == series_key
            ]
            if len(pts) < 2:
                continue
            pts = sorted(pts, key=lambda row: _float(row, "bpp"))
            label = _series_label(pts[0])
            bpp = np.array([_float(row, "bpp") for row in pts], dtype=float)
            psnr = np.array([min(_float(row, "psnr_db"), 99.0) for row in pts], dtype=float)
            curves.append(_interp_sorted_x(bpp, psnr, grid))
        if not curves:
            continue
        mat = np.vstack(curves)
        valid = np.isfinite(mat)
        counts = valid.sum(axis=0)
        mean = np.full(grid.shape, np.nan, dtype=float)
        ci95 = np.zeros(grid.shape, dtype=float)
        for col_idx, count in enumerate(counts):
            if count == 0:
                continue
            values = mat[valid[:, col_idx], col_idx]
            mean[col_idx] = float(np.mean(values))
            if count > 1:
                ci95[col_idx] = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(float(count))
        ax.plot(grid, mean, label=label, color=colors[series_key], linewidth=2.0)
        has_series = True
        if np.any(counts > 1):
            ax.fill_between(grid, mean - ci95, mean + ci95, color=colors[series_key], alpha=0.16)

    _style_axis(
        ax,
        "Средняя зависимость PSNR от bpp по набору изображений",
        "Бит на пиксель (bpp)",
        "PSNR, дБ",
    )
    if has_series:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, output_path)


def _reference_rows(
    rows: Sequence[dict[str, str]],
    *,
    baseline_quality_ref: float,
    jpeg2000_quality_ref: float,
    imcs_ratio_ref: float,
) -> list[dict[str, str]]:
    collapsed = _collapse_repeats(rows)
    selected: list[dict[str, str]] = []
    for row in collapsed:
        series_key = _series_key(row)
        value = _param_float(row)
        if series_key.startswith("imcs"):
            if abs(value - imcs_ratio_ref) < 1e-9:
                selected.append(row)
        elif series_key == "jpeg2000":
            if abs(value - jpeg2000_quality_ref) < 1e-9:
                selected.append(row)
        elif abs(value - baseline_quality_ref) < 1e-9:
            selected.append(row)
    return selected


def plot_encode_decode_bars(
    rows: Sequence[dict[str, str]],
    output_path: Path,
    *,
    baseline_quality_ref: int = 70,
    jpeg2000_quality_ref: int = 40,
    imcs_ratio_ref: float = 0.08,
) -> None:
    """Encoding and decoding time at representative parameter points."""

    selected = _reference_rows(
        rows,
        baseline_quality_ref=baseline_quality_ref,
        jpeg2000_quality_ref=jpeg2000_quality_ref,
        imcs_ratio_ref=imcs_ratio_ref,
    )
    if not selected:
        return

    series_keys = _series_order(selected)
    colors = _series_color_map(series_keys)
    labels = []
    enc_means = []
    enc_ci = []
    dec_means = []
    dec_ci = []
    for series_key in series_keys:
        pts = [row for row in selected if _series_key(row) == series_key]
        if not pts:
            continue
        labels.append(_series_label(pts[0]))
        enc_mean, enc_interval, _ = _mean_ci(_float(row, "encode_time_s") for row in pts)
        dec_mean, dec_interval, _ = _mean_ci(_float(row, "decode_time_s") for row in pts)
        enc_means.append(enc_mean)
        enc_ci.append(enc_interval)
        dec_means.append(dec_mean)
        dec_ci.append(dec_interval)

    x = np.arange(len(labels))
    width = 0.36
    fig = Figure(figsize=(max(8.0, 1.15 * len(labels)), 5.2), facecolor="#ffffff")
    ax = fig.add_subplot(1, 1, 1, facecolor="#ffffff")
    ax.bar(
        x - width / 2,
        enc_means,
        width=width,
        yerr=enc_ci,
        capsize=4,
        label="Кодирование",
        color="#2f6fed",
    )
    ax.bar(
        x + width / 2,
        dec_means,
        width=width,
        yerr=dec_ci,
        capsize=4,
        label="Декодирование",
        color="#12b76a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_axis(ax, "Среднее время работы методов", "Метод", "Время, с")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, output_path)


def plot_quality_time_tradeoff(
    rows: Sequence[dict[str, str]],
    output_path: Path,
) -> None:
    """PSNR against decoding time for all parameter points."""

    collapsed = _collapse_repeats(rows)
    if not collapsed:
        return
    series_keys = _series_order(collapsed)
    colors = _series_color_map(series_keys)

    fig = Figure(figsize=(9.2, 5.4), facecolor="#ffffff")
    ax = fig.add_subplot(1, 1, 1, facecolor="#ffffff")
    for series_key in series_keys:
        pts = [row for row in collapsed if _series_key(row) == series_key]
        if not pts:
            continue
        x = np.array([max(_float(row, "decode_time_s"), 1e-7) for row in pts], dtype=float)
        y = np.array([min(_float(row, "psnr_db"), 99.0) for row in pts], dtype=float)
        marker = "s" if series_key.startswith("imcs") else "o"
        ax.scatter(
            x,
            y,
            s=28,
            marker=marker,
            alpha=0.78,
            color=colors[series_key],
            label=_series_label(pts[0]),
        )

    ax.set_xscale("log")
    _style_axis(ax, "Компромисс качества и времени декодирования", "Время декодирования, с", "PSNR, дБ")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, output_path)


def write_markdown_tables(
    rows: Sequence[dict[str, str]],
    output_path: Path,
    *,
    baseline_quality_ref: int = 70,
    jpeg2000_quality_ref: int = 40,
    imcs_ratio_ref: float = 0.08,
) -> None:
    selected = _reference_rows(
        rows,
        baseline_quality_ref=baseline_quality_ref,
        jpeg2000_quality_ref=jpeg2000_quality_ref,
        imcs_ratio_ref=imcs_ratio_ref,
    )
    collapsed = _collapse_repeats(rows)

    lines: list[str] = []
    lines.append("# Таблицы для раздела сравнения с JPEG\n\n")
    lines.append("Значения представлены как среднее ± 95% доверительный интервал по изображениям.\n\n")
    lines.append(
        f"Репрезентативные точки: JPEG/WebP quality={baseline_quality_ref}, "
        f"JPEG2000 quality={jpeg2000_quality_ref}, IMCS compression_ratio={imcs_ratio_ref}.\n\n"
    )

    lines.append("## Сводная таблица по методам\n\n")
    lines.append("| Метод | bpp | PSNR, дБ | SSIM | MAE | Кодирование, с | Декодирование, с |\n")
    lines.append("|-------|-----|----------|------|-----|----------------|------------------|\n")
    for series_key in _series_order(selected):
        pts = [row for row in selected if _series_key(row) == series_key]
        if not pts:
            continue
        lines.append(
            f"| {_series_label(pts[0])} | "
            f"{_format_mean_ci((_float(r, 'bpp') for r in pts), 3)} | "
            f"{_format_mean_ci((_float(r, 'psnr_db') for r in pts), 2)} | "
            f"{_format_mean_ci((_float(r, 'ssim') for r in pts), 4)} | "
            f"{_format_mean_ci((_float(r, 'mae') for r in pts), 3)} | "
            f"{_format_mean_ci((_float(r, 'encode_time_s') for r in pts), 5)} | "
            f"{_format_mean_ci((_float(r, 'decode_time_s') for r in pts), 5)} |\n"
        )

    lines.append("\n## Репрезентативные точки по каждому изображению\n\n")
    lines.append("| Изображение | Метод | bpp | PSNR, дБ | SSIM | Кодирование, с | Декодирование, с |\n")
    lines.append("|-------------|-------|-----|----------|------|----------------|------------------|\n")
    for row in sorted(selected, key=lambda item: (item["case_key"], _series_label(item))):
        lines.append(
            f"| {row['case_key']} | {_series_label(row)} | {_float(row, 'bpp'):.3f} | "
            f"{_float(row, 'psnr_db'):.2f} | {_float(row, 'ssim'):.4f} | "
            f"{_float(row, 'encode_time_s'):.5f} | {_float(row, 'decode_time_s'):.5f} |\n"
        )

    lines.append("\n## Полный список серий на графиках\n\n")
    lines.append("| Серия | Метод | Параметр | Число точек |\n")
    lines.append("|-------|-------|----------|-------------|\n")
    for series_key in _series_order(collapsed):
        pts = [row for row in collapsed if _series_key(row) == series_key]
        if not pts:
            continue
        param_values = sorted({_float(row, "param_value") for row in pts})
        lines.append(
            f"| {_series_label(pts[0])} | {pts[0]['method']} | "
            f"{pts[0]['param_name']}={param_values} | {len(pts)} |\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def generate_all_figures_and_tables(
    benchmark_dir: Path,
    *,
    max_facets: int = 8,
    jpeg_quality_ref: int = 70,
    baseline_quality_ref: int | None = None,
    jpeg2000_quality_ref: int = 40,
    imcs_ratio_ref: float = 0.08,
) -> None:
    csv_path = benchmark_dir / "rd_long.csv"
    if not csv_path.is_file():
        return
    rows = read_rd_long_csv(csv_path)
    if not rows:
        return
    quality_ref = int(jpeg_quality_ref if baseline_quality_ref is None else baseline_quality_ref)
    plot_rd_facets_overlay(
        rows,
        benchmark_dir / "fig_5_7_psnr_bpp_by_image.png",
        max_images=max_facets,
    )
    plot_rd_mean_with_ci(rows, benchmark_dir / "fig_5_8_mean_psnr_bpp_ci.png")
    plot_encode_decode_bars(
        rows,
        benchmark_dir / "fig_5_9_encode_decode_time_ci.png",
        baseline_quality_ref=quality_ref,
        jpeg2000_quality_ref=jpeg2000_quality_ref,
        imcs_ratio_ref=imcs_ratio_ref,
    )
    plot_quality_time_tradeoff(rows, benchmark_dir / "fig_5_10_quality_time_tradeoff.png")
    write_markdown_tables(
        rows,
        benchmark_dir / "tables_for_thesis.md",
        baseline_quality_ref=quality_ref,
        jpeg2000_quality_ref=jpeg2000_quality_ref,
        imcs_ratio_ref=imcs_ratio_ref,
    )
