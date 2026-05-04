"""Общие величины для сравнения RD (IMCS, JPEG, …)."""

from __future__ import annotations


def bpp_from_compressed_size(num_bytes: int, width: int, height: int) -> float:
    """Биты на пространственный пиксель: 8 * размер_потока / (H·W)."""
    if width <= 0 or height <= 0:
        return float("nan")
    return 8.0 * float(num_bytes) / float(height * width)
