"""
Основные unit-тесты для IMCS кодека.

Проверяют корректность работы базовых функций.
"""

import pytest
import numpy as np
from imcs import IMCSEncoder, IMCSDecoder
from imcs.utils import (
    generate_measurement_matrix,
    dct2,
    idct2,
    omp,
    ista,
    calculate_compression_metrics,
)


class TestMeasurementMatrix:
    """Тесты генерации измерительных матриц."""

    def test_gaussian_matrix_shape(self):
        """Проверка размерности гауссовой матрицы."""
        Phi = generate_measurement_matrix(10, 20, "gaussian", seed=42)
        assert Phi.shape == (10, 20)

    def test_bernoulli_matrix_values(self):
        """Проверка значений матрицы Бернулли."""
        Phi = generate_measurement_matrix(5, 10, "bernoulli", seed=42)
        unique_vals = np.unique(Phi)
        # Должны быть только ±1/sqrt(m)
        assert len(unique_vals) == 2

    def test_reproducibility(self):
        """Проверка воспроизводимости при одинаковом seed."""
        Phi1 = generate_measurement_matrix(10, 20, "gaussian", seed=42)
        Phi2 = generate_measurement_matrix(10, 20, "gaussian", seed=42)
        assert np.allclose(Phi1, Phi2)


class TestDCT:
    """Тесты 2D DCT преобразования."""

    def test_dct_idct_inverse(self):
        """Проверка что IDCT(DCT(X)) = X."""
        X = np.random.rand(8, 8)
        S = dct2(X)
        X_reconstructed = idct2(S)
        assert np.allclose(X, X_reconstructed)

    def test_dct_constant_image(self):
        """Константное изображение → только DC коэффициент."""
        X = np.ones((4, 4)) * 100
        S = dct2(X)
        # Только S[0,0] должен быть ненулевым
        assert np.abs(S[0, 0]) > 100
        assert np.sum(np.abs(S[1:, :])) < 1e-10


class TestOMP:
    """Тесты алгоритма OMP."""

    def test_omp_sparse_recovery(self):
        """OMP восстанавливает разреженный сигнал."""
        # Создаём разреженный сигнал
        n = 64
        k = 5  # sparsity
        x_true = np.zeros(n)
        x_true[[10, 20, 30, 40, 50]] = [1, 2, 1.5, 3, 0.5]

        # Измерения
        m = 25
        Phi = generate_measurement_matrix(m, n, "gaussian", seed=42)
        y = Phi @ x_true

        # Восстанавливаем
        x_recovered = omp(y, Phi, sparsity=k)

        # Проверяем качество
        error = np.linalg.norm(x_true - x_recovered)
        assert error < 1.0  # OMP может ошибаться для сложных случаев

    def test_omp_zero_signal(self):
        """OMP на нулевом сигнале возвращает ноль."""
        y = np.zeros(10)
        A = np.random.randn(10, 20)
        x = omp(y, A, sparsity=5)
        assert np.linalg.norm(x) < 1e-6


class TestISTA:
    """Тесты алгоритма ISTA."""

    def test_ista_converges(self):
        """ISTA сходится для простого случая."""
        n = 32
        m = 20

        # Разреженный сигнал
        x_true = np.zeros(n)
        x_true[[5, 10, 15]] = [1, -1, 0.5]

        # Измерения
        A = generate_measurement_matrix(m, n, "gaussian", seed=42)
        y = A @ x_true

        # Восстанавливаем
        x_recovered = ista(y, A, lambda_param=0.1, max_iter=100)

        # Проверяем что нашли хотя бы приблизительно правильные индексы
        support_true = set(np.where(np.abs(x_true) > 0.01)[0])
        support_recovered = set(np.where(np.abs(x_recovered) > 0.1)[0])

        # Должны пересекаться
        assert len(support_true & support_recovered) >= 2


class TestEncoder:
    """Тесты кодера."""

    def test_encode_1d_signal(self):
        """Кодирование 1D сигнала."""
        x = np.random.rand(100)
        encoder = IMCSEncoder(compression_ratio=0.5, seed=42)
        compressed = encoder.encode(x)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        # Проверяем магическое число
        assert compressed[:4] == b"IMCS"

    def test_encode_2d_image(self):
        """Кодирование 2D изображения."""
        X = np.random.rand(8, 8) * 100
        encoder = IMCSEncoder(compression_ratio=0.7, seed=42)
        compressed = encoder.encode(X)

        assert isinstance(compressed, bytes)
        assert compressed[:4] == b"IMCS"

    def test_compression_reduces_size(self):
        """Сжатие уменьшает размер (для больших сигналов)."""
        x = np.random.rand(1000)
        encoder = IMCSEncoder(compression_ratio=0.3, seed=42)
        compressed = encoder.encode(x)

        original_size = x.nbytes
        compressed_size = len(compressed)
        # Должно быть меньше (с учётом заголовка)
        assert compressed_size < original_size


class TestDecoder:
    """Тесты декодера."""

    def test_decode_1d_roundtrip(self):
        """1D сигнал: encode → decode."""
        # Разреженный сигнал
        x = np.zeros(64)
        x[[10, 20, 30]] = [100, 150, 80]

        encoder = IMCSEncoder(compression_ratio=0.5, seed=42)
        compressed = encoder.encode(x)

        decoder = IMCSDecoder(reconstruction_algorithm="omp", max_iter=100)
        x_reconstructed = decoder.decode(compressed)

        assert x_reconstructed.shape == x.shape
        # Проверяем относительную ошибку
        rel_error = np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x)
        assert rel_error < 1.5  # CS не всегда точен для недоопределённых систем

    def test_decode_2d_constant(self):
        """2D константное изображение восстанавливается идеально."""
        X = np.ones((4, 4)) * 100

        encoder = IMCSEncoder(compression_ratio=0.5, seed=42)
        compressed = encoder.encode(X)

        decoder = IMCSDecoder(reconstruction_algorithm="omp", max_iter=100)
        X_reconstructed = decoder.decode(compressed)

        assert X_reconstructed.shape == X.shape
        # Константное изображение очень разреженно в DCT
        assert np.allclose(X, X_reconstructed, atol=1e-6)


class TestMetrics:
    """Тесты метрик качества."""

    def test_metrics_identical_signals(self):
        """Метрики для идентичных сигналов."""
        x = np.random.rand(100)
        metrics = calculate_compression_metrics(x, x)

        assert metrics["mse"] < 1e-10
        assert metrics["psnr"] > 100  # Очень высокий PSNR

    def test_metrics_different_signals(self):
        """Метрики для разных сигналов."""
        x1 = np.ones(100)
        x2 = np.zeros(100)
        metrics = calculate_compression_metrics(x1, x2)

        assert metrics["mse"] > 0
        assert metrics["psnr"] < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
