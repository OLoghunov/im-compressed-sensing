"""Tests for IMCS utility functions."""

import pytest
import numpy as np
from imcs import utils


@pytest.mark.skip(reason="Measurement matrix generation not yet implemented")
@pytest.mark.parametrize(
    "m,n",
    [
        (50, 100),
        (30, 100),
        (70, 100),
        (100, 256),
        (128, 512),
    ],
)
def test_generate_measurement_matrix_shape(m, n):
    matrix = utils.generate_measurement_matrix(m, n)
    assert matrix.shape == (m, n)


@pytest.mark.skip(reason="Measurement matrix generation not yet implemented")
@pytest.mark.parametrize(
    "matrix_type",
    [
        "gaussian",
        "bernoulli",
        "random",
    ],
)
def test_generate_different_matrix_types(matrix_type):
    matrix = utils.generate_measurement_matrix(50, 100, matrix_type=matrix_type)
    assert matrix is not None
    assert matrix.shape == (50, 100)


@pytest.mark.skip(reason="Measurement matrix generation not yet implemented")
@pytest.mark.parametrize(
    "m,n,matrix_type",
    [
        (25, 64, "gaussian"),
        (32, 128, "bernoulli"),
        (40, 100, "random"),
        (50, 200, "gaussian"),
    ],
)
def test_generate_matrix_combinations(m, n, matrix_type):
    matrix = utils.generate_measurement_matrix(m, n, matrix_type=matrix_type)
    assert matrix.shape == (m, n)


@pytest.mark.skip(reason="Sparsity computation not yet implemented")
@pytest.mark.parametrize("signal_size", [50, 100, 256, 512, 1024])
def test_compute_sparsity_dense_signal(signal_size):
    signal = np.random.rand(signal_size)
    sparsity = utils.compute_sparsity(signal)
    assert sparsity > 0
    assert sparsity <= signal_size


@pytest.mark.skip(reason="Sparsity computation not yet implemented")
@pytest.mark.parametrize(
    "signal_size,num_nonzero",
    [
        (100, 1),
        (100, 2),
        (100, 5),
        (100, 10),
        (100, 20),
        (256, 10),
        (512, 25),
    ],
)
def test_compute_sparsity_sparse_signal(signal_size, num_nonzero):
    signal = np.zeros(signal_size)
    indices = np.random.choice(signal_size, num_nonzero, replace=False)
    signal[indices] = np.random.rand(num_nonzero)
    sparsity = utils.compute_sparsity(signal)
    assert sparsity == num_nonzero


@pytest.mark.skip(reason="Sparsity computation not yet implemented")
@pytest.mark.parametrize("threshold", [1e-10, 1e-8, 1e-6, 1e-4])
def test_compute_sparsity_with_thresholds(threshold):
    signal = np.array([1.0, 1e-9, 1e-7, 1e-5, 1e-3, 2.0])
    sparsity = utils.compute_sparsity(signal, threshold=threshold)
    assert sparsity >= 2


@pytest.mark.skip(reason="Format validation not yet implemented")
def test_validate_valid_imcs_format():
    pass


@pytest.mark.skip(reason="Format validation not yet implemented")
@pytest.mark.parametrize(
    "invalid_data",
    [
        b"invalid data",
        b"",
        b"random bytes",
        b"\x00\x00\x00",
        bytes(100),
    ],
)
def test_validate_invalid_imcs_format(invalid_data):
    result = utils.validate_imcs_format(invalid_data)
    assert result is False


@pytest.mark.skip(reason="Metrics calculation not yet implemented")
@pytest.mark.parametrize("signal_size", [64, 128, 256, 512])
def test_calculate_metrics_identical_signals(signal_size):
    np.random.seed(42)
    signal = np.random.rand(signal_size)
    metrics = utils.calculate_compression_metrics(signal, signal)
    assert metrics is not None
    assert "mse" in metrics or "MSE" in metrics


@pytest.mark.skip(reason="Metrics calculation not yet implemented")
@pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.1, 0.2])
def test_calculate_metrics_noisy_signals(noise_level):
    np.random.seed(42)
    original = np.random.rand(100)
    reconstructed = original + np.random.rand(100) * noise_level
    metrics = utils.calculate_compression_metrics(original, reconstructed)
    assert metrics is not None


@pytest.mark.skip(reason="Metrics calculation not yet implemented")
@pytest.mark.parametrize(
    "signal_size,noise_level",
    [
        (64, 0.01),
        (128, 0.05),
        (256, 0.1),
        (512, 0.15),
    ],
)
def test_calculate_metrics_combinations(signal_size, noise_level):
    np.random.seed(42)
    original = np.random.rand(signal_size)
    reconstructed = original + np.random.rand(signal_size) * noise_level
    metrics = utils.calculate_compression_metrics(original, reconstructed)
    assert metrics is not None
