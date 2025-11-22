"""
Pytest configuration and shared fixtures for IMCS tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_signal_1d():
    """Generate a sample 1D signal for testing."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture(params=[10, 50, 100, 200])
def sample_signal_1d_parametrized(request):
    """Generate sample 1D signals of different sizes."""
    np.random.seed(42)
    return np.random.rand(request.param)


@pytest.fixture
def sample_sparse_signal():
    """Generate a sparse signal for testing."""
    signal = np.zeros(100)
    signal[10] = 1.0
    signal[25] = 2.5
    signal[50] = -1.5
    signal[75] = 3.0
    return signal


@pytest.fixture(params=[2, 5, 10, 20])
def sample_sparse_signal_parametrized(request):
    """Generate sparse signals with different sparsity levels."""
    signal_size = 100
    num_nonzero = request.param
    signal = np.zeros(signal_size)
    np.random.seed(42)
    indices = np.random.choice(signal_size, num_nonzero, replace=False)
    signal[indices] = np.random.rand(num_nonzero)
    return signal


@pytest.fixture
def sample_signal_2d():
    """Generate a sample 2D signal (image) for testing."""
    np.random.seed(42)
    return np.random.rand(64, 64)


@pytest.fixture(params=[(32, 32), (64, 64), (128, 128)])
def sample_signal_2d_parametrized(request):
    """Generate sample 2D signals of different sizes."""
    np.random.seed(42)
    return np.random.rand(*request.param)


@pytest.fixture
def encoder_default():
    """Create a default encoder instance."""
    from imcs.encoder import IMCSEncoder

    return IMCSEncoder(compression_ratio=0.5, sparsity_basis="dct")


@pytest.fixture(params=[0.3, 0.5, 0.7])
def encoder_parametrized(request):
    """Create encoder instances with different compression ratios."""
    from imcs.encoder import IMCSEncoder

    return IMCSEncoder(compression_ratio=request.param, sparsity_basis="dct")


@pytest.fixture
def decoder_default():
    """Create a default decoder instance."""
    from imcs.decoder import IMCSDecoder

    return IMCSDecoder(reconstruction_algorithm="omp")


@pytest.fixture(params=["omp", "basis_pursuit", "iterative_threshold"])
def decoder_parametrized(request):
    """Create decoder instances with different reconstruction algorithms."""
    from imcs.decoder import IMCSDecoder

    return IMCSDecoder(reconstruction_algorithm=request.param)


# Test data combinations
@pytest.fixture(
    params=[
        (0.3, "dct"),
        (0.5, "dct"),
        (0.5, "wavelet"),
        (0.7, "wavelet"),
    ]
)
def encoder_configs(request):
    """Various encoder configurations for testing."""
    compression_ratio, sparsity_basis = request.param
    return {"compression_ratio": compression_ratio, "sparsity_basis": sparsity_basis}


@pytest.fixture(
    params=[
        ("omp", 0.5),
        ("basis_pursuit", 0.5),
        ("iterative_threshold", 0.3),
    ]
)
def codec_pair(request):
    """Create encoder-decoder pairs with compatible settings."""
    from imcs.encoder import IMCSEncoder
    from imcs.decoder import IMCSDecoder

    algorithm, compression_ratio = request.param
    encoder = IMCSEncoder(compression_ratio=compression_ratio)
    decoder = IMCSDecoder(reconstruction_algorithm=algorithm)

    return encoder, decoder
