"""
Pytest configuration and shared fixtures for IMCS tests.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_signal_1d():
    """Generate a sample 1D signal for testing."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture
def sample_sparse_signal():
    """Generate a sparse signal for testing."""
    signal = np.zeros(100)
    signal[10] = 1.0
    signal[25] = 2.5
    signal[50] = -1.5
    signal[75] = 3.0
    return signal


@pytest.fixture
def sample_signal_2d():
    """Generate a sample 2D signal (image) for testing."""
    np.random.seed(42)
    return np.random.rand(64, 64)


@pytest.fixture
def encoder_default():
    """Create a default encoder instance."""
    from imcs.encoder import IMCSEncoder
    return IMCSEncoder(compression_ratio=0.5, sparsity_basis='dct')


@pytest.fixture
def decoder_default():
    """Create a default decoder instance."""
    from imcs.decoder import IMCSDecoder
    return IMCSDecoder(reconstruction_algorithm='omp')

