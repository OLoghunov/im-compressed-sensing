"""Tests for IMCS Encoder."""

import pytest
import numpy as np
from imcs.encoder import IMCSEncoder


@pytest.mark.parametrize(
    "compression_ratio,sparsity_basis",
    [
        (0.5, "dct"),
        (0.3, "wavelet"),
        (0.7, "dct"),
        (0.1, "dct"),
        (0.9, "wavelet"),
    ],
)
def test_encoder_initialization(compression_ratio, sparsity_basis):
    encoder = IMCSEncoder(compression_ratio=compression_ratio, sparsity_basis=sparsity_basis)
    assert encoder.compression_ratio == compression_ratio
    assert encoder.sparsity_basis == sparsity_basis


@pytest.mark.parametrize("invalid_ratio", [1.5, 0, -0.5, 1.0, 2.0])
def test_encoder_invalid_compression_ratio(invalid_ratio):
    with pytest.raises(ValueError):
        IMCSEncoder(compression_ratio=invalid_ratio)


@pytest.mark.parametrize(
    "initial_ratio,new_ratio",
    [
        (0.5, 0.3),
        (0.7, 0.2),
        (0.3, 0.8),
        (0.1, 0.9),
    ],
)
def test_set_compression_ratio(initial_ratio, new_ratio):
    encoder = IMCSEncoder(compression_ratio=initial_ratio)
    encoder.set_compression_ratio(new_ratio)
    assert encoder.compression_ratio == new_ratio


@pytest.mark.parametrize("invalid_ratio", [1.2, 0, -0.3, 1.0])
def test_set_invalid_compression_ratio(invalid_ratio):
    encoder = IMCSEncoder(compression_ratio=0.5)
    with pytest.raises(ValueError):
        encoder.set_compression_ratio(invalid_ratio)


@pytest.mark.skip(reason="Encoding algorithm not yet implemented")
@pytest.mark.parametrize(
    "data_size,compression_ratio",
    [
        (100, 0.5),
        (256, 0.3),
        (512, 0.7),
        (1024, 0.2),
    ],
)
def test_encode_simple_data(data_size, compression_ratio):
    encoder = IMCSEncoder(compression_ratio=compression_ratio)
    data = np.random.rand(data_size)
    compressed = encoder.encode(data)
    assert isinstance(compressed, bytes)
    assert len(compressed) < data.nbytes


@pytest.mark.skip(reason="File encoding not yet implemented")
def test_encode_file():
    encoder = IMCSEncoder(compression_ratio=0.5)
    pass


@pytest.mark.skip(reason="Encoding algorithm not yet implemented")
@pytest.mark.parametrize("data_size", [10, 100, 500, 1000])
def test_encode_zero_array(data_size):
    encoder = IMCSEncoder(compression_ratio=0.5)
    data = np.zeros(data_size)
    compressed = encoder.encode(data)
    assert isinstance(compressed, bytes)


@pytest.mark.skip(reason="Encoding algorithm not yet implemented")
@pytest.mark.parametrize(
    "data_size,sparsity_level",
    [
        (100, 2),
        (100, 5),
        (100, 10),
        (100, 20),
    ],
)
def test_encode_sparse_data(data_size, sparsity_level):
    encoder = IMCSEncoder(compression_ratio=0.5)
    data = np.zeros(data_size)
    indices = np.random.choice(data_size, sparsity_level, replace=False)
    data[indices] = np.random.rand(sparsity_level)
    compressed = encoder.encode(data)
    assert isinstance(compressed, bytes)
