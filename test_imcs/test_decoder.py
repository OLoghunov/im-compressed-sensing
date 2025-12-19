"""Tests for IMCS Decoder."""

import pytest
from imcs.decoder import IMCSDecoder


@pytest.mark.parametrize(
    "algorithm",
    [
        "omp",
        "basis_pursuit",
        "iterative_threshold",
        "simulated_annealing",
    ],
)
def test_decoder_initialization(algorithm):
    decoder = IMCSDecoder(reconstruction_algorithm=algorithm)
    assert decoder.reconstruction_algorithm == algorithm


@pytest.mark.parametrize(
    "invalid_algorithm",
    [
        "invalid_algo",
        "random_algorithm",
        "compressed_sensing",
        "",
        "OMP",
    ],
)
def test_decoder_invalid_algorithm(invalid_algorithm):
    with pytest.raises(ValueError):
        IMCSDecoder(reconstruction_algorithm=invalid_algorithm)


@pytest.mark.parametrize(
    "initial_algo,new_algo",
    [
        ("omp", "basis_pursuit"),
        ("basis_pursuit", "iterative_threshold"),
        ("iterative_threshold", "omp"),
        ("omp", "iterative_threshold"),
        ("simulated_annealing", "omp"),
        ("omp", "simulated_annealing"),
    ],
)
def test_set_reconstruction_algorithm(initial_algo, new_algo):
    decoder = IMCSDecoder(reconstruction_algorithm=initial_algo)
    decoder.set_reconstruction_algorithm(new_algo)
    assert decoder.reconstruction_algorithm == new_algo


@pytest.mark.parametrize(
    "invalid_algorithm",
    [
        "invalid",
        "wrong_algo",
        "lasso",
        "ridge",
    ],
)
def test_set_invalid_reconstruction_algorithm(invalid_algorithm):
    decoder = IMCSDecoder(reconstruction_algorithm="omp")
    with pytest.raises(ValueError):
        decoder.set_reconstruction_algorithm(invalid_algorithm)


@pytest.mark.parametrize(
    "algorithm", ["omp", "basis_pursuit", "iterative_threshold", "simulated_annealing"]
)
def test_decode_with_different_algorithms(algorithm):
    """Test that decoding works with all supported algorithms."""
    import numpy as np
    from imcs import IMCSEncoder

    encoder = IMCSEncoder(compression_ratio=0.5, seed=42)
    decoder = IMCSDecoder(reconstruction_algorithm=algorithm)

    # Create sparse 1D signal
    data = np.zeros(64)
    data[[5, 15, 30, 45]] = [1.0, 2.0, 1.5, 0.8]

    compressed = encoder.encode(data)
    reconstructed = decoder.decode(compressed)

    assert reconstructed.shape == data.shape
    assert isinstance(reconstructed, np.ndarray)


@pytest.mark.skip(reason="File decoding requires test files setup")
def test_decode_file():
    decoder = IMCSDecoder(reconstruction_algorithm="omp")
    pass


def test_decode_invalid_data():
    """Test that decoding invalid data raises appropriate errors."""
    decoder = IMCSDecoder(reconstruction_algorithm="omp")

    with pytest.raises(ValueError):
        decoder.decode(b"invalid data")

    with pytest.raises(ValueError):
        decoder.decode(b"")


@pytest.mark.parametrize("data_size", [32, 64, 128])
def test_decode_different_sizes(data_size):
    """Test encoding and decoding signals of different sizes."""
    import numpy as np
    from imcs import IMCSEncoder

    encoder = IMCSEncoder(compression_ratio=0.5, seed=42)
    decoder = IMCSDecoder(reconstruction_algorithm="omp")

    # Create sparse signal (5% non-zero)
    data = np.zeros(data_size)
    num_nonzero = max(3, data_size // 20)
    indices = np.random.choice(data_size, num_nonzero, replace=False)
    data[indices] = np.random.rand(num_nonzero) * 2

    compressed = encoder.encode(data)
    reconstructed = decoder.decode(compressed)

    assert reconstructed.shape == data.shape
