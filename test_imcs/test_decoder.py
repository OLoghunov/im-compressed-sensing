"""Tests for IMCS Decoder."""

import pytest
from imcs.decoder import IMCSDecoder


@pytest.mark.parametrize(
    "algorithm",
    [
        "omp",
        "basis_pursuit",
        "iterative_threshold",
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


@pytest.mark.skip(reason="Decoding algorithm not yet implemented")
@pytest.mark.parametrize("algorithm", ["omp", "basis_pursuit", "iterative_threshold"])
def test_decode_with_different_algorithms(algorithm):
    decoder = IMCSDecoder(reconstruction_algorithm=algorithm)
    pass


@pytest.mark.skip(reason="File decoding not yet implemented")
def test_decode_file():
    decoder = IMCSDecoder(reconstruction_algorithm="omp")
    pass


@pytest.mark.skip(reason="Decoding algorithm not yet implemented")
def test_decode_empty_data():
    decoder = IMCSDecoder(reconstruction_algorithm="omp")
    pass


@pytest.mark.skip(reason="Decoding algorithm not yet implemented")
@pytest.mark.parametrize("data_size", [50, 100, 256, 512])
def test_decode_different_sizes(data_size):
    decoder = IMCSDecoder(reconstruction_algorithm="omp")
    pass
