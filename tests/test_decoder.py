"""
Tests for IMCS Decoder.
"""

import pytest
import numpy as np
from imcs.decoder import IMCSDecoder


class TestIMCSDecoder:
    """Test suite for IMCSDecoder class."""
    
    def test_decoder_initialization(self):
        """Test that decoder initializes with valid parameters."""
        decoder = IMCSDecoder(reconstruction_algorithm='omp')
        assert decoder.reconstruction_algorithm == 'omp'
    
    def test_decoder_invalid_algorithm(self):
        """Test that decoder raises error for invalid algorithm."""
        with pytest.raises(ValueError):
            IMCSDecoder(reconstruction_algorithm='invalid_algo')
    
    def test_valid_reconstruction_algorithms(self):
        """Test that all valid algorithms can be set."""
        algorithms = ['omp', 'basis_pursuit', 'iterative_threshold']
        for algo in algorithms:
            decoder = IMCSDecoder(reconstruction_algorithm=algo)
            assert decoder.reconstruction_algorithm == algo
    
    def test_set_reconstruction_algorithm(self):
        """Test updating reconstruction algorithm."""
        decoder = IMCSDecoder(reconstruction_algorithm='omp')
        decoder.set_reconstruction_algorithm('basis_pursuit')
        assert decoder.reconstruction_algorithm == 'basis_pursuit'
    
    def test_set_invalid_reconstruction_algorithm(self):
        """Test that setting invalid algorithm raises error."""
        decoder = IMCSDecoder(reconstruction_algorithm='omp')
        with pytest.raises(ValueError):
            decoder.set_reconstruction_algorithm('invalid')
    
    @pytest.mark.skip(reason="Decoding algorithm not yet implemented")
    def test_decode_simple_data(self):
        """Test decoding of compressed data."""
        decoder = IMCSDecoder(reconstruction_algorithm='omp')
        # TODO: Need valid compressed data to test
        pass
    
    @pytest.mark.skip(reason="File decoding not yet implemented")
    def test_decode_file(self):
        """Test decoding of a file."""
        decoder = IMCSDecoder(reconstruction_algorithm='omp')
        # TODO: Implement when decode_file is ready
        pass


class TestDecoderEdgeCases:
    """Test edge cases for decoder."""
    
    @pytest.mark.skip(reason="Decoding algorithm not yet implemented")
    def test_decode_empty_data(self):
        """Test decoding of empty compressed data."""
        decoder = IMCSDecoder(reconstruction_algorithm='omp')
        # TODO: Implement when decode is ready
        pass

