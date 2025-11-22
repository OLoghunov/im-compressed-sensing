"""
Tests for IMCS Encoder.
"""

import pytest
import numpy as np
from imcs.encoder import IMCSEncoder


class TestIMCSEncoder:
    """Test suite for IMCSEncoder class."""
    
    def test_encoder_initialization(self):
        """Test that encoder initializes with valid parameters."""
        encoder = IMCSEncoder(compression_ratio=0.5, sparsity_basis='dct')
        assert encoder.compression_ratio == 0.5
        assert encoder.sparsity_basis == 'dct'
    
    def test_encoder_invalid_compression_ratio(self):
        """Test that encoder raises error for invalid compression ratio."""
        with pytest.raises(ValueError):
            IMCSEncoder(compression_ratio=1.5)
        
        with pytest.raises(ValueError):
            IMCSEncoder(compression_ratio=0)
        
        with pytest.raises(ValueError):
            IMCSEncoder(compression_ratio=-0.5)
    
    def test_set_compression_ratio(self):
        """Test updating compression ratio."""
        encoder = IMCSEncoder(compression_ratio=0.5)
        encoder.set_compression_ratio(0.3)
        assert encoder.compression_ratio == 0.3
    
    def test_set_invalid_compression_ratio(self):
        """Test that setting invalid compression ratio raises error."""
        encoder = IMCSEncoder(compression_ratio=0.5)
        with pytest.raises(ValueError):
            encoder.set_compression_ratio(1.2)
    
    @pytest.mark.skip(reason="Encoding algorithm not yet implemented")
    def test_encode_simple_data(self):
        """Test encoding of simple data."""
        encoder = IMCSEncoder(compression_ratio=0.5)
        data = np.random.rand(100)
        compressed = encoder.encode(data)
        assert isinstance(compressed, bytes)
        assert len(compressed) < data.nbytes
    
    @pytest.mark.skip(reason="File encoding not yet implemented")
    def test_encode_file(self):
        """Test encoding of a file."""
        encoder = IMCSEncoder(compression_ratio=0.5)
        # TODO: Implement when encode_file is ready
        pass


class TestEncoderEdgeCases:
    """Test edge cases for encoder."""
    
    @pytest.mark.skip(reason="Encoding algorithm not yet implemented")
    def test_encode_zero_array(self):
        """Test encoding of array with all zeros."""
        encoder = IMCSEncoder(compression_ratio=0.5)
        data = np.zeros(100)
        compressed = encoder.encode(data)
        assert isinstance(compressed, bytes)
    
    @pytest.mark.skip(reason="Encoding algorithm not yet implemented")
    def test_encode_sparse_data(self):
        """Test encoding of already sparse data."""
        encoder = IMCSEncoder(compression_ratio=0.5)
        data = np.zeros(100)
        data[10] = 1.0
        data[50] = 2.0
        compressed = encoder.encode(data)
        assert isinstance(compressed, bytes)

