"""
Tests for IMCS utility functions.
"""

import pytest
import numpy as np
from imcs import utils


class TestMeasurementMatrix:
    """Test suite for measurement matrix generation."""
    
    @pytest.mark.skip(reason="Measurement matrix generation not yet implemented")
    def test_generate_measurement_matrix_shape(self):
        """Test that generated matrix has correct shape."""
        m, n = 50, 100
        matrix = utils.generate_measurement_matrix(m, n)
        assert matrix.shape == (m, n)
    
    @pytest.mark.skip(reason="Measurement matrix generation not yet implemented")
    def test_generate_gaussian_matrix(self):
        """Test generation of Gaussian measurement matrix."""
        matrix = utils.generate_measurement_matrix(50, 100, matrix_type='gaussian')
        assert matrix is not None


class TestSparsityComputation:
    """Test suite for sparsity computation."""
    
    @pytest.mark.skip(reason="Sparsity computation not yet implemented")
    def test_compute_sparsity_dense_signal(self):
        """Test sparsity computation for dense signal."""
        signal = np.random.rand(100)
        sparsity = utils.compute_sparsity(signal)
        assert sparsity > 0
    
    @pytest.mark.skip(reason="Sparsity computation not yet implemented")
    def test_compute_sparsity_sparse_signal(self):
        """Test sparsity computation for sparse signal."""
        signal = np.zeros(100)
        signal[10] = 1.0
        signal[50] = 2.0
        sparsity = utils.compute_sparsity(signal)
        assert sparsity == 2


class TestFormatValidation:
    """Test suite for IMCS format validation."""
    
    @pytest.mark.skip(reason="Format validation not yet implemented")
    def test_validate_valid_imcs_format(self):
        """Test validation of valid IMCS data."""
        # TODO: Need valid IMCS data to test
        pass
    
    @pytest.mark.skip(reason="Format validation not yet implemented")
    def test_validate_invalid_imcs_format(self):
        """Test validation of invalid IMCS data."""
        invalid_data = b"invalid data"
        result = utils.validate_imcs_format(invalid_data)
        assert result is False


class TestCompressionMetrics:
    """Test suite for compression metrics calculation."""
    
    @pytest.mark.skip(reason="Metrics calculation not yet implemented")
    def test_calculate_metrics_identical_signals(self):
        """Test metrics for identical signals."""
        signal = np.random.rand(100)
        metrics = utils.calculate_compression_metrics(signal, signal)
        # For identical signals, MSE should be 0 and PSNR should be inf
        assert metrics is not None
    
    @pytest.mark.skip(reason="Metrics calculation not yet implemented")
    def test_calculate_metrics_different_signals(self):
        """Test metrics for different signals."""
        original = np.random.rand(100)
        reconstructed = original + np.random.rand(100) * 0.1
        metrics = utils.calculate_compression_metrics(original, reconstructed)
        assert metrics is not None

