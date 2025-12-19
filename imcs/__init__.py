"""
IMCS - Image Compression using Compressed Sensing

A file format and codec implementation based on compressed sensing algorithms.

The key idea of Compressed Sensing:
- Most natural signals are sparse in some basis (e.g., images in DCT)
- We can capture the essential information with M << N random linear measurements
- Recovery is possible via L1-minimization or greedy algorithms (OMP)

Usage:
    >>> from imcs import IMCSEncoder, IMCSDecoder
    >>> import numpy as np
    >>>
    >>> # Create sparse signal
    >>> x = np.zeros(100)
    >>> x[[10, 30, 50]] = [1.0, 2.0, 0.5]
    >>>
    >>> # Encode
    >>> encoder = IMCSEncoder(compression_ratio=0.3)
    >>> compressed = encoder.encode(x)
    >>>
    >>> # Decode
    >>> decoder = IMCSDecoder()
    >>> reconstructed = decoder.decode(compressed)
"""

__version__ = "0.1.0"
__author__ = "Oleg Y. Logunov"

from .encoder import IMCSEncoder
from .decoder import IMCSDecoder
from .utils import (
    generate_measurement_matrix,
    compute_sparsity,
    calculate_compression_metrics,
    validate_imcs_format,
    dct2,
    idct2,
    omp,
    ista,
    simulated_annealing,
)

__all__ = [
    "IMCSEncoder",
    "IMCSDecoder",
    "generate_measurement_matrix",
    "compute_sparsity",
    "calculate_compression_metrics",
    "validate_imcs_format",
    "dct2",
    "idct2",
    "omp",
    "ista",
    "simulated_annealing",
]
