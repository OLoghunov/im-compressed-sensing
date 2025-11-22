"""
IMCS - Image Compression using Compressed Sensing

A file format and codec implementation based on compressed sensing algorithms.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .encoder import IMCSEncoder
from .decoder import IMCSDecoder

__all__ = ["IMCSEncoder", "IMCSDecoder"]
