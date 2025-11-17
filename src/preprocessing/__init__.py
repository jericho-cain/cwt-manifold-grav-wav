"""
Preprocessing for LISA gravitational wave data.

Includes Continuous Wavelet Transform (CWT) implementation optimized
for LISA frequency range and signal characteristics.
"""

from .cwt import CWTConfig, LISACWTTransform, compute_global_normalization_stats, plot_cwt
