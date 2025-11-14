"""
Preprocessing for LISA gravitational wave data.

Includes Continuous Wavelet Transform (CWT) adapted from LIGO AE work
but made flexible for LISA characteristics.
"""

from .cwt import CWTConfig, LISACWTTransform, compute_global_normalization_stats, plot_cwt
