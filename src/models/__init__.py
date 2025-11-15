"""
Neural network models for LISA gravitational wave detection.

This module provides autoencoder architectures for learning latent
representations of LISA time-frequency data (CWT scalograms).
"""

from src.models.cwtlstm import (
    CWTAutoencoder,
    SimpleCWTAutoencoder,
    create_model,
    save_model,
    load_model,
)

__all__ = [
    "CWTAutoencoder",
    "SimpleCWTAutoencoder",
    "create_model",
    "save_model",
    "load_model",
]
