"""
Neural network models for LISA gravitational wave detection.

This module provides autoencoder architectures for learning latent
representations of LISA time-frequency data (CWT scalograms).
"""

from src.models.cwtlstm import (
    CWT_LSTM_Autoencoder,
    SimpleCWTAutoencoder,
    create_model,
    save_model,
    load_model,
)

__all__ = [
    "CWT_LSTM_Autoencoder",
    "SimpleCWTAutoencoder",
    "create_model",
    "save_model",
    "load_model",
]
