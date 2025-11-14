"""
Manifold geometry for LISA latent space analysis.

This module provides tools for building and analyzing manifolds in the
autoencoder latent space, including k-NN graphs, tangent space geometry,
and anomaly scoring based on off-manifold distances.
"""

from src.geometry.latent_manifold import (
    LatentManifold,
    LatentManifoldConfig,
)

__all__ = [
    "LatentManifold",
    "LatentManifoldConfig",
]
