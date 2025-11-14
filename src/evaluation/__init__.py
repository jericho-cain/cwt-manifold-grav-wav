"""
Evaluation and scoring for LISA gravitational wave detection.

This module provides tools for combining autoencoder reconstruction errors
with manifold geometry scores to detect resolvable sources in LISA data.
"""

from src.evaluation.manifold_scorer import (
    ManifoldScorer,
    ManifoldScorerConfig,
    ScoringMode,
)

__all__ = [
    "ManifoldScorer",
    "ManifoldScorerConfig",
    "ScoringMode",
]
