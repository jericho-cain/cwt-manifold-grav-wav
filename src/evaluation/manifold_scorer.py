"""
Manifold-Based Scoring for LISA Gravitational Wave Detection

This module combines autoencoder reconstruction error with manifold geometry
to detect resolvable sources in LISA confusion noise.

Scoring formula:
    combined_score = alpha * reconstruction_error + beta * off_manifold_distance

The beta coefficient directly measures whether manifold geometry helps!

For LISA:
- alpha weight for AE reconstruction (signal vs background)
- beta weight for manifold geometry (on-manifold vs off-manifold)
- Grid search over alpha, beta to find optimal combination

Adapted from LIGO manifold learning work.
Date: November 2024
"""

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from src.geometry.latent_manifold import LatentManifold


ScoringMode = Literal["ae_only", "manifold_only", "ae_plus_manifold"]


@dataclass
class ManifoldScorerConfig:
    """
    Configuration for manifold-based scoring.
    
    Parameters
    ----------
    mode : ScoringMode, optional
        Scoring mode ('ae_only', 'manifold_only', 'ae_plus_manifold'), 
        by default 'ae_plus_manifold'
    alpha_ae : float, optional
        Weight for AE reconstruction error, by default 1.0
    beta_manifold : float, optional
        Weight for off-manifold distance (normal deviation), by default 1.0
    use_density : bool, optional
        Whether to include density term, by default False
    gamma_density : float, optional
        Weight for density score (if used), by default 0.0
        
    Examples
    --------
    >>> # AE only (baseline)
    >>> config = ManifoldScorerConfig(mode='ae_only')
    >>> 
    >>> # AE + manifold
    >>> config = ManifoldScorerConfig(
    ...     mode='ae_plus_manifold',
    ...     alpha_ae=1.0,
    ...     beta_manifold=0.5
    ... )
    """
    mode: ScoringMode = "ae_plus_manifold"
    alpha_ae: float = 1.0          # weight for AE recon error
    beta_manifold: float = 1.0     # weight for normal deviation
    use_density: bool = False      # optional second manifold term
    gamma_density: float = 0.0     # weight for density (if used)


class ManifoldScorer:
    """
    Combines AE reconstruction error with manifold geometry for anomaly detection.
    
    For LISA, this combines:
    - Reconstruction error: How well does AE reconstruct the segment?
    - Off-manifold distance: How far is latent from confusion background manifold?
    
    The beta coefficient measures the importance of manifold geometry.
    
    Parameters
    ----------
    manifold : LatentManifold
        Built manifold from training latents (confusion background)
    config : ManifoldScorerConfig
        Scoring configuration (alpha, beta weights)
        
    Attributes
    ----------
    manifold : LatentManifold
        The manifold instance
    config : ManifoldScorerConfig
        Scoring configuration
        
    Examples
    --------
    >>> # Build manifold from training data
    >>> manifold = LatentManifold(train_latents, manifold_config)
    >>> 
    >>> # Create scorer with alpha=1.0, beta=0.5
    >>> config = ManifoldScorerConfig(alpha_ae=1.0, beta_manifold=0.5)
    >>> scorer = ManifoldScorer(manifold, config)
    >>> 
    >>> # Score test data
    >>> scores = scorer.score_batch(recon_errors, test_latents)
    >>> combined = scores['combined']  # Final anomaly scores
    """

    def __init__(self, manifold: LatentManifold, config: ManifoldScorerConfig):
        """
        Initialize manifold scorer.
        
        Parameters
        ----------
        manifold : LatentManifold
            Built manifold from training latents
        config : ManifoldScorerConfig
            Scoring configuration
        """
        self.manifold = manifold
        self.config = config

    def score_batch(
        self,
        recon_errors: np.ndarray,  # shape (N,)
        latents: np.ndarray,       # shape (N, d)
    ) -> Dict[str, np.ndarray]:
        """
        Score batch of test samples.
        
        Returns a dictionary of score arrays for analysis:
        - ae_error: Reconstruction errors from autoencoder
        - manifold_norm: Off-manifold distances (normal deviation)
        - density: Density scores (if enabled)
        - combined: Final anomaly scores (alpha * AE + beta * manifold)
        
        Parameters
        ----------
        recon_errors : np.ndarray
            Reconstruction errors from autoencoder, shape (N,)
        latents : np.ndarray
            Latent representations, shape (N, d)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'ae_error': Reconstruction errors
            - 'manifold_norm': Off-manifold distances
            - 'combined': Final anomaly scores
            - 'density': Density scores (if use_density=True)
            
        Examples
        --------
        >>> scores = scorer.score_batch(recon_errors, latents)
        >>> print(f"AE errors: {scores['ae_error'][:5]}")
        >>> print(f"Off-manifold: {scores['manifold_norm'][:5]}")
        >>> print(f"Combined: {scores['combined'][:5]}")
        """
        ae_error = np.asarray(recon_errors, dtype=np.float32).reshape(-1)
        Z = np.asarray(latents, dtype=np.float32)

        # Compute off-manifold distances
        manifold_norm = self.manifold.batch_normal_deviation(Z)

        # Optionally compute density
        if self.config.use_density:
            density = self.manifold.batch_density_score(Z)
        else:
            density = None

        # Build combined score depending on config.mode
        if self.config.mode == "ae_only":
            combined = ae_error

        elif self.config.mode == "manifold_only":
            combined = manifold_norm

        elif self.config.mode == "ae_plus_manifold":
            combined = (
                self.config.alpha_ae * ae_error
                + self.config.beta_manifold * manifold_norm
            )
            if self.config.use_density and density is not None:
                # Treat larger mean distance as more anomalous
                combined = combined + self.config.gamma_density * density
        else:
            raise ValueError(f"Unknown scoring mode: {self.config.mode}")

        out = {
            "ae_error": ae_error,
            "manifold_norm": manifold_norm,
            "combined": combined,
        }
        if density is not None:
            out["density"] = density

        return out

