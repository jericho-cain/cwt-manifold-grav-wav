"""
Latent Manifold Geometry for LISA Gravitational Wave Detection

This module implements manifold learning in the autoencoder latent space.
Builds k-NN graphs and computes local tangent space geometry for anomaly detection.

For LISA:
- Training latents from confusion background (noise + unresolved GBs)
- Manifold captures "typical" LISA background structure
- Resolvable sources (MBHBs, EMRIs) appear off-manifold

Author: Jericho Cain
Date: November 2025
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


@dataclass
class LatentManifoldConfig:
    """
    Configuration for latent manifold construction.
    
    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors for local geometry, by default 32
    tangent_dim : Optional[int], optional
        Intrinsic dimensionality of tangent space.
        If None, auto-estimate from PCA (95% variance), by default None
    metric : str, optional
        Distance metric for k-NN ('euclidean', 'cosine', etc.), by default 'euclidean'
    """
    k_neighbors: int = 32
    tangent_dim: Optional[int] = None
    metric: str = "euclidean"


class LatentManifold:
    """
    Latent Manifold models the geometry of the autoencoder latent space.
    
    For LISA, this manifold represents the structure of typical LISA background
    (confusion noise from unresolved galactic binaries). Resolvable sources
    (MBHBs, EMRIs) should appear as off-manifold anomalies.
    
    Key operations:
    - Builds k-NN index on training latents (confusion background)
    - Estimates local tangent spaces via PCA
    - Computes normal deviation (off-manifold distance)
    - Computes density scores (k-NN distances)
    
    Parameters
    ----------
    train_latents : np.ndarray
        Training latent vectors from normal/background data, shape (N, d)
    config : LatentManifoldConfig
        Configuration for manifold construction
        
    Attributes
    ----------
    train_latents : np.ndarray
        Training latent vectors
    config : LatentManifoldConfig
        Configuration
    _nn : NearestNeighbors
        k-NN index for efficient neighbor queries
        
    Examples
    --------
    >>> # Build manifold from training latents
    >>> config = LatentManifoldConfig(k_neighbors=32)
    >>> manifold = LatentManifold(train_latents, config)
    >>> 
    >>> # Score test latent
    >>> off_manifold_score = manifold.normal_deviation(test_latent)
    >>> density_score = manifold.density_score(test_latent)
    """

    def __init__(self, train_latents: np.ndarray, config: LatentManifoldConfig):
        """
        Initialize latent manifold from training data.
        
        Parameters
        ----------
        train_latents : np.ndarray
            Training latent vectors from normal/background data, shape (N, d)
        config : LatentManifoldConfig
            Configuration for manifold construction
        """
        assert train_latents.ndim == 2, "Expected (N, d) latent array"
        self.train_latents = train_latents.astype(np.float32)
        self.config = config

        self._nn = NearestNeighbors(
            n_neighbors=config.k_neighbors,
            metric=config.metric,
        )
        self._nn.fit(self.train_latents)

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        Save manifold to disk.
        
        Parameters
        ----------
        path : str
            Path to save manifold (.npz format)
        """
        np.savez_compressed(
            path,
            train_latents=self.train_latents,
            k_neighbors=self.config.k_neighbors,
            tangent_dim=-1 if self.config.tangent_dim is None else self.config.tangent_dim,
            metric=self.config.metric,
        )

    @classmethod
    def load(cls, path: str) -> "LatentManifold":
        """
        Load manifold from disk.
        
        Parameters
        ----------
        path : str
            Path to load manifold from (.npz format)
            
        Returns
        -------
        LatentManifold
            Loaded manifold instance
        """
        data = np.load(path, allow_pickle=True)
        train_latents = data["train_latents"]
        k_neighbors = int(data["k_neighbors"])
        tangent_dim = int(data["tangent_dim"])
        if tangent_dim < 0:
            tangent_dim = None
        metric = str(data["metric"])
        cfg = LatentManifoldConfig(
            k_neighbors=k_neighbors,
            tangent_dim=tangent_dim,
            metric=metric,
        )
        return cls(train_latents=train_latents, config=cfg)

    # ---------- core geometry ----------

    def _local_geometry(
        self, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute local geometry at latent point z.
        
        For a single latent point z (d,), computes:
        - mu: local mean of neighbors (d,)
        - U: tangent basis matrix (d, tangent_dim)
        - explained: eigenvalues or explained variance
        
        Parameters
        ----------
        z : np.ndarray
            Latent point, shape (d,)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (mu, U, explained) - local mean, tangent basis, variance explained
        """
        z = np.asarray(z, dtype=np.float32).reshape(1, -1)
        distances, indices = self._nn.kneighbors(z, return_distance=True)
        neighbors = self.train_latents[indices[0]]  # (k, d)

        mu = neighbors.mean(axis=0)

        # PCA on centered neighbors
        centered = neighbors - mu
        pca = PCA()
        pca.fit(centered)

        if self.config.tangent_dim is None:
            # simple heuristic: keep components explaining 95% variance
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            tangent_dim = int(np.searchsorted(cumulative, 0.95) + 1)
        else:
            tangent_dim = self.config.tangent_dim

        U = pca.components_[:tangent_dim].T  # (d, tangent_dim)
        explained = pca.explained_variance_[:tangent_dim]

        return mu, U, explained

    def normal_deviation(self, z: np.ndarray) -> float:
        """
        Compute off-manifold distance (normal deviation).
        
        Measures ||normal component|| of z w.r.t. local tangent space.
        High values indicate off-manifold anomalies (e.g., resolvable sources in LISA).
        
        Parameters
        ----------
        z : np.ndarray
            Latent point, shape (d,)
            
        Returns
        -------
        float
            Off-manifold distance (normal deviation)
        """
        mu, U, _ = self._local_geometry(z)
        r = z - mu           # (d,)
        r_tan = U @ (U.T @ r)
        r_norm = r - r_tan
        return float(np.linalg.norm(r_norm))

    def density_score(self, z: np.ndarray) -> float:
        """
        Compute density score (mean k-NN distance).
        
        Simple k-NN-based density: mean distance to neighbors.
        Lower distances = higher density (more typical).
        Higher distances = lower density (more anomalous).
        
        Parameters
        ----------
        z : np.ndarray
            Latent point, shape (d,)
            
        Returns
        -------
        float
            Mean distance to k nearest neighbors
        """
        z = np.asarray(z, dtype=np.float32).reshape(1, -1)
        distances, _ = self._nn.kneighbors(z, return_distance=True)
        # skip the first neighbor (it might be itself if train/test overlap)
        return float(distances[0][1:].mean())

    # ---------- batch helpers ----------

    def batch_normal_deviation(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute off-manifold distances for batch of latents.
        
        Parameters
        ----------
        Z : np.ndarray
            Batch of latent points, shape (N, d)
            
        Returns
        -------
        np.ndarray
            Off-manifold distances, shape (N,)
        """
        return np.array([self.normal_deviation(z) for z in Z], dtype=np.float32)

    def batch_density_score(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute density scores for batch of latents.
        
        Parameters
        ----------
        Z : np.ndarray
            Batch of latent points, shape (N, d)
            
        Returns
        -------
        np.ndarray
            Density scores, shape (N,)
        """
        return np.array([self.density_score(z) for z in Z], dtype=np.float32)

