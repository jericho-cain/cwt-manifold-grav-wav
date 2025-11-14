"""
Tests for LISA latent manifold geometry.

Tests k-NN manifold construction and geometric scoring.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.geometry.latent_manifold import (
    LatentManifold,
    LatentManifoldConfig,
)


class TestLatentManifoldConfig:
    """Test manifold configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LatentManifoldConfig()
        
        assert config.k_neighbors == 32
        assert config.tangent_dim is None
        assert config.metric == 'euclidean'
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LatentManifoldConfig(
            k_neighbors=16,
            tangent_dim=8,
            metric='cosine'
        )
        
        assert config.k_neighbors == 16
        assert config.tangent_dim == 8
        assert config.metric == 'cosine'


class TestLatentManifold:
    """Test latent manifold construction and geometry."""
    
    @pytest.fixture
    def simple_latents(self):
        """Create simple latent data for testing."""
        np.random.seed(42)
        # 100 points in 32-dimensional space
        return np.random.randn(100, 32).astype(np.float32)
    
    @pytest.fixture
    def manifold(self, simple_latents):
        """Create manifold from simple latents."""
        config = LatentManifoldConfig(k_neighbors=10)
        return LatentManifold(simple_latents, config)
    
    def test_manifold_creation(self, simple_latents):
        """Test creating manifold."""
        config = LatentManifoldConfig(k_neighbors=10)
        manifold = LatentManifold(simple_latents, config)
        
        assert manifold.train_latents.shape == (100, 32)
        assert manifold.config.k_neighbors == 10
    
    def test_manifold_requires_2d_input(self):
        """Test manifold requires 2D input."""
        config = LatentManifoldConfig()
        
        # 1D input should fail
        with pytest.raises(AssertionError):
            LatentManifold(np.random.randn(100), config)
        
        # 3D input should fail
        with pytest.raises(AssertionError):
            LatentManifold(np.random.randn(10, 32, 2), config)
    
    def test_normal_deviation(self, manifold, simple_latents):
        """Test computing normal deviation (off-manifold distance)."""
        # Test point from training set (should be low)
        z_train = simple_latents[0]
        dev_train = manifold.normal_deviation(z_train)
        
        # Should be a scalar
        assert isinstance(dev_train, float)
        assert dev_train >= 0
        
        # Test outlier (should be higher)
        z_outlier = simple_latents[0] + 10.0  # Far from manifold
        dev_outlier = manifold.normal_deviation(z_outlier)
        
        # Outlier should have higher deviation
        assert dev_outlier > dev_train
    
    def test_density_score(self, manifold, simple_latents):
        """Test computing density score."""
        z = simple_latents[0]
        density = manifold.density_score(z)
        
        # Should be a scalar
        assert isinstance(density, float)
        assert density >= 0
    
    def test_batch_normal_deviation(self, manifold, simple_latents):
        """Test batch computation of normal deviations."""
        Z = simple_latents[:10]
        
        deviations = manifold.batch_normal_deviation(Z)
        
        # Check shape and values
        assert deviations.shape == (10,)
        assert np.all(deviations >= 0)
    
    def test_batch_density_score(self, manifold, simple_latents):
        """Test batch computation of density scores."""
        Z = simple_latents[:10]
        
        densities = manifold.batch_density_score(Z)
        
        # Check shape and values
        assert densities.shape == (10,)
        assert np.all(densities >= 0)
    
    def test_different_k_neighbors(self, simple_latents):
        """Test manifolds with different k values."""
        for k in [5, 10, 20, 32]:
            config = LatentManifoldConfig(k_neighbors=k)
            manifold = LatentManifold(simple_latents, config)
            
            z = simple_latents[0]
            dev = manifold.normal_deviation(z)
            
            # Should work for all k
            assert isinstance(dev, float)
            assert dev >= 0
    
    def test_tangent_dim_auto_estimate(self, manifold, simple_latents):
        """Test auto-estimation of tangent dimension."""
        z = simple_latents[0]
        
        # Get local geometry (should auto-estimate tangent_dim)
        mu, U, explained = manifold._local_geometry(z)
        
        # Tangent basis should be estimated (95% variance)
        assert U.shape[0] == 32  # Full latent dim
        assert U.shape[1] <= 32  # Tangent dim <= latent dim
        assert U.shape[1] > 0     # At least 1 dimension
    
    def test_tangent_dim_fixed(self, simple_latents):
        """Test fixed tangent dimension."""
        config = LatentManifoldConfig(k_neighbors=10, tangent_dim=8)
        manifold = LatentManifold(simple_latents, config)
        
        z = simple_latents[0]
        mu, U, explained = manifold._local_geometry(z)
        
        # Should use fixed tangent_dim
        assert U.shape == (32, 8)


class TestManifoldPersistence:
    """Test saving and loading manifolds."""
    
    @pytest.fixture
    def simple_latents(self):
        """Create simple latent data."""
        np.random.seed(42)
        return np.random.randn(100, 32).astype(np.float32)
    
    def test_save_load_manifold(self, simple_latents):
        """Test saving and loading manifold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save manifold
            config = LatentManifoldConfig(k_neighbors=16, tangent_dim=8)
            manifold = LatentManifold(simple_latents, config)
            
            save_path = Path(tmpdir) / 'test_manifold.npz'
            manifold.save(str(save_path))
            
            # Check file exists
            assert save_path.exists()
            
            # Load manifold
            loaded_manifold = LatentManifold.load(str(save_path))
            
            # Check loaded correctly
            assert loaded_manifold.config.k_neighbors == 16
            assert loaded_manifold.config.tangent_dim == 8
            assert loaded_manifold.train_latents.shape == (100, 32)
    
    def test_save_load_preserves_geometry(self, simple_latents):
        """Test that saving and loading preserves geometric computations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifold
            config = LatentManifoldConfig(k_neighbors=10)
            manifold = LatentManifold(simple_latents, config)
            
            # Compute deviations
            z = simple_latents[0]
            original_dev = manifold.normal_deviation(z)
            
            # Save and load
            save_path = Path(tmpdir) / 'test_manifold.npz'
            manifold.save(str(save_path))
            loaded_manifold = LatentManifold.load(str(save_path))
            
            # Compute deviation on loaded manifold
            loaded_dev = loaded_manifold.normal_deviation(z)
            
            # Should be identical
            assert np.isclose(original_dev, loaded_dev, rtol=1e-5)
    
    def test_save_load_auto_tangent_dim(self, simple_latents):
        """Test saving/loading with auto-estimated tangent_dim."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifold with auto tangent_dim
            config = LatentManifoldConfig(k_neighbors=10, tangent_dim=None)
            manifold = LatentManifold(simple_latents, config)
            
            # Save and load
            save_path = Path(tmpdir) / 'test_manifold.npz'
            manifold.save(str(save_path))
            loaded_manifold = LatentManifold.load(str(save_path))
            
            # tangent_dim should still be None (auto-estimate)
            assert loaded_manifold.config.tangent_dim is None


class TestManifoldWithLISADimensions:
    """Test manifold with LISA-specific latent dimensions."""
    
    def test_manifold_with_lisa_latent_dim(self):
        """Test manifold with 32-dimensional latents (LISA default)."""
        np.random.seed(42)
        # 1000 training points, 32-dimensional (LISA latent dim)
        train_latents = np.random.randn(1000, 32).astype(np.float32)
        
        config = LatentManifoldConfig(k_neighbors=32)
        manifold = LatentManifold(train_latents, config)
        
        # Test on new point
        test_latent = np.random.randn(32).astype(np.float32)
        
        dev = manifold.normal_deviation(test_latent)
        density = manifold.density_score(test_latent)
        
        assert isinstance(dev, float)
        assert isinstance(density, float)
        assert dev >= 0
        assert density >= 0
    
    def test_batch_processing_realistic_size(self):
        """Test batch processing with realistic LISA dataset size."""
        np.random.seed(42)
        # Realistic sizes
        train_latents = np.random.randn(1000, 32).astype(np.float32)  # 1000 training segments
        test_latents = np.random.randn(200, 32).astype(np.float32)    # 200 test segments
        
        config = LatentManifoldConfig(k_neighbors=32)
        manifold = LatentManifold(train_latents, config)
        
        # Batch process all test data
        deviations = manifold.batch_normal_deviation(test_latents)
        densities = manifold.batch_density_score(test_latents)
        
        assert deviations.shape == (200,)
        assert densities.shape == (200,)
        assert np.all(deviations >= 0)
        assert np.all(densities >= 0)


class TestManifoldEdgeCases:
    """Test edge cases and error handling."""
    
    def test_manifold_with_few_points(self):
        """Test manifold with very few training points."""
        np.random.seed(42)
        # Only 20 points, k=10
        train_latents = np.random.randn(20, 32).astype(np.float32)
        
        config = LatentManifoldConfig(k_neighbors=10)
        manifold = LatentManifold(train_latents, config)
        
        # Should still work
        z = train_latents[0]
        dev = manifold.normal_deviation(z)
        
        assert isinstance(dev, float)
        assert dev >= 0
    
    def test_manifold_with_high_k(self):
        """Test manifold with k close to dataset size."""
        np.random.seed(42)
        train_latents = np.random.randn(50, 32).astype(np.float32)
        
        # k=30 (high relative to N=50)
        config = LatentManifoldConfig(k_neighbors=30)
        manifold = LatentManifold(train_latents, config)
        
        z = train_latents[0]
        dev = manifold.normal_deviation(z)
        
        assert isinstance(dev, float)
    
    def test_outlier_detection(self):
        """Test manifold can distinguish outliers."""
        np.random.seed(42)
        # Clustered data
        train_latents = np.random.randn(100, 32).astype(np.float32) * 0.1
        
        config = LatentManifoldConfig(k_neighbors=10)
        manifold = LatentManifold(train_latents, config)
        
        # Inlier (close to training distribution)
        inlier = np.random.randn(32).astype(np.float32) * 0.1
        dev_inlier = manifold.normal_deviation(inlier)
        
        # Outlier (far from training distribution)
        outlier = np.random.randn(32).astype(np.float32) * 10.0
        dev_outlier = manifold.normal_deviation(outlier)
        
        # Outlier should have much higher deviation
        assert dev_outlier > dev_inlier * 2

