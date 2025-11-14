"""
Tests for LISA manifold-based evaluation.

Tests the ManifoldScorer that combines AE reconstruction error
with manifold geometry (the β coefficient).
"""

import pytest
import numpy as np

from src.geometry.latent_manifold import LatentManifold, LatentManifoldConfig
from src.evaluation.manifold_scorer import (
    ManifoldScorer,
    ManifoldScorerConfig,
)


class TestManifoldScorerConfig:
    """Test manifold scorer configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ManifoldScorerConfig()
        
        assert config.mode == 'ae_plus_manifold'
        assert config.alpha_ae == 1.0
        assert config.beta_manifold == 1.0
        assert config.use_density is False
        assert config.gamma_density == 0.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ManifoldScorerConfig(
            mode='ae_only',
            alpha_ae=2.0,
            beta_manifold=0.5,
            use_density=True,
            gamma_density=0.1
        )
        
        assert config.mode == 'ae_only'
        assert config.alpha_ae == 2.0
        assert config.beta_manifold == 0.5
        assert config.use_density is True
        assert config.gamma_density == 0.1


class TestManifoldScorer:
    """Test manifold scorer functionality."""
    
    @pytest.fixture
    def manifold(self):
        """Create a simple manifold for testing."""
        np.random.seed(42)
        train_latents = np.random.randn(100, 32).astype(np.float32)
        config = LatentManifoldConfig(k_neighbors=10)
        return LatentManifold(train_latents, config)
    
    @pytest.fixture
    def test_data(self):
        """Create test reconstruction errors and latents."""
        np.random.seed(42)
        n_test = 20
        
        # Reconstruction errors (random for testing)
        recon_errors = np.random.rand(n_test).astype(np.float32)
        
        # Test latents
        latents = np.random.randn(n_test, 32).astype(np.float32)
        
        return recon_errors, latents
    
    def test_scorer_creation(self, manifold):
        """Test creating manifold scorer."""
        config = ManifoldScorerConfig()
        scorer = ManifoldScorer(manifold, config)
        
        assert scorer.manifold is manifold
        assert scorer.config.mode == 'ae_plus_manifold'
    
    def test_ae_only_mode(self, manifold, test_data):
        """Test AE-only mode (baseline, no manifold)."""
        config = ManifoldScorerConfig(mode='ae_only')
        scorer = ManifoldScorer(manifold, config)
        
        recon_errors, latents = test_data
        scores = scorer.score_batch(recon_errors, latents)
        
        # Combined should equal AE error only
        assert np.allclose(scores['combined'], scores['ae_error'])
    
    def test_manifold_only_mode(self, manifold, test_data):
        """Test manifold-only mode (no AE)."""
        config = ManifoldScorerConfig(mode='manifold_only')
        scorer = ManifoldScorer(manifold, config)
        
        recon_errors, latents = test_data
        scores = scorer.score_batch(recon_errors, latents)
        
        # Combined should equal manifold norm only
        assert np.allclose(scores['combined'], scores['manifold_norm'])
    
    def test_ae_plus_manifold_mode(self, manifold, test_data):
        """Test AE + manifold mode (the key test!)."""
        config = ManifoldScorerConfig(
            mode='ae_plus_manifold',
            alpha_ae=1.0,
            beta_manifold=0.5
        )
        scorer = ManifoldScorer(manifold, config)
        
        recon_errors, latents = test_data
        scores = scorer.score_batch(recon_errors, latents)
        
        # Combined should be weighted sum
        expected = 1.0 * scores['ae_error'] + 0.5 * scores['manifold_norm']
        assert np.allclose(scores['combined'], expected)
    
    def test_score_batch_returns_all_components(self, manifold, test_data):
        """Test score_batch returns all score components."""
        config = ManifoldScorerConfig()
        scorer = ManifoldScorer(manifold, config)
        
        recon_errors, latents = test_data
        scores = scorer.score_batch(recon_errors, latents)
        
        # Should have all components
        assert 'ae_error' in scores
        assert 'manifold_norm' in scores
        assert 'combined' in scores
        
        # Shapes should match
        n_test = len(recon_errors)
        assert scores['ae_error'].shape == (n_test,)
        assert scores['manifold_norm'].shape == (n_test,)
        assert scores['combined'].shape == (n_test,)
    
    def test_different_alpha_beta_weights(self, manifold, test_data):
        """Test different α, β weights produce different scores."""
        recon_errors, latents = test_data
        
        # Configuration 1: α=1, β=0
        config1 = ManifoldScorerConfig(alpha_ae=1.0, beta_manifold=0.0)
        scorer1 = ManifoldScorer(manifold, config1)
        scores1 = scorer1.score_batch(recon_errors, latents)
        
        # Configuration 2: α=1, β=1
        config2 = ManifoldScorerConfig(alpha_ae=1.0, beta_manifold=1.0)
        scorer2 = ManifoldScorer(manifold, config2)
        scores2 = scorer2.score_batch(recon_errors, latents)
        
        # Scores should be different (unless manifold_norm is all zeros)
        assert not np.allclose(scores1['combined'], scores2['combined'])
    
    def test_density_scoring(self, manifold, test_data):
        """Test optional density scoring."""
        config = ManifoldScorerConfig(
            mode='ae_plus_manifold',
            alpha_ae=1.0,
            beta_manifold=0.5,
            use_density=True,
            gamma_density=0.1
        )
        scorer = ManifoldScorer(manifold, config)
        
        recon_errors, latents = test_data
        scores = scorer.score_batch(recon_errors, latents)
        
        # Should include density scores
        assert 'density' in scores
        assert scores['density'].shape == (len(recon_errors),)
        
        # Combined should include density term
        expected = (
            1.0 * scores['ae_error'] 
            + 0.5 * scores['manifold_norm']
            + 0.1 * scores['density']
        )
        assert np.allclose(scores['combined'], expected)
    
    def test_invalid_mode_raises_error(self, manifold, test_data):
        """Test invalid scoring mode raises error."""
        # Manually create invalid config (bypass validation)
        config = ManifoldScorerConfig()
        config.mode = 'invalid_mode'
        
        scorer = ManifoldScorer(manifold, config)
        recon_errors, latents = test_data
        
        with pytest.raises(ValueError, match="Unknown scoring mode"):
            scorer.score_batch(recon_errors, latents)


class TestBetaCoefficientSimulation:
    """Test β coefficient behavior (what we're measuring!)."""
    
    @pytest.fixture
    def setup_experiment(self):
        """Setup experiment with normal and anomalous data."""
        np.random.seed(42)
        
        # Training latents (normal/background)
        train_latents = np.random.randn(100, 32).astype(np.float32) * 0.5
        
        # Test latents
        # - Half normal (similar to training)
        # - Half anomalous (far from training)
        normal_latents = np.random.randn(50, 32).astype(np.float32) * 0.5
        anomalous_latents = np.random.randn(50, 32).astype(np.float32) * 2.0 + 5.0
        
        test_latents = np.vstack([normal_latents, anomalous_latents])
        
        # Reconstruction errors (assume AE performs similarly on both)
        recon_errors = np.random.rand(100).astype(np.float32)
        
        # Labels (0=normal, 1=anomalous)
        labels = np.array([0]*50 + [1]*50)
        
        # Build manifold
        config = LatentManifoldConfig(k_neighbors=10)
        manifold = LatentManifold(train_latents, config)
        
        return manifold, recon_errors, test_latents, labels
    
    def test_beta_zero_equivalent_to_ae_only(self, setup_experiment):
        """Test β=0 is equivalent to AE-only (LIGO result)."""
        manifold, recon_errors, test_latents, labels = setup_experiment
        
        # β=0 (LIGO result: manifold doesn't help)
        config_beta0 = ManifoldScorerConfig(alpha_ae=1.0, beta_manifold=0.0)
        scorer_beta0 = ManifoldScorer(manifold, config_beta0)
        scores_beta0 = scorer_beta0.score_batch(recon_errors, test_latents)
        
        # AE only
        config_ae = ManifoldScorerConfig(mode='ae_only')
        scorer_ae = ManifoldScorer(manifold, config_ae)
        scores_ae = scorer_ae.score_batch(recon_errors, test_latents)
        
        # Should be identical
        assert np.allclose(scores_beta0['combined'], scores_ae['combined'])
    
    def test_beta_positive_uses_manifold_geometry(self, setup_experiment):
        """Test β>0 incorporates manifold geometry."""
        manifold, recon_errors, test_latents, labels = setup_experiment
        
        # Get scores with β>0
        config = ManifoldScorerConfig(alpha_ae=1.0, beta_manifold=1.0)
        scorer = ManifoldScorer(manifold, config)
        scores = scorer.score_batch(recon_errors, test_latents)
        
        # Manifold component should be non-zero
        assert np.any(scores['manifold_norm'] > 0)
        
        # Combined should differ from AE-only
        assert not np.allclose(scores['combined'], scores['ae_error'])
    
    def test_manifold_geometry_helps_distinguish_anomalies(self, setup_experiment):
        """Test that manifold geometry can distinguish anomalies."""
        manifold, recon_errors, test_latents, labels = setup_experiment
        
        config = ManifoldScorerConfig()
        scorer = ManifoldScorer(manifold, config)
        scores = scorer.score_batch(recon_errors, test_latents)
        
        # Separate normal vs anomalous
        normal_manifold_scores = scores['manifold_norm'][:50]
        anomalous_manifold_scores = scores['manifold_norm'][50:]
        
        # Anomalous should have higher off-manifold distance
        assert np.mean(anomalous_manifold_scores) > np.mean(normal_manifold_scores)


class TestManifoldScorerWithLISARealisticData:
    """Test manifold scorer with LISA-realistic scenarios."""
    
    def test_confusion_vs_resolvable_sources(self):
        """
        Simulate LISA scenario:
        - Training: Confusion background
        - Test: Confusion background + resolvable sources
        """
        np.random.seed(42)
        
        # Training latents: confusion background
        # (clustered distribution representing typical confusion)
        confusion_latents_train = np.random.randn(1000, 32).astype(np.float32) * 0.3
        
        # Test latents:
        # - Background (similar to training confusion)
        background_latents = np.random.randn(200, 32).astype(np.float32) * 0.3
        # - Resolvable sources (MBHBs, EMRIs - different from confusion)
        resolvable_latents = np.random.randn(200, 32).astype(np.float32) * 1.0 + 2.0
        
        test_latents = np.vstack([background_latents, resolvable_latents])
        
        # Reconstruction errors (assume resolvable sources have higher error)
        bg_errors = np.random.rand(200).astype(np.float32) * 0.5
        resolvable_errors = np.random.rand(200).astype(np.float32) * 0.5 + 0.5
        recon_errors = np.hstack([bg_errors, resolvable_errors])
        
        # Build manifold from confusion background
        config = LatentManifoldConfig(k_neighbors=32)
        manifold = LatentManifold(confusion_latents_train, config)
        
        # Score with manifold
        scorer_config = ManifoldScorerConfig(alpha_ae=1.0, beta_manifold=0.5)
        scorer = ManifoldScorer(manifold, scorer_config)
        scores = scorer.score_batch(recon_errors, test_latents)
        
        # Resolvable sources should have higher scores
        bg_scores = scores['combined'][:200]
        resolvable_scores = scores['combined'][200:]
        
        # Resolvable should score higher (more anomalous)
        assert np.mean(resolvable_scores) > np.mean(bg_scores)
        
        # Manifold component should contribute
        bg_manifold = scores['manifold_norm'][:200]
        resolvable_manifold = scores['manifold_norm'][200:]
        assert np.mean(resolvable_manifold) > np.mean(bg_manifold)
    
    def test_grid_search_simulation(self):
        """Simulate grid search over α, β (what we'll do for real!)."""
        np.random.seed(42)
        
        # Setup data
        train_latents = np.random.randn(100, 32).astype(np.float32)
        test_latents = np.random.randn(50, 32).astype(np.float32)
        recon_errors = np.random.rand(50).astype(np.float32)
        
        # Build manifold
        config = LatentManifoldConfig(k_neighbors=10)
        manifold = LatentManifold(train_latents, config)
        
        # Grid search
        alpha_range = [0.5, 1.0, 2.0]
        beta_range = [0, 0.1, 0.5, 1.0]
        
        results = []
        for alpha in alpha_range:
            for beta in beta_range:
                scorer_config = ManifoldScorerConfig(
                    alpha_ae=alpha,
                    beta_manifold=beta
                )
                scorer = ManifoldScorer(manifold, scorer_config)
                scores = scorer.score_batch(recon_errors, test_latents)
                
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'mean_score': np.mean(scores['combined']),
                    'std_score': np.std(scores['combined'])
                })
        
        # Should have results for all combinations
        assert len(results) == len(alpha_range) * len(beta_range)
        
        # Different α, β should produce different scores
        scores_list = [r['mean_score'] for r in results]
        assert len(set(scores_list)) > 1  # Not all the same

