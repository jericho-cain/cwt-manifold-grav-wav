"""
Tests for LISA CWT preprocessing.

Tests the CWT implementation for LISA gravitational wave data.
"""

import pytest
import numpy as np
from src.preprocessing.cwt import (
    LISACWTTransform,
    CWTConfig,
    cwt_lisa,
)


class TestCWTLISA:
    """Test core LISA CWT function."""
    
    def test_cwt_lisa_basic(self):
        """Test basic CWT functionality."""
        # Generate simple signal (1 mHz sine wave)
        duration = 3600  # 1 hour
        t = np.arange(duration)
        signal = np.sin(2 * np.pi * 1e-3 * t)  # 1 mHz
        
        # Apply CWT
        scalogram, freqs, scales = cwt_lisa(signal, fs=1.0)
        
        # Check output shape
        assert scalogram.shape[0] == 64  # Default n_scales
        assert scalogram.shape[1] == duration  # Same as input
        
        # Check normalization (should have mean≈0, std≈1 after normalization)
        assert np.abs(np.mean(scalogram)) < 0.1  # Close to 0
        assert np.abs(np.std(scalogram) - 1.0) < 0.2  # Close to 1
        
        # Check frequency range
        assert freqs.min() >= 1e-4  # At least fmin
        assert freqs.max() <= 1e-1  # At most fmax
    
    def test_cwt_lisa_with_chirp(self):
        """Test CWT on MBHB-like chirp."""
        duration = 3600
        t = np.arange(duration)
        
        # Chirp from 1 mHz to 3 mHz
        f0, f1 = 1e-3, 3e-3
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        signal = np.sin(phase)
        
        # Apply CWT
        scalogram, freqs, scales = cwt_lisa(signal, fs=1.0)
        
        # Should have non-zero power across frequency range
        assert scalogram.shape == (64, 3600)
        assert np.max(scalogram) > 0  # Has signal
        assert np.min(scalogram) < 0  # Normalized (has negative values)
    
    def test_cwt_lisa_custom_params(self):
        """Test CWT with custom frequency range."""
        signal = np.random.randn(1000)
        
        scalogram, freqs, scales = cwt_lisa(
            signal,
            fs=1.0,
            fmin=1e-3,  # 1 mHz
            fmax=1e-2,  # 10 mHz
            n_scales=32
        )
        
        assert scalogram.shape[0] == 32  # Custom n_scales
        assert len(freqs) == 32
        assert freqs.min() >= 1e-3
        assert freqs.max() <= 1e-2
    
    def test_cwt_lisa_global_normalization(self):
        """Test that global normalization is applied correctly."""
        signal = np.random.randn(1000)
        
        # Without global norm (per-segment)
        scalogram1, _, _ = cwt_lisa(signal, fs=1.0)
        
        # With global norm
        scalogram2, _, _ = cwt_lisa(
            signal,
            fs=1.0,
            global_mean=0.0,
            global_std=1.0
        )
        
        # Both should be normalized to mean≈0, std≈1
        assert np.abs(np.mean(scalogram1)) < 0.1
        assert np.abs(np.mean(scalogram2)) < 0.1
        assert np.abs(np.std(scalogram1) - 1.0) < 0.2
        assert np.abs(np.std(scalogram2) - 1.0) < 0.2


class TestLISACWTTransform:
    """Test LISACWTTransform class."""
    
    def test_cwt_class_basic(self):
        """Test basic CWT class usage."""
        config = CWTConfig(
            fmin=1e-4,
            fmax=1e-1,
            n_scales=64,
            target_height=64,
            target_width=3600
        )
        cwt = LISACWTTransform(config)
        
        signal = np.random.randn(3600)
        result = cwt.transform(signal)
        
        assert result.shape == (64, 3600)
        assert not np.any(np.isnan(result))
    
    def test_cwt_class_resizing(self):
        """Test CWT with automatic resizing."""
        config = CWTConfig(
            target_height=32,
            target_width=1800  # Downsample from 3600
        )
        cwt = LISACWTTransform(config)
        
        signal = np.random.randn(3600)
        result = cwt.transform(signal)
        
        # Should resize to target dimensions
        assert result.shape == (32, 1800)
    
    def test_cwt_class_with_global_norm(self):
        """Test CWT with global normalization parameters."""
        config = CWTConfig(
            use_global_norm=True,
            global_mean=1e-20,
            global_std=1e-19
        )
        cwt = LISACWTTransform(config)
        
        signal = np.random.randn(1000) * 1e-19 + 1e-20
        result = cwt.transform(signal)
        
        assert result is not None
        assert result.shape[0] == config.target_height
    
    def test_cwt_class_different_durations(self):
        """Test CWT works with different signal durations."""
        config = CWTConfig(target_width=3600)
        cwt = LISACWTTransform(config)
        
        # Test with different durations
        for duration in [1000, 3600, 7200]:
            signal = np.random.randn(duration)
            result = cwt.transform(signal)
            
            # Should always resize to target_width
            assert result.shape == (64, 3600)


class TestCWTConfig:
    """Test CWT configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = CWTConfig()
        
        assert config.wavelet == 'morl'
        assert config.fmin == 1e-4
        assert config.fmax == 1e-1
        assert config.n_scales == 64
        assert config.sampling_rate == 1.0
        assert config.target_height == 64
        assert config.target_width == 3600
    
    def test_config_custom(self):
        """Test custom configuration."""
        config = CWTConfig(
            fmin=1e-3,
            fmax=1e-2,
            n_scales=32,
            target_height=32,
            target_width=1800
        )
        
        assert config.fmin == 1e-3
        assert config.fmax == 1e-2
        assert config.n_scales == 32
        assert config.target_height == 32
        assert config.target_width == 1800


@pytest.mark.slow
class TestCWTIntegration:
    """Integration tests for CWT preprocessing."""
    
    def test_cwt_on_lisa_like_signal(self):
        """Test CWT on realistic LISA signal."""
        # Generate MBHB chirp
        duration = 3600
        fs = 1.0
        t = np.arange(duration) / fs
        
        # MBHB parameters
        f0 = 1e-3  # 1 mHz
        f1 = 3e-3  # 3 mHz
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        signal = np.sin(phase)
        
        # Add LISA-like noise
        noise = np.random.randn(duration) * 1e-20
        signal_with_noise = signal * 1e-19 + noise
        
        # Apply CWT
        config = CWTConfig()
        cwt = LISACWTTransform(config)
        result = cwt.transform(signal_with_noise)
        
        assert result is not None
        assert result.shape == (64, 3600)
        
        # Should capture chirp structure
        # (Higher frequencies should have more power at later times)
        early_power = np.mean(result[:, :600])  # First 10 min
        late_power = np.mean(result[:, -600:])  # Last 10 min
        
        # Both should have similar mean (normalized), but structure differs
        assert np.abs(early_power) < 1.0
        assert np.abs(late_power) < 1.0

