"""
Tests for LISA noise model.
"""

import numpy as np
import pytest
from src.data.lisa_noise import LISANoise


class TestLISANoise:
    """Test suite for LISANoise class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        noise = LISANoise()
        
        assert noise.f_min == 1e-4
        assert noise.f_max == 1e-1
        assert noise.n_frequencies == 1024
        assert len(noise.frequencies) == 1024
        assert len(noise.psd) == 1024
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        noise = LISANoise(
            f_min=1e-5,
            f_max=1e-2,
            n_frequencies=512,
        )
        
        assert noise.f_min == 1e-5
        assert noise.f_max == 1e-2
        assert noise.n_frequencies == 512
        assert len(noise.frequencies) == 512
    
    def test_frequency_range(self):
        """Test frequency array is correctly spaced."""
        noise = LISANoise()
        
        # Check bounds
        assert np.isclose(noise.frequencies[0], noise.f_min)
        assert np.isclose(noise.frequencies[-1], noise.f_max)
        
        # Check log spacing
        log_freqs = np.log10(noise.frequencies)
        dlog = np.diff(log_freqs)
        assert np.allclose(dlog, dlog[0], rtol=1e-6)
    
    def test_psd_positive(self):
        """Test PSD is always positive."""
        noise = LISANoise()
        assert np.all(noise.psd > 0)
    
    def test_psd_shape(self):
        """Test PSD has correct shape."""
        noise = LISANoise()
        assert noise.psd.shape == (noise.n_frequencies,)
    
    def test_asd_is_sqrt_psd(self):
        """Test ASD is square root of PSD."""
        noise = LISANoise()
        asd = noise.amplitude_spectral_density()
        
        assert np.allclose(asd**2, noise.psd)
    
    def test_generate_noise_td_shape(self):
        """Test time-domain noise has correct shape."""
        noise = LISANoise()
        duration = 100.0
        sampling_rate = 1.0
        
        times, noise_td = noise.generate_noise_td(duration, sampling_rate)
        
        expected_samples = int(duration * sampling_rate)
        assert len(times) == expected_samples
        assert len(noise_td) == expected_samples
    
    def test_generate_noise_td_time_array(self):
        """Test time array is correctly generated."""
        noise = LISANoise()
        duration = 100.0
        sampling_rate = 2.0
        
        times, _ = noise.generate_noise_td(duration, sampling_rate)
        
        assert times[0] == 0.0
        assert np.isclose(times[-1], duration - 1.0/sampling_rate)
        assert np.allclose(np.diff(times), 1.0/sampling_rate)
    
    def test_noise_statistics(self):
        """Test noise has approximately correct statistics."""
        noise = LISANoise()
        duration = 10000.0  # Long duration for good statistics
        sampling_rate = 1.0
        
        _, noise_td = noise.generate_noise_td(duration, sampling_rate, seed=42)
        
        # Noise should be zero mean (approximately)
        assert np.abs(np.mean(noise_td)) < 1e-21
        
        # Should have non-zero variance
        assert np.std(noise_td) > 0
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same noise."""
        noise = LISANoise()
        duration = 100.0
        sampling_rate = 1.0
        
        _, noise1 = noise.generate_noise_td(duration, sampling_rate, seed=42)
        _, noise2 = noise.generate_noise_td(duration, sampling_rate, seed=42)
        
        assert np.allclose(noise1, noise2)
    
    def test_different_seeds_produce_different_noise(self):
        """Test that different seeds produce different noise."""
        noise = LISANoise()
        duration = 100.0
        sampling_rate = 1.0
        
        _, noise1 = noise.generate_noise_td(duration, sampling_rate, seed=42)
        _, noise2 = noise.generate_noise_td(duration, sampling_rate, seed=142)
        
        # Noise should be significantly different with different seeds
        assert not np.allclose(noise1, noise2, rtol=1e-3, atol=1e-25)
    
    def test_snr_computation(self):
        """Test SNR computation for a simple signal."""
        noise = LISANoise()
        
        # Create a simple sinusoidal signal
        duration = 1000.0
        sampling_rate = 1.0
        f_signal = 1e-3
        n_samples = int(duration * sampling_rate)
        
        times = np.arange(n_samples) / sampling_rate
        signal = 1e-20 * np.sin(2 * np.pi * f_signal * times)
        
        # Compute FFT
        signal_fd = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sampling_rate)
        
        # Compute SNR
        snr = noise.snr(signal_fd, freqs)
        
        # SNR should be positive and reasonable
        assert snr > 0
        assert snr < 1000  # Shouldn't be unreasonably large
    
    def test_noise_real_valued(self):
        """Test that generated noise is real-valued."""
        noise = LISANoise()
        duration = 100.0
        sampling_rate = 1.0
        
        _, noise_td = noise.generate_noise_td(duration, sampling_rate)
        
        # Should be real (no imaginary component)
        assert noise_td.dtype in [np.float32, np.float64]
        assert not np.isnan(noise_td).any()
        assert not np.isinf(noise_td).any()


class TestLISANoisePSDShape:
    """Test PSD has expected frequency-dependent behavior."""
    
    def test_psd_minimum_at_sweet_spot(self):
        """Test PSD has minimum around LISA's sweet spot (~3 mHz)."""
        noise = LISANoise(f_min=1e-4, f_max=1e-1, n_frequencies=4096)
        
        # Find frequency of minimum ASD
        asd = noise.amplitude_spectral_density()
        char_strain = np.sqrt(noise.frequencies * noise.psd)
        min_idx = np.argmin(char_strain)
        f_min = noise.frequencies[min_idx]
        
        # Should be roughly in the sweet spot range (0.1 - 10 mHz)
        # The actual minimum depends on the model details
        assert 1e-4 < f_min < 1e-2
    
    def test_low_frequency_behavior(self):
        """Test PSD increases at low frequencies (acceleration noise dominated)."""
        noise = LISANoise()
        
        # Get low-frequency portion
        low_freq_mask = noise.frequencies < 5e-4
        psd_low = noise.psd[low_freq_mask]
        
        # Should be increasing as frequency decreases
        # (PSD should be higher at lower frequencies)
        assert psd_low[0] > psd_low[-1]
    
    def test_high_frequency_behavior(self):
        """Test PSD increases at high frequencies (shot noise dominated)."""
        noise = LISANoise()
        
        # Get high-frequency portion
        high_freq_mask = noise.frequencies > 5e-2
        psd_high = noise.psd[high_freq_mask]
        
        # Should be increasing as frequency increases
        assert psd_high[-1] > psd_high[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

