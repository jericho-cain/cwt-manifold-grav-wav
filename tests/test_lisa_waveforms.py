"""
Tests for LISA waveform generators.
"""

import numpy as np
import pytest
from src.data.lisa_waveforms import (
    LISAWaveformGenerator,
    MBHBParameters,
    EMRIParameters,
    GalacticBinaryParameters,
)


class TestLISAWaveformGenerator:
    """Test suite for LISAWaveformGenerator class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        duration = 3600.0
        sampling_rate = 1.0
        
        gen = LISAWaveformGenerator(duration, sampling_rate)
        
        assert gen.duration == duration
        assert gen.sampling_rate == sampling_rate
        assert gen.n_samples == 3600
        assert len(gen.times) == 3600
    
    def test_time_array(self):
        """Test time array is correctly generated."""
        gen = LISAWaveformGenerator(100.0, 2.0)
        
        assert gen.times[0] == 0.0
        assert np.isclose(gen.times[-1], 100.0 - 0.5)
        assert np.allclose(np.diff(gen.times), 0.5)


class TestMBHBGeneration:
    """Tests for Massive Black Hole Binary waveform generation."""
    
    @pytest.fixture
    def generator(self):
        """Create a waveform generator."""
        return LISAWaveformGenerator(duration=3600.0, sampling_rate=1.0)
    
    def test_mbhb_params_creation(self):
        """Test MBHB parameter creation."""
        params = MBHBParameters(
            m1=1e6,
            m2=5e5,
            distance=5.0,
            f_start=1e-3,
            iota=np.pi/4,
            phi_c=0.0,
            t_c=1800.0,
        )
        
        assert params.m1 == 1e6
        assert params.m2 == 5e5
        assert params.distance == 5.0
    
    def test_generate_mbhb_basic(self, generator):
        """Test basic MBHB waveform generation."""
        params = MBHBParameters(
            m1=1e6,
            m2=5e5,
            distance=5.0,
            f_start=1e-3,
            iota=np.pi/4,
            phi_c=0.0,
            t_c=1800.0,
        )
        
        h = generator.generate_mbhb(params)
        
        # Check shape
        assert len(h) == generator.n_samples
        
        # Check it's real-valued
        assert h.dtype in [np.float32, np.float64]
        assert not np.isnan(h).any()
        assert not np.isinf(h).any()
    
    def test_mbhb_amplitude_scales_with_distance(self, generator):
        """Test MBHB amplitude decreases with distance."""
        params_near = MBHBParameters(
            m1=1e6, m2=5e5, distance=1.0, f_start=1e-3,
            iota=0.0, phi_c=0.0, t_c=1800.0,
        )
        params_far = MBHBParameters(
            m1=1e6, m2=5e5, distance=10.0, f_start=1e-3,
            iota=0.0, phi_c=0.0, t_c=1800.0,
        )
        
        h_near = generator.generate_mbhb(params_near)
        h_far = generator.generate_mbhb(params_far)
        
        # Farther source should be weaker
        assert np.max(np.abs(h_near)) > np.max(np.abs(h_far))
    
    def test_random_mbhb_params(self, generator):
        """Test random MBHB parameter generation."""
        params = generator.random_mbhb_params(seed=42)
        
        assert isinstance(params, MBHBParameters)
        assert params.m1 >= 1e4
        assert params.m1 <= 1e7
        assert params.m2 <= params.m1
        assert params.m2 > 0
        assert 0 <= params.iota <= np.pi
        assert 0 <= params.phi_c <= 2*np.pi
    
    def test_random_mbhb_reproducibility(self, generator):
        """Test random MBHB params are reproducible with seed."""
        params1 = generator.random_mbhb_params(seed=42)
        params2 = generator.random_mbhb_params(seed=42)
        
        assert params1.m1 == params2.m1
        assert params1.m2 == params2.m2
        assert params1.distance == params2.distance


class TestEMRIGeneration:
    """Tests for EMRI waveform generation."""
    
    @pytest.fixture
    def generator(self):
        """Create a waveform generator."""
        return LISAWaveformGenerator(duration=3600.0, sampling_rate=1.0)
    
    def test_emri_params_creation(self):
        """Test EMRI parameter creation."""
        params = EMRIParameters(
            M=1e6,
            mu=10.0,
            distance=3.0,
            p=10.0,
            e=0.3,
            iota=np.pi/3,
            phi_0=0.0,
        )
        
        assert params.M == 1e6
        assert params.mu == 10.0
        assert params.e == 0.3
    
    def test_generate_emri_basic(self, generator):
        """Test basic EMRI waveform generation."""
        params = EMRIParameters(
            M=1e6,
            mu=10.0,
            distance=3.0,
            p=10.0,
            e=0.3,
            iota=np.pi/3,
            phi_0=0.0,
        )
        
        h = generator.generate_emri(params)
        
        # Check shape
        assert len(h) == generator.n_samples
        
        # Check it's real-valued
        assert h.dtype in [np.float32, np.float64]
        assert not np.isnan(h).any()
        assert not np.isinf(h).any()
    
    def test_emri_has_harmonics(self, generator):
        """Test EMRI waveform has multiple frequency components."""
        params = EMRIParameters(
            M=1e6,
            mu=10.0,
            distance=1.0,
            p=10.0,
            e=0.3,
            iota=0.0,
            phi_0=0.0,
        )
        
        h = generator.generate_emri(params)
        
        # Compute FFT
        h_fft = np.fft.rfft(h)
        power = np.abs(h_fft)**2
        
        # Should have power at multiple frequencies (harmonics)
        # Find peaks in power spectrum
        threshold = 0.1 * np.max(power)
        peaks = power > threshold
        n_peaks = np.sum(np.diff(peaks.astype(int)) == 1)
        
        # Should have multiple peaks (harmonics)
        assert n_peaks >= 2
    
    def test_random_emri_params(self, generator):
        """Test random EMRI parameter generation."""
        params = generator.random_emri_params(seed=42)
        
        assert isinstance(params, EMRIParameters)
        assert params.M >= 1e5
        assert params.M <= 1e7
        assert params.mu >= 1.0
        assert params.mu <= 100.0
        assert 0 <= params.e < 1  # Eccentricity must be < 1
        assert 0 <= params.iota <= np.pi


class TestGalacticBinaryGeneration:
    """Tests for Galactic Binary waveform generation."""
    
    @pytest.fixture
    def generator(self):
        """Create a waveform generator."""
        return LISAWaveformGenerator(duration=3600.0, sampling_rate=1.0)
    
    def test_gb_params_creation(self):
        """Test Galactic Binary parameter creation."""
        params = GalacticBinaryParameters(
            f_gw=1e-3,
            amplitude=1e-21,
            f_dot=1e-14,
            phi_0=0.0,
            iota=np.pi/4,
        )
        
        assert params.f_gw == 1e-3
        assert params.amplitude == 1e-21
        assert params.f_dot == 1e-14
    
    def test_generate_gb_basic(self, generator):
        """Test basic Galactic Binary waveform generation."""
        params = GalacticBinaryParameters(
            f_gw=1e-3,
            amplitude=1e-21,
            f_dot=1e-14,
            phi_0=0.0,
            iota=np.pi/4,
        )
        
        h = generator.generate_galactic_binary(params)
        
        # Check shape
        assert len(h) == generator.n_samples
        
        # Check it's real-valued
        assert h.dtype in [np.float32, np.float64]
        assert not np.isnan(h).any()
        assert not np.isinf(h).any()
    
    def test_gb_nearly_monochromatic(self, generator):
        """Test GB waveform is nearly monochromatic."""
        params = GalacticBinaryParameters(
            f_gw=1e-3,
            amplitude=1e-21,
            f_dot=0.0,  # No chirp
            phi_0=0.0,
            iota=0.0,
        )
        
        h = generator.generate_galactic_binary(params)
        
        # Compute FFT
        h_fft = np.fft.rfft(h)
        freqs = np.fft.rfftfreq(len(h), generator.dt)
        power = np.abs(h_fft)**2
        
        # Find peak frequency
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        
        # Should be close to f_gw (within FFT resolution)
        # FFT resolution is 1/duration = 1/3600 = 0.000278 Hz
        freq_resolution = 1.0 / generator.duration
        assert np.abs(peak_freq - params.f_gw) < 2 * freq_resolution
    
    def test_gb_amplitude_scaling(self, generator):
        """Test GB amplitude scales correctly."""
        params_weak = GalacticBinaryParameters(
            f_gw=1e-3,
            amplitude=1e-22,
            f_dot=0.0,
            phi_0=0.0,
            iota=0.0,
        )
        params_strong = GalacticBinaryParameters(
            f_gw=1e-3,
            amplitude=1e-20,
            f_dot=0.0,
            phi_0=0.0,
            iota=0.0,
        )
        
        h_weak = generator.generate_galactic_binary(params_weak)
        h_strong = generator.generate_galactic_binary(params_strong)
        
        # Stronger signal should have larger amplitude
        assert np.max(np.abs(h_strong)) > np.max(np.abs(h_weak))
    
    def test_random_gb_params(self, generator):
        """Test random Galactic Binary parameter generation."""
        params = generator.random_galactic_binary_params(seed=42)
        
        assert isinstance(params, GalacticBinaryParameters)
        assert params.f_gw >= 1e-4
        assert params.f_gw <= 1e-2
        assert params.amplitude > 0
        assert 0 <= params.iota <= np.pi
        assert 0 <= params.phi_0 <= 2*np.pi


class TestWaveformPhysicalProperties:
    """Test physical properties of generated waveforms."""
    
    @pytest.fixture
    def generator(self):
        """Create a waveform generator."""
        return LISAWaveformGenerator(duration=1000.0, sampling_rate=1.0)
    
    def test_all_waveforms_finite(self, generator):
        """Test all waveform types produce finite values."""
        mbhb_params = generator.random_mbhb_params(seed=42)
        emri_params = generator.random_emri_params(seed=43)
        gb_params = generator.random_galactic_binary_params(seed=44)
        
        h_mbhb = generator.generate_mbhb(mbhb_params)
        h_emri = generator.generate_emri(emri_params)
        h_gb = generator.generate_galactic_binary(gb_params)
        
        for h in [h_mbhb, h_emri, h_gb]:
            assert np.all(np.isfinite(h))
    
    def test_waveforms_have_nonzero_power(self, generator):
        """Test all waveforms have non-zero power."""
        mbhb_params = generator.random_mbhb_params(seed=42)
        emri_params = generator.random_emri_params(seed=43)
        gb_params = generator.random_galactic_binary_params(seed=44)
        
        h_mbhb = generator.generate_mbhb(mbhb_params)
        h_emri = generator.generate_emri(emri_params)
        h_gb = generator.generate_galactic_binary(gb_params)
        
        for h in [h_mbhb, h_emri, h_gb]:
            power = np.sum(h**2)
            assert power > 0
    
    def test_waveform_amplitudes_reasonable(self, generator):
        """Test waveform amplitudes are in reasonable range for LISA."""
        mbhb_params = generator.random_mbhb_params(seed=42)
        emri_params = generator.random_emri_params(seed=43)
        gb_params = generator.random_galactic_binary_params(seed=44)
        
        h_mbhb = generator.generate_mbhb(mbhb_params)
        h_emri = generator.generate_emri(emri_params)
        h_gb = generator.generate_galactic_binary(gb_params)
        
        # LISA strain should be roughly 1e-23 to 1e-18
        for h in [h_mbhb, h_emri, h_gb]:
            max_strain = np.max(np.abs(h))
            assert 1e-25 < max_strain < 1e-15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

