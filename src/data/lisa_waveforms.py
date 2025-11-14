"""
LISA waveform generators for diverse source types.

Implements simplified analytical waveforms for:
- Massive Black Hole Binaries (MBHBs)
- Extreme Mass Ratio Inspirals (EMRIs)
- Galactic Binaries (GBs)

Note: These are analytical approximations, not full numerical relativity waveforms.
They capture the essential phenomenology for testing manifold-based detection.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class SourceParameters:
    """Base class for source parameters."""
    pass


@dataclass
class MBHBParameters(SourceParameters):
    """Massive Black Hole Binary parameters."""
    m1: float  # Primary mass in solar masses
    m2: float  # Secondary mass in solar masses
    distance: float  # Luminosity distance in Gpc
    f_start: float  # Starting frequency in Hz
    iota: float  # Inclination angle in radians
    phi_c: float  # Coalescence phase
    t_c: float  # Coalescence time in seconds


@dataclass
class EMRIParameters(SourceParameters):
    """Extreme Mass Ratio Inspiral parameters."""
    M: float  # Central massive black hole mass in solar masses
    mu: float  # Small body mass in solar masses
    distance: float  # Luminosity distance in Gpc
    p: float  # Semi-latus rectum (orbital parameter)
    e: float  # Eccentricity
    iota: float  # Inclination angle
    phi_0: float  # Initial phase


@dataclass
class GalacticBinaryParameters(SourceParameters):
    """Galactic Binary parameters."""
    f_gw: float  # GW frequency in Hz
    amplitude: float  # Strain amplitude
    f_dot: float  # Frequency derivative in Hz/s
    phi_0: float  # Initial phase
    iota: float  # Inclination


class LISAWaveformGenerator:
    """
    Generator for LISA waveforms of various source types.
    
    Parameters
    ----------
    duration : float
        Signal duration in seconds
    sampling_rate : float
        Sampling rate in Hz
    """
    
    def __init__(self, duration: float, sampling_rate: float):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.n_samples = int(duration * sampling_rate)
        self.times = np.arange(self.n_samples) / sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # Physical constants
        self.G = 6.674e-11  # m^3 kg^-1 s^-2
        self.c = 2.998e8  # m/s
        self.M_sun = 1.989e30  # kg
        self.pc = 3.086e16  # m
    
    def generate_mbhb(self, params: MBHBParameters) -> np.ndarray:
        """
        Generate Massive Black Hole Binary waveform using post-Newtonian approximation.
        
        Parameters
        ----------
        params : MBHBParameters
            Source parameters
            
        Returns
        -------
        h_t : np.ndarray
            Time-domain strain
        """
        # Convert to SI units
        m1_kg = params.m1 * self.M_sun
        m2_kg = params.m2 * self.M_sun
        M_total = m1_kg + m2_kg
        M_chirp = (m1_kg * m2_kg)**(3/5) / M_total**(1/5)
        eta = m1_kg * m2_kg / M_total**2
        distance_m = params.distance * 1e9 * self.pc
        
        # Time to coalescence
        t_c_idx = int(params.t_c * self.sampling_rate)
        if t_c_idx < 0 or t_c_idx >= self.n_samples:
            t_c_idx = self.n_samples // 2
        
        t_from_merger = self.times - self.times[t_c_idx]
        
        # Instantaneous frequency evolution (PN approximation)
        # f(t) = f_start * (1 - t/tau)^(-3/8) for t < 0
        tau = 5 * self.c**5 / (256 * (np.pi * params.f_start)**(8/3) * (self.G * M_chirp)**(5/3))
        
        # Only valid before merger
        mask = t_from_merger < 0
        t_rel = np.abs(t_from_merger[mask])
        
        # Frequency evolution
        f_t = np.zeros_like(self.times)
        valid_mask = t_rel < tau
        f_t[mask] = params.f_start * np.where(
            valid_mask,
            (1 - t_rel / tau)**(-3/8),
            0
        )
        
        # Phase evolution (integrate frequency)
        phase = 2 * np.pi * np.cumsum(f_t) * self.dt + params.phi_c
        
        # Amplitude evolution (depends on frequency)
        # h_0 = (G M_chirp / c^2)^(5/3) * (pi f / c)^(2/3) / distance
        h_0 = (self.G * M_chirp / self.c**2)**(5/3) / distance_m
        amplitude = h_0 * (np.pi * f_t / self.c)**(2/3)
        
        # Apply inclination angle and polarization
        A_plus = amplitude * (1 + np.cos(params.iota)**2) / 2
        A_cross = amplitude * np.cos(params.iota)
        
        # Generate strain (h_+ and h_x combined)
        h_t = A_plus * np.cos(phase) + A_cross * np.sin(phase)
        
        # Apply smooth window to avoid edge effects
        window = np.ones_like(h_t)
        window[:100] = np.hanning(200)[:100]
        window[-100:] = np.hanning(200)[100:]
        h_t *= window
        
        return h_t
    
    def generate_emri(self, params: EMRIParameters) -> np.ndarray:
        """
        Generate simplified EMRI waveform using quasi-circular orbit approximation.
        
        Parameters
        ----------
        params : EMRIParameters
            Source parameters
            
        Returns
        -------
        h_t : np.ndarray
            Time-domain strain
        """
        # Convert to SI
        M_kg = params.M * self.M_sun
        mu_kg = params.mu * self.M_sun
        distance_m = params.distance * 1e9 * self.pc
        
        # Orbital frequency (Kepler's law for given semi-latus rectum)
        # f_orb = sqrt(GM / (p * r_g)^3) where r_g = GM/c^2
        r_g = self.G * M_kg / self.c**2
        r = params.p * r_g
        f_orb = np.sqrt(self.G * M_kg / r**3) / (2 * np.pi)
        
        # GW frequency is 2 * orbital frequency for dominant mode
        f_gw = 2 * f_orb
        
        # Include frequency evolution due to radiation reaction (slow inspiral)
        # df/dt ~ f^(11/3) for leading order
        f_dot = (96 / 5) * (np.pi * f_gw)**(11/3) * \
                (self.G * M_kg / self.c**3)**(5/3) * \
                (params.mu / params.M)
        
        # Frequency evolution (linear approximation for short duration)
        f_t = f_gw + f_dot * self.times
        
        # Phase evolution
        phase = 2 * np.pi * (f_gw * self.times + 0.5 * f_dot * self.times**2) + params.phi_0
        
        # Amplitude (Peters-Mathews formula)
        h_0 = 4 * (self.G * mu_kg / self.c**2) * \
              (2 * np.pi * self.G * M_kg * f_gw / self.c**3)**(2/3) / distance_m
        
        # Modulation from eccentricity (simplified)
        # EMRI waveforms have rich harmonic structure
        n_harmonics = 5
        h_t = np.zeros_like(self.times)
        
        for n in range(1, n_harmonics + 1):
            # Bessel function approximation for eccentricity harmonics
            J_n = np.sinc(n * params.e / np.pi)  # Simplified Bessel
            h_n = h_0 * J_n * np.cos(n * phase)
            h_t += h_n
        
        # Apply inclination
        h_t *= (1 + np.cos(params.iota)**2) / 2
        
        # Smooth edges
        window = np.hanning(self.n_samples)
        h_t *= window
        
        return h_t
    
    def generate_galactic_binary(self, params: GalacticBinaryParameters) -> np.ndarray:
        """
        Generate Galactic Binary waveform (nearly monochromatic).
        
        Parameters
        ----------
        params : GalacticBinaryParameters
            Source parameters
            
        Returns
        -------
        h_t : np.ndarray
            Time-domain strain
        """
        # Frequency evolution (slowly chirping)
        f_t = params.f_gw + params.f_dot * self.times
        
        # Phase evolution
        phase = 2 * np.pi * (params.f_gw * self.times + 
                             0.5 * params.f_dot * self.times**2) + params.phi_0
        
        # Amplitude with inclination
        A_plus = params.amplitude * (1 + np.cos(params.iota)**2) / 2
        A_cross = params.amplitude * np.cos(params.iota)
        
        # Generate signal
        h_t = A_plus * np.cos(phase) + A_cross * np.sin(phase)
        
        return h_t
    
    def random_mbhb_params(
        self,
        seed: Optional[int] = None,
        m_range: Tuple[float, float] = (1e4, 1e7),
        q_range: Tuple[float, float] = (0.1, 1.0),
        distance_range: Tuple[float, float] = (1.0, 20.0),
    ) -> MBHBParameters:
        """
        Generate random MBHB parameters.
        
        Parameters
        ----------
        seed : int, optional
            Random seed
        m_range : tuple
            Primary mass range in solar masses
        q_range : tuple
            Mass ratio range (m2/m1)
        distance_range : tuple
            Distance range in Gpc
            
        Returns
        -------
        params : MBHBParameters
        """
        if seed is not None:
            np.random.seed(seed)
        
        m1 = np.random.uniform(m_range[0], m_range[1])
        q = np.random.uniform(q_range[0], q_range[1])
        m2 = q * m1
        distance = np.random.uniform(distance_range[0], distance_range[1])
        f_start = np.random.uniform(1e-4, 1e-2)
        iota = np.arccos(np.random.uniform(-1, 1))
        phi_c = np.random.uniform(0, 2 * np.pi)
        t_c = self.duration * np.random.uniform(0.3, 0.7)
        
        return MBHBParameters(m1, m2, distance, f_start, iota, phi_c, t_c)
    
    def random_emri_params(
        self,
        seed: Optional[int] = None,
        M_range: Tuple[float, float] = (1e5, 1e7),
        mu_range: Tuple[float, float] = (1.0, 100.0),
        distance_range: Tuple[float, float] = (0.5, 10.0),
    ) -> EMRIParameters:
        """Generate random EMRI parameters."""
        if seed is not None:
            np.random.seed(seed)
        
        M = np.random.uniform(M_range[0], M_range[1])
        mu = np.random.uniform(mu_range[0], mu_range[1])
        distance = np.random.uniform(distance_range[0], distance_range[1])
        p = np.random.uniform(6, 20)  # In units of r_g
        e = np.random.uniform(0, 0.5)
        iota = np.arccos(np.random.uniform(-1, 1))
        phi_0 = np.random.uniform(0, 2 * np.pi)
        
        return EMRIParameters(M, mu, distance, p, e, iota, phi_0)
    
    def random_galactic_binary_params(
        self,
        seed: Optional[int] = None,
        f_range: Tuple[float, float] = (1e-4, 1e-2),
        amp_range: Tuple[float, float] = (1e-22, 1e-20),
    ) -> GalacticBinaryParameters:
        """Generate random Galactic Binary parameters."""
        if seed is not None:
            np.random.seed(seed)
        
        f_gw = np.random.uniform(f_range[0], f_range[1])
        amplitude = np.random.uniform(amp_range[0], amp_range[1])
        f_dot = np.random.uniform(-1e-14, 1e-15)  # Slow chirp
        phi_0 = np.random.uniform(0, 2 * np.pi)
        iota = np.arccos(np.random.uniform(-1, 1))
        
        return GalacticBinaryParameters(f_gw, amplitude, f_dot, phi_0, iota)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Generating example LISA waveforms...")
    
    duration = 3600.0  # 1 hour
    sampling_rate = 1.0  # 1 Hz
    
    generator = LISAWaveformGenerator(duration, sampling_rate)
    
    # Generate examples of each source type
    print("\n1. Massive Black Hole Binary")
    mbhb_params = generator.random_mbhb_params(seed=42)
    h_mbhb = generator.generate_mbhb(mbhb_params)
    print(f"   Max strain: {np.max(np.abs(h_mbhb)):.2e}")
    
    print("\n2. Extreme Mass Ratio Inspiral")
    emri_params = generator.random_emri_params(seed=43)
    h_emri = generator.generate_emri(emri_params)
    print(f"   Max strain: {np.max(np.abs(h_emri)):.2e}")
    
    print("\n3. Galactic Binary")
    gb_params = generator.random_galactic_binary_params(seed=44)
    h_gb = generator.generate_galactic_binary(gb_params)
    print(f"   Max strain: {np.max(np.abs(h_gb)):.2e}")
    
    # Plot examples
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Show only first 500 seconds for visibility
    t_plot = generator.times[:500]
    
    axes[0].plot(t_plot, h_mbhb[:500])
    axes[0].set_ylabel('Strain')
    axes[0].set_title('Massive Black Hole Binary')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_plot, h_emri[:500])
    axes[1].set_ylabel('Strain')
    axes[1].set_title('Extreme Mass Ratio Inspiral')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t_plot, h_gb[:500])
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Strain')
    axes[2].set_title('Galactic Binary')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/lisa_waveform_examples.png', dpi=150)
    print("\nSaved waveform examples to results/lisa_waveform_examples.png")

