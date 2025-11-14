"""
LISA noise model.

Implements the characteristic strain noise PSD for LISA based on
acceleration noise and optical metrology system (OMS) noise.

References:
- LISA Science Requirements Document (ESA/SRE(2018)1)
- Cornish & Robson (2017), arXiv:1703.09858
"""

import numpy as np
from typing import Optional


class LISANoise:
    """
    LISA noise power spectral density model.
    
    Parameters
    ----------
    f_min : float
        Minimum frequency in Hz
    f_max : float
        Maximum frequency in Hz
    n_frequencies : int
        Number of frequency bins
    L : float
        Arm length in meters (default: 2.5e9 m)
    f_star : float
        Transfer frequency in Hz (default: 19.09e-3 Hz)
    """
    
    def __init__(
        self,
        f_min: float = 1e-4,
        f_max: float = 1e-1,
        n_frequencies: int = 1024,
        L: float = 2.5e9,
        f_star: float = 19.09e-3,
    ):
        self.f_min = f_min
        self.f_max = f_max
        self.n_frequencies = n_frequencies
        self.L = L  # Arm length
        self.f_star = f_star  # Transfer frequency
        
        # Generate frequency array (log-spaced for better coverage)
        self.frequencies = np.logspace(
            np.log10(f_min), np.log10(f_max), n_frequencies
        )
        
        # Compute PSD
        self.psd = self._compute_psd()
    
    def _compute_psd(self) -> np.ndarray:
        """
        Compute the LISA noise PSD.
        
        Returns
        -------
        psd : np.ndarray
            One-sided power spectral density in Hz^-1
        """
        f = self.frequencies
        
        # Acceleration noise (low frequency)
        # S_acc(f) = (3e-15 m/s^2/Hz^(1/2))^2 * (1 + (0.4 mHz / f)^2) * (1 + (f / 8 mHz)^4)
        S_acc = (3e-15)**2 * (1 + (4e-4 / f)**2) * (1 + (f / 8e-3)**4)
        
        # Optical metrology system noise (high frequency)
        # S_oms(f) = (15e-12 m/Hz^(1/2))^2
        S_oms = (15e-12)**2 * np.ones_like(f)
        
        # Combine into strain noise PSD
        # S_n(f) = (10/3L^2) * [S_oms + (3 + cos(2f/f_star))S_acc / (2π f)^2]
        psd = (10 / (3 * self.L**2)) * (
            S_oms + (3 + np.cos(2 * f / self.f_star)) * S_acc / (2 * np.pi * f)**2
        )
        
        return psd
    
    def amplitude_spectral_density(self) -> np.ndarray:
        """
        Get amplitude spectral density (ASD = sqrt(PSD)).
        
        Returns
        -------
        asd : np.ndarray
            Amplitude spectral density in Hz^(-1/2)
        """
        return np.sqrt(self.psd)
    
    def generate_noise_td(
        self,
        duration: float,
        sampling_rate: float,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate time-domain LISA noise.
        
        Parameters
        ----------
        duration : float
            Duration in seconds
        sampling_rate : float
            Sampling rate in Hz
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        times : np.ndarray
            Time array in seconds
        noise_td : np.ndarray
            Time-domain noise strain
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = int(duration * sampling_rate)
        dt = 1.0 / sampling_rate
        times = np.arange(n_samples) * dt
        
        # Generate white noise in frequency domain
        n_freq = n_samples // 2 + 1
        df = 1.0 / duration
        freqs = np.fft.rfftfreq(n_samples, dt)
        
        # Interpolate PSD to match frequency bins
        psd_interp = np.interp(freqs, self.frequencies, self.psd, left=0, right=0)
        
        # Generate complex white noise
        real_part = np.random.randn(n_freq)
        imag_part = np.random.randn(n_freq)
        white_noise = real_part + 1j * imag_part
        
        # Color the noise with PSD
        # Factor of sqrt(2) because we're using rfft (one-sided)
        # Factor of sqrt(df) for correct normalization
        colored_noise_fd = white_noise * np.sqrt(psd_interp / (2 * df))
        
        # Ensure DC and Nyquist are real
        colored_noise_fd[0] = colored_noise_fd[0].real
        if n_samples % 2 == 0:
            colored_noise_fd[-1] = colored_noise_fd[-1].real
        
        # Transform to time domain
        noise_td = np.fft.irfft(colored_noise_fd, n=n_samples)
        
        return times, noise_td
    
    def snr(
        self,
        signal_fd: np.ndarray,
        signal_freqs: np.ndarray,
    ) -> float:
        """
        Compute matched-filter SNR for a signal.
        
        Parameters
        ----------
        signal_fd : np.ndarray
            Fourier transform of signal
        signal_freqs : np.ndarray
            Frequency array for signal
            
        Returns
        -------
        snr : float
            Signal-to-noise ratio
        """
        # Interpolate PSD to signal frequencies
        psd_interp = np.interp(signal_freqs, self.frequencies, self.psd)
        
        # Compute SNR^2 = 4 * integral(|h(f)|^2 / S_n(f) df)
        df = signal_freqs[1] - signal_freqs[0]
        snr_squared = 4 * np.sum(np.abs(signal_fd)**2 / psd_interp) * df
        
        return np.sqrt(snr_squared)


def plot_lisa_psd(noise_model: Optional[LISANoise] = None, save_path: Optional[str] = None):
    """
    Plot the LISA noise PSD and ASD.
    
    Parameters
    ----------
    noise_model : LISANoise, optional
        Noise model to plot. If None, creates default model.
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    if noise_model is None:
        noise_model = LISANoise()
    
    f = noise_model.frequencies
    psd = noise_model.psd
    asd = noise_model.amplitude_spectral_density()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot ASD
    ax1.loglog(f, np.sqrt(f) * asd, 'b-', linewidth=2)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel(r'Characteristic Strain $\sqrt{f \cdot S_n(f)}$ [Hz$^{-1/2}$]')
    ax1.set_title('LISA Noise Amplitude Spectral Density')
    ax1.grid(True, alpha=0.3)
    
    # Plot PSD
    ax2.loglog(f, psd, 'r-', linewidth=2)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'PSD $S_n(f)$ [Hz$^{-1}$]')
    ax2.set_title('LISA Noise Power Spectral Density')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Demo: Generate and plot LISA noise
    print("Generating LISA noise model...")
    lisa_noise = LISANoise()
    
    print(f"Frequency range: {lisa_noise.f_min:.2e} - {lisa_noise.f_max:.2e} Hz")
    print(f"Number of frequency bins: {lisa_noise.n_frequencies}")
    
    # Plot PSD
    plot_lisa_psd(lisa_noise, save_path="results/lisa_noise_psd.png")
    print("Saved PSD plot to results/lisa_noise_psd.png")
    
    # Generate time-domain noise
    print("\nGenerating time-domain noise...")
    duration = 3600.0  # 1 hour
    sampling_rate = 1.0  # 1 Hz
    times, noise = lisa_noise.generate_noise_td(duration, sampling_rate, seed=42)
    
    print(f"Generated {len(noise)} samples")
    print(f"Noise RMS: {np.std(noise):.2e}")
    print(f"Duration: {duration} seconds")

