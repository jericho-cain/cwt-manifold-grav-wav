"""
Continuous Wavelet Transform for LISA data.

Adapted from LIGO autoencoder legacy code but with LISA-specific parameters.

Key differences from LIGO CWT:
- Frequency range: mHz instead of Hz (0.1 mHz to 100 mHz vs 20-512 Hz)
- Time scales: longer signals (3600s vs 32s)
- Sampling rate: 1 Hz vs 4096 Hz
- Normalization: From confusion background vs pure noise

Critical design choices from legacy LIGO code that we preserve:
1. Global normalization statistics (prevents batch effects)
2. Log transform of magnitude scalogram
3. Per-segment normalization after log transform
4. High-pass filtering before CWT
"""

import numpy as np
import pywt
import logging
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import zoom
from typing import Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CWTConfig:
    """Configuration for Continuous Wavelet Transform.
    
    Parameters adapted for LISA data characteristics.
    """
    # Wavelet parameters
    wavelet: str = "morl"  # Morlet wavelet (good for chirps)
    
    # Frequency parameters (LISA range)
    fmin: float = 1e-4  # 0.1 mHz
    fmax: float = 1e-1  # 100 mHz
    n_scales: int = 64   # Number of frequency scales
    
    # Time parameters
    sampling_rate: float = 1.0  # Hz (LISA samples at 1 Hz)
    
    # Target dimensions for autoencoder
    target_height: int = 64   # Number of frequency scales
    target_width: int = 3600  # Time samples (1 hour at 1 Hz)
    
    # Normalization (following legacy approach)
    use_global_norm: bool = True  # Use global stats from training background
    global_mean: Optional[float] = None
    global_std: Optional[float] = None


def compute_global_normalization_stats(
    training_background_files: List[Path],
    sample_rate: float = 1.0,
    fmin: float = 1e-4
) -> Tuple[float, float]:
    """
    Compute global whitening statistics from training background files.
    
    Adapted from LIGO legacy code. Computes mean/std from training data
    to prevent batch effects across different segments.
    
    Parameters
    ----------
    training_background_files : List[Path]
        List of training background files (HDF5 with 'strain' dataset)
    sample_rate : float, optional
        Sampling rate in Hz, by default 1.0
    fmin : float, optional
        High-pass filter cutoff frequency, by default 1e-4
        
    Returns
    -------
    Tuple[float, float]
        (global_mean, global_std) - Statistics for whitening normalization
    """
    logger.info(f"Computing global normalization from {len(training_background_files)} training background files...")
    
    all_filtered = []
    
    for i, bg_file in enumerate(training_background_files):
        try:
            # Load LISA background data
            import h5py
            with h5py.File(bg_file, 'r') as f:
                strain = f['strain'][:]
            
            # Apply same high-pass filter as in cwt_lisa
            sos = butter(4, fmin, btype='high', fs=sample_rate, output='sos')
            filtered = sosfiltfilt(sos, strain)
            
            all_filtered.append(filtered)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Loaded {i + 1}/{len(training_background_files)} files...")
                
        except Exception as e:
            logger.warning(f"  Failed to load {bg_file.name}: {e}")
            continue
    
    # Concatenate all training background
    combined_background = np.concatenate(all_filtered)
    logger.info(f"Combined {len(all_filtered)} files -> {len(combined_background)} samples")
    
    # Compute global statistics
    global_mean = np.mean(combined_background)
    global_std = np.std(combined_background)
    
    logger.info(f"Global normalization stats computed:")
    logger.info(f"  Mean: {global_mean:.6e}")
    logger.info(f"  Std:  {global_std:.6e}")
    
    return global_mean, global_std


def cwt_lisa(
    x: np.ndarray,
    fs: float = 1.0,
    fmin: float = 1e-4,
    fmax: float = 1e-1,
    n_scales: int = 64,
    wavelet: str = 'morl',
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LISA CWT implementation adapted from LIGO legacy code.
    
    Key steps (matching legacy approach):
    1. High-pass filter at fmin
    2. Whitening (zero mean, unit variance) using global or local stats
    3. CWT with logarithmic frequency spacing
    4. Magnitude (absolute value)
    5. Log transform
    6. Per-segment normalization
    
    Parameters
    ----------
    x : np.ndarray
        Input time series data
    fs : float, optional
        Sampling frequency in Hz, by default 1.0
    fmin : float, optional
        Minimum frequency for CWT analysis, by default 1e-4
    fmax : float, optional
        Maximum frequency for CWT analysis, by default 1e-1
    n_scales : int, optional
        Number of scales for CWT, by default 64
    wavelet : str, optional
        Wavelet type, by default 'morl'
    global_mean : float, optional
        Global mean for whitening (if None, uses per-file mean)
    global_std : float, optional
        Global std for whitening (if None, uses per-file std)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - scalogram: Normalized log-magnitude CWT coefficients
        - frequencies: Frequency values in Hz
        - scales: Wavelet scales used
    """
    
    # Input validation
    if len(x) == 0:
        raise ValueError("Input signal is empty")
    
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    # High-pass filter (fmin cutoff) - same as legacy
    try:
        sos = butter(4, fmin, btype='high', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, x)
        logger.debug(f"High-pass filtering applied: {fmin} Hz cutoff")
    except Exception as e:
        logger.warning(f"High-pass filtering failed: {e}")
        filtered = x
    
    # Whitening (zero mean, unit variance)
    try:
        if global_mean is not None and global_std is not None:
            # Use GLOBAL statistics (recommended - prevents batch effects)
            whitened = (filtered - global_mean) / (global_std + 1e-10)
            logger.debug(f"Global whitening applied: mean={global_mean:.6e}, std={global_std:.6e}")
        else:
            # Fall back to per-file whitening (legacy - causes batch effects)
            whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
            logger.warning("Using per-file whitening (may cause batch effects)")
        logger.debug(f"Whitened output: mean={whitened.mean():.6e}, std={whitened.std():.6e}")
    except Exception as e:
        logger.warning(f"Whitening failed: {e}")
        whitened = filtered
    
    # Generate scales for CWT (logarithmic spacing)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    scales = fs / freqs
    logger.debug(f"CWT scales: {len(scales)} scales covering {fmin}-{fmax} Hz")
    
    # Compute CWT - same as legacy
    try:
        coefficients, frequencies = pywt.cwt(
            whitened, scales, wavelet, sampling_period=1/fs
        )
        logger.debug(f"CWT computed: coefficients shape={coefficients.shape}")
    except Exception as e:
        logger.error(f"CWT computation failed: {e}")
        raise
    
    # Magnitude scalogram
    scalogram = np.abs(coefficients).astype(np.float32)
    
    # Resize to target height if needed (same as legacy)
    if scalogram.shape[0] != n_scales:
        zoom_factor = n_scales / scalogram.shape[0]
        scalogram = zoom(scalogram, (zoom_factor, 1), order=1)
        logger.info(f"Resized to target height: {scalogram.shape}")
    
    # Log transform and normalize (LEGACY APPROACH)
    log_scalogram = np.log10(scalogram + 1e-10)
    normalized = (log_scalogram - np.mean(log_scalogram)) / (np.std(log_scalogram) + 1e-10)
    normalized = normalized.astype(np.float32)
    
    logger.info(f"LISA CWT completed: shape={normalized.shape}, range={normalized.min():.6e} to {normalized.max():.6e}")
    logger.info(f"Normalized data: mean={normalized.mean():.6e}, std={normalized.std():.6e}")
    
    return normalized, freqs, scales


class LISACWTTransform:
    """
    Continuous Wavelet Transform for LISA gravitational wave data.
    
    Adapted from LIGO legacy code with LISA-specific parameters.
    
    Parameters
    ----------
    config : CWTConfig
        CWT configuration
    """
    
    def __init__(self, config: CWTConfig):
        self.config = config
        
        logger.info(f"LISACWTTransform initialized")
        logger.info(f"  Frequency range: {config.fmin:.2e} - {config.fmax:.2e} Hz")
        logger.info(f"  Number of scales: {config.n_scales}")
        logger.info(f"  Target dimensions: {config.target_height} x {config.target_width}")
        logger.info(f"  Sampling rate: {config.sampling_rate} Hz")
        if config.use_global_norm and config.global_mean is not None:
            logger.info(f"  Global normalization: mean={config.global_mean:.6e}, std={config.global_std:.6e}")
        else:
            logger.warning(f"  Using per-segment whitening (may cause batch effects)")
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply CWT to time-domain LISA signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Time-domain signal
            
        Returns
        -------
        cwt_matrix : np.ndarray
            CWT coefficients (n_scales x n_times), normalized
        """
        # Apply LISA CWT
        scalogram, freqs, scales = cwt_lisa(
            signal,
            fs=self.config.sampling_rate,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            n_scales=self.config.n_scales,
            wavelet=self.config.wavelet,
            global_mean=self.config.global_mean,
            global_std=self.config.global_std,
        )
        
        # Resize to target dimensions if needed
        zoom_factors = [1.0, 1.0]
        
        # Height (frequency) resizing
        if scalogram.shape[0] != self.config.target_height:
            zoom_factors[0] = self.config.target_height / scalogram.shape[0]
        
        # Width (time) resizing
        if scalogram.shape[1] != self.config.target_width:
            zoom_factors[1] = self.config.target_width / scalogram.shape[1]
        
        # Apply zoom if needed
        if zoom_factors != [1.0, 1.0]:
            scalogram = zoom(scalogram, zoom_factors, order=1)
            logger.info(f"Resized to target dimensions: {scalogram.shape}")
        
        # Check for NaN values
        if np.any(np.isnan(scalogram)):
            logger.warning(f"NaN values detected in CWT output")
            return None
        
        return scalogram


def plot_cwt(
    signal: np.ndarray,
    cwt_matrix: np.ndarray,
    sampling_rate: float = 1.0,
    fmin: float = 1e-4,
    fmax: float = 1e-1,
    save_path: Optional[str] = None
):
    """
    Plot time series and CWT time-frequency representation.
    
    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal
    cwt_matrix : np.ndarray
        CWT coefficients
    sampling_rate : float
        Sampling rate in Hz
    fmin, fmax : float
        Frequency range for plot
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    times = np.arange(len(signal)) / sampling_rate
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), cwt_matrix.shape[0])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series
    ax1.plot(times, signal, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Strain')
    ax1.set_title('LISA Time Series')
    ax1.grid(True, alpha=0.3)
    
    # CWT spectrogram
    im = ax2.pcolormesh(
        times,
        freqs,
        cwt_matrix,
        shading='auto',
        cmap='viridis'
    )
    ax2.set_yscale('log')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_title('CWT Time-Frequency Representation (LISA)')
    ax2.set_ylim([fmin, fmax])
    plt.colorbar(im, ax=ax2, label='Normalized CWT coefficient')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo: Generate LISA signal and apply CWT
    print("=" * 80)
    print("LISA CWT Demo - Adapted from LIGO Legacy Code")
    print("=" * 80)
    
    # Generate synthetic MBHB chirp
    duration = 3600.0  # 1 hour
    sampling_rate = 1.0  # 1 Hz
    t = np.arange(int(duration * sampling_rate)) / sampling_rate
    
    # MBHB chirp (frequency increases slowly)
    f0 = 1e-3  # 1 mHz
    f1 = 3e-3  # 3 mHz
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    chirp = np.sin(phase)
    
    # Add realistic LISA noise level
    noise = np.random.randn(len(t)) * 1e-20
    signal = chirp * 1e-19 + noise
    
    print(f"\nGenerated LISA Signal:")
    print(f"  Duration: {duration} s ({duration/3600:.1f} hour)")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Chirp: {f0*1e3:.2f} mHz -> {f1*1e3:.2f} mHz")
    print(f"  Signal samples: {len(signal)}")
    
    # Create CWT transformer with LISA parameters
    config = CWTConfig(
        fmin=1e-4,      # 0.1 mHz
        fmax=1e-1,      # 100 mHz
        n_scales=64,
        sampling_rate=1.0,
        target_height=64,
        target_width=3600,
    )
    cwt = LISACWTTransform(config)
    
    print(f"\nCWT Configuration (LISA):")
    print(f"  Wavelet: {config.wavelet}")
    print(f"  Frequency range: {config.fmin*1e3:.2f} - {config.fmax*1e3:.1f} mHz")
    print(f"  Number of scales: {config.n_scales}")
    print(f"  Target dimensions: {config.target_height} x {config.target_width}")
    print(f"  Sampling rate: {config.sampling_rate} Hz")
    
    # Apply CWT
    print(f"\nApplying CWT...")
    cwt_matrix = cwt.transform(signal)
    
    if cwt_matrix is not None:
        print(f"  Input shape: {signal.shape}")
        print(f"  CWT shape: {cwt_matrix.shape}")
        print(f"  CWT range: [{np.min(cwt_matrix):.3f}, {np.max(cwt_matrix):.3f}]")
        print(f"  CWT mean: {np.mean(cwt_matrix):.3e}, std: {np.std(cwt_matrix):.3f}")
        
        # Plot first 10 minutes for visualization
        plot_duration = 600  # 10 minutes
        print(f"\nPlotting first {plot_duration}s for visualization...")
        fig = plot_cwt(
            signal[:plot_duration], 
            cwt_matrix[:, :plot_duration], 
            sampling_rate, 
            config.fmin, 
            config.fmax,
            save_path="results/lisa_cwt_demo.png"
        )
        
        print("\n" + "=" * 80)
        print("CWT Demo Complete!")
        print("=" * 80)
        print("\nKey Differences from LIGO:")
        print(f"  LIGO:  20-512 Hz,      4096 Hz sampling, 32s duration")
        print(f"  LISA:  {config.fmin*1e3:.1f}-{config.fmax*1e3:.0f} mHz, {config.sampling_rate} Hz sampling, {duration}s duration")
        print(f"\n  Frequency ratio: {20/config.fmin:.1e}× lower")
        print(f"  Sampling ratio:  {4096/config.sampling_rate:.0f}× slower")
        print(f"  Duration ratio:  {duration/32:.0f}× longer")
        
        print("\n(Close plot window to exit)")
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print("ERROR: CWT returned None (NaN values detected)")

