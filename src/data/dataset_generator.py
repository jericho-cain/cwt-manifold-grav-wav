"""
LISA dataset generator for training and testing autoencoder anomaly detection.

Creates datasets with:
- Pure noise samples (training)
- Noise + signal samples (testing, labeled as anomalies)
- Multiple signal types for diversity
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import json
from tqdm import tqdm

from .lisa_noise import LISANoise
from .lisa_waveforms import (
    LISAWaveformGenerator,
    MBHBParameters,
    EMRIParameters,
    GalacticBinaryParameters,
)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # Time series parameters
    duration: float = 3600.0  # Duration of each segment in seconds
    sampling_rate: float = 1.0  # Sampling rate in Hz
    
    # Dataset sizes
    n_train_background: int = 1000  # Training on typical LISA background
    n_test_background: int = 200     # Test background (no resolvable sources)
    n_test_signals: int = 400        # Test with resolvable sources
    
    # Signal composition (must sum to 1.0)
    signal_fractions: Dict[str, float] = None
    
    # SNR ranges for resolvable signals
    snr_range: Tuple[float, float] = (10.0, 50.0)
    
    # Galactic confusion noise parameters
    confusion_enabled: bool = True
    n_confusion_sources: int = 50  # Number of unresolved GBs per segment
    confusion_snr_range: Tuple[float, float] = (0.5, 5.0)  # Weak, unresolved
    
    # Random seed
    seed: int = 42
    
    # Output paths
    output_dir: str = "data/raw"
    dataset_name: str = "lisa_dataset"
    
    def __post_init__(self):
        if self.signal_fractions is None:
            self.signal_fractions = {
                "mbhb": 0.5,
                "emri": 0.3,
                "galactic_binary": 0.2,
            }
        
        # Validate fractions sum to 1
        total = sum(self.signal_fractions.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Signal fractions must sum to 1.0, got {total}")


class LISADatasetGenerator:
    """
    Generate LISA training and test datasets.
    
    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize noise model and waveform generator
        self.noise_model = LISANoise(
            f_min=1e-4,
            f_max=config.sampling_rate / 2,  # Nyquist
            n_frequencies=2048,
        )
        
        self.waveform_gen = LISAWaveformGenerator(
            duration=config.duration,
            sampling_rate=config.sampling_rate,
        )
        
        # Set random seed
        np.random.seed(config.seed)
        
        # Storage for metadata
        self.metadata = {
            "config": asdict(config),
            "train_noise": [],
            "test_noise": [],
            "test_signals": [],
        }
    
    def generate_noise_segment(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a single noise segment.
        
        Parameters
        ----------
        seed : int, optional
            Random seed
            
        Returns
        -------
        noise : np.ndarray
            Time-domain noise
        """
        _, noise = self.noise_model.generate_noise_td(
            duration=self.config.duration,
            sampling_rate=self.config.sampling_rate,
            seed=seed,
        )
        return noise
    
    def generate_confusion_background(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate galactic confusion noise (many weak unresolved binaries).
        
        Parameters
        ----------
        seed : int, optional
            Random seed
            
        Returns
        -------
        confusion : np.ndarray
            Time-domain confusion noise (sum of many weak GBs)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = int(self.config.duration * self.config.sampling_rate)
        confusion = np.zeros(n_samples)
        
        for i in range(self.config.n_confusion_sources):
            # Random SNR for this GB
            snr = np.random.uniform(*self.config.confusion_snr_range)
            
            # Generate weak GB
            _, gb_with_noise, _ = self.generate_signal_segment(
                signal_type="galactic_binary",
                target_snr=snr,
                seed=seed + i if seed is not None else None,
            )
            
            # Add to confusion (note: gb_with_noise includes noise, we want just signal)
            # We'll regenerate properly
            params = self.waveform_gen.random_galactic_binary_params(
                seed=seed + i if seed is not None else None
            )
            gb_signal = self.waveform_gen.generate_galactic_binary(params)
            
            # Scale to target SNR
            gb_fft = np.fft.rfft(gb_signal)
            freqs = np.fft.rfftfreq(len(gb_signal), 1.0 / self.config.sampling_rate)
            current_snr = self.noise_model.snr(gb_fft, freqs)
            if current_snr > 0:
                gb_signal *= snr / current_snr
            
            confusion += gb_signal
        
        return confusion
    
    def generate_background_segment(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate typical LISA background = instrumental noise + confusion.
        
        Parameters
        ----------
        seed : int, optional
            Random seed
            
        Returns
        -------
        background : np.ndarray
            Time-domain background
        """
        noise = self.generate_noise_segment(seed=seed)
        
        if self.config.confusion_enabled:
            confusion = self.generate_confusion_background(
                seed=seed + 10000 if seed is not None else None
            )
            background = noise + confusion
        else:
            background = noise
        
        return background
    
    def generate_signal_segment(
        self,
        signal_type: str,
        target_snr: float,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate a signal+noise segment with specified SNR.
        
        Parameters
        ----------
        signal_type : str
            Type of signal: 'mbhb', 'emri', or 'galactic_binary'
        target_snr : float
            Target signal-to-noise ratio
        seed : int, optional
            Random seed
            
        Returns
        -------
        signal : np.ndarray
            Pure signal (no noise)
        signal_plus_noise : np.ndarray
            Signal + noise
        params : dict
            Source parameters
        """
        # Generate signal based on type
        if signal_type == "mbhb":
            params = self.waveform_gen.random_mbhb_params(seed=seed)
            signal = self.waveform_gen.generate_mbhb(params)
        elif signal_type == "emri":
            params = self.waveform_gen.random_emri_params(seed=seed)
            signal = self.waveform_gen.generate_emri(params)
        elif signal_type == "galactic_binary":
            params = self.waveform_gen.random_galactic_binary_params(seed=seed)
            signal = self.waveform_gen.generate_galactic_binary(params)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Compute current SNR
        signal_fd = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.config.sampling_rate)
        current_snr = self.noise_model.snr(signal_fd, freqs)
        
        # Scale signal to target SNR
        if current_snr > 0:
            scale_factor = target_snr / current_snr
            signal *= scale_factor
        
        # Generate noise and add to signal
        noise = self.generate_noise_segment(seed=seed)
        signal_plus_noise = signal + noise
        
        # Convert params to dict
        params_dict = asdict(params)
        params_dict["signal_type"] = signal_type
        params_dict["snr"] = target_snr
        
        return signal, signal_plus_noise, params_dict
    
    def generate_training_set(self) -> np.ndarray:
        """
        Generate training set (typical LISA background = noise + confusion).
        
        Returns
        -------
        train_data : np.ndarray
            Training data array of shape (n_train_background, n_samples)
        """
        print(f"Generating {self.config.n_train_background} training background segments...")
        if self.config.confusion_enabled:
            print(f"  Each segment contains {self.config.n_confusion_sources} unresolved galactic binaries")
        
        n_samples = int(self.config.duration * self.config.sampling_rate)
        train_data = np.zeros((self.config.n_train_background, n_samples))
        
        for i in tqdm(range(self.config.n_train_background)):
            seed = self.config.seed + i
            train_data[i] = self.generate_background_segment(seed=seed)
            
            self.metadata["train_noise"].append({  # Keep key name for compatibility
                "index": i,
                "seed": seed,
                "label": 0,  # 0 = background (no resolvable source)
                "confusion_enabled": self.config.confusion_enabled,
                "n_confusion": self.config.n_confusion_sources if self.config.confusion_enabled else 0,
            })
        
        return train_data
    
    def generate_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate test set (background with/without resolvable signals).
        
        Returns
        -------
        test_data : np.ndarray
            Test data array of shape (n_test_total, n_samples)
        test_labels : np.ndarray
            Test labels (0=background only, 1=background + resolvable signal)
        """
        n_test_total = self.config.n_test_background + self.config.n_test_signals
        n_samples = int(self.config.duration * self.config.sampling_rate)
        
        test_data = np.zeros((n_test_total, n_samples))
        test_labels = np.zeros(n_test_total, dtype=int)
        
        # Generate test background (no resolvable sources)
        print(f"\nGenerating {self.config.n_test_background} test background segments (no resolvable sources)...")
        seed_offset = self.config.n_train_background
        
        for i in tqdm(range(self.config.n_test_background)):
            seed = self.config.seed + seed_offset + i
            test_data[i] = self.generate_background_segment(seed=seed)
            test_labels[i] = 0
            
            self.metadata["test_noise"].append({  # Keep key name for compatibility
                "index": i,
                "seed": seed,
                "label": 0,
                "confusion_enabled": self.config.confusion_enabled,
                "n_confusion": self.config.n_confusion_sources if self.config.confusion_enabled else 0,
            })
        
        # Generate test signals (resolvable sources + background)
        print(f"\nGenerating {self.config.n_test_signals} test segments with resolvable sources...")
        if self.config.confusion_enabled:
            print(f"  Each segment contains background confusion + 1 resolvable source")
        
        # Determine number of each signal type
        signal_counts = {}
        remaining = self.config.n_test_signals
        
        for signal_type, fraction in self.config.signal_fractions.items():
            count = int(self.config.n_test_signals * fraction)
            signal_counts[signal_type] = count
            remaining -= count
        
        # Add remaining to first type
        first_type = list(signal_counts.keys())[0]
        signal_counts[first_type] += remaining
        
        # Generate signals
        idx = self.config.n_test_background
        seed_offset = self.config.n_train_background + self.config.n_test_background
        
        for signal_type, count in signal_counts.items():
            print(f"  Generating {count} {signal_type} signals...")
            
            for i in tqdm(range(count)):
                seed = self.config.seed + seed_offset + idx
                snr = np.random.uniform(*self.config.snr_range)
                
                # Start with background
                background = self.generate_background_segment(seed=seed)
                
                # Generate resolvable signal (just the signal, no noise)
                params = None
                if signal_type == "mbhb":
                    params = self.waveform_gen.random_mbhb_params(seed=seed)
                    signal = self.waveform_gen.generate_mbhb(params)
                elif signal_type == "emri":
                    params = self.waveform_gen.random_emri_params(seed=seed)
                    signal = self.waveform_gen.generate_emri(params)
                elif signal_type == "galactic_binary":
                    params = self.waveform_gen.random_galactic_binary_params(seed=seed)
                    signal = self.waveform_gen.generate_galactic_binary(params)
                
                # Scale signal to target SNR
                signal_fd = np.fft.rfft(signal)
                freqs = np.fft.rfftfreq(len(signal), 1.0 / self.config.sampling_rate)
                current_snr = self.noise_model.snr(signal_fd, freqs)
                if current_snr > 0:
                    signal *= snr / current_snr
                
                # Add to background
                test_data[idx] = background + signal
                test_labels[idx] = 1
                
                # Convert params to dict
                params_dict = asdict(params)
                params_dict["signal_type"] = signal_type
                params_dict["snr"] = snr
                
                self.metadata["test_signals"].append({
                    "index": idx,
                    "seed": seed,
                    "label": 1,
                    "signal_type": signal_type,
                    "params": params_dict,
                    "confusion_enabled": self.config.confusion_enabled,
                    "n_confusion": self.config.n_confusion_sources if self.config.confusion_enabled else 0,
                })
                
                idx += 1
        
        return test_data, test_labels
    
    def save_dataset(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ):
        """
        Save dataset to HDF5 files and metadata to JSON.
        
        Parameters
        ----------
        train_data : np.ndarray
            Training data
        test_data : np.ndarray
            Test data
        test_labels : np.ndarray
            Test labels
        """
        dataset_path = self.output_dir / self.config.dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        # Save training data
        print("\nSaving training data...")
        with h5py.File(dataset_path / "train.h5", "w") as f:
            f.create_dataset("data", data=train_data, compression="gzip")
            f.attrs["n_segments"] = len(train_data)
            f.attrs["duration"] = self.config.duration
            f.attrs["sampling_rate"] = self.config.sampling_rate
        
        # Save test data
        print("Saving test data...")
        with h5py.File(dataset_path / "test.h5", "w") as f:
            f.create_dataset("data", data=test_data, compression="gzip")
            f.create_dataset("labels", data=test_labels, compression="gzip")
            f.attrs["n_segments"] = len(test_data)
            f.attrs["n_background"] = self.config.n_test_background
            f.attrs["n_signals"] = self.config.n_test_signals
            f.attrs["duration"] = self.config.duration
            f.attrs["sampling_rate"] = self.config.sampling_rate
            f.attrs["confusion_enabled"] = self.config.confusion_enabled
            if self.config.confusion_enabled:
                f.attrs["n_confusion_sources"] = self.config.n_confusion_sources
        
        # Save metadata
        print("Saving metadata...")
        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save PSD
        print("Saving noise PSD...")
        np.savez(
            dataset_path / "noise_psd.npz",
            frequencies=self.noise_model.frequencies,
            psd=self.noise_model.psd,
        )
        
        print(f"\nDataset saved to {dataset_path}")
        print(f"  Training segments: {len(train_data)}")
        print(f"  Test background segments: {self.config.n_test_background}")
        print(f"  Test signal segments: {self.config.n_test_signals}")
        
        # Print signal type distribution
        print("\n  Signal type distribution:")
        for signal_type, fraction in self.config.signal_fractions.items():
            count = int(self.config.n_test_signals * fraction)
            print(f"    {signal_type}: {count} ({fraction*100:.1f}%)")
    
    def generate_and_save(self):
        """
        Generate complete dataset and save.
        """
        print("="*60)
        print("LISA Dataset Generation")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Duration: {self.config.duration} s")
        print(f"Sampling rate: {self.config.sampling_rate} Hz")
        print(f"SNR range: {self.config.snr_range}")
        print("="*60)
        
        # Generate datasets
        train_data = self.generate_training_set()
        test_data, test_labels = self.generate_test_set()
        
        # Save
        self.save_dataset(train_data, test_data, test_labels)
        
        print("\nDataset generation complete!")


if __name__ == "__main__":
    # Example: Generate a small dataset
    config = DatasetConfig(
        duration=3600.0,
        sampling_rate=1.0,
        n_train_noise=100,
        n_test_noise=30,
        n_test_signals=30,
        signal_fractions={
            "mbhb": 0.4,
            "emri": 0.4,
            "galactic_binary": 0.2,
        },
        snr_range=(5.0, 20.0),
        seed=42,
        output_dir="data/raw",
        dataset_name="lisa_dataset_small",
    )
    
    generator = LISADatasetGenerator(config)
    generator.generate_and_save()

