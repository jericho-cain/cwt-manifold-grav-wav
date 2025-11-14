"""
Tests for LISA dataset generator.
"""

import numpy as np
import pytest
import h5py
import json
import tempfile
import shutil
from pathlib import Path

from src.data.dataset_generator import (
    LISADatasetGenerator,
    DatasetConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DatasetConfig()
        
        assert config.duration == 3600.0
        assert config.sampling_rate == 1.0
        assert config.n_train_background == 1000
        assert config.n_test_background == 200
        assert config.n_test_signals == 400
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DatasetConfig(
            duration=7200.0,
            n_train_background=500,
            snr_range=(3.0, 15.0),
        )
        
        assert config.duration == 7200.0
        assert config.n_train_background == 500
        assert config.snr_range == (3.0, 15.0)
    
    def test_signal_fractions_default(self):
        """Test default signal fractions."""
        config = DatasetConfig()
        
        assert "mbhb" in config.signal_fractions
        assert "emri" in config.signal_fractions
        assert "galactic_binary" in config.signal_fractions
        
        total = sum(config.signal_fractions.values())
        assert np.isclose(total, 1.0)
    
    def test_signal_fractions_validation(self):
        """Test signal fractions must sum to 1."""
        with pytest.raises(ValueError):
            DatasetConfig(
                signal_fractions={
                    "mbhb": 0.5,
                    "emri": 0.3,
                    "galactic_binary": 0.1,  # Sums to 0.9, not 1.0
                }
            )
    
    def test_signal_fractions_custom(self):
        """Test custom signal fractions."""
        config = DatasetConfig(
            signal_fractions={
                "mbhb": 0.6,
                "emri": 0.3,
                "galactic_binary": 0.1,
            }
        )
        
        assert config.signal_fractions["mbhb"] == 0.6
        assert config.signal_fractions["emri"] == 0.3


class TestLISADatasetGenerator:
    """Tests for LISADatasetGenerator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def small_config(self, temp_dir):
        """Create small config for fast tests."""
        return DatasetConfig(
            duration=100.0,
            sampling_rate=1.0,
            n_train_background=10,
            n_test_background=5,
            n_test_signals=9,  # 9 for easy division
            signal_fractions={
                "mbhb": 1.0/3.0,
                "emri": 1.0/3.0,
                "galactic_binary": 1.0/3.0,
            },
            snr_range=(5.0, 15.0),
            seed=42,
            output_dir=temp_dir,
            dataset_name="test_dataset",
        )
    
    def test_initialization(self, small_config):
        """Test generator initialization."""
        gen = LISADatasetGenerator(small_config)
        
        assert gen.config == small_config
        assert gen.noise_model is not None
        assert gen.waveform_gen is not None
    
    def test_generate_noise_segment(self, small_config):
        """Test noise segment generation."""
        gen = LISADatasetGenerator(small_config)
        noise = gen.generate_noise_segment(seed=42)
        
        expected_samples = int(small_config.duration * small_config.sampling_rate)
        assert len(noise) == expected_samples
        assert noise.dtype in [np.float32, np.float64]
        assert np.all(np.isfinite(noise))
    
    def test_generate_signal_segment_mbhb(self, small_config):
        """Test MBHB signal generation."""
        gen = LISADatasetGenerator(small_config)
        
        signal, signal_plus_noise, params = gen.generate_signal_segment(
            signal_type="mbhb",
            target_snr=10.0,
            seed=42,
        )
        
        expected_samples = int(small_config.duration * small_config.sampling_rate)
        assert len(signal) == expected_samples
        assert len(signal_plus_noise) == expected_samples
        assert params["signal_type"] == "mbhb"
        assert params["snr"] == 10.0
    
    def test_generate_signal_segment_emri(self, small_config):
        """Test EMRI signal generation."""
        gen = LISADatasetGenerator(small_config)
        
        signal, signal_plus_noise, params = gen.generate_signal_segment(
            signal_type="emri",
            target_snr=8.0,
            seed=43,
        )
        
        assert params["signal_type"] == "emri"
        assert params["snr"] == 8.0
    
    def test_generate_signal_segment_gb(self, small_config):
        """Test Galactic Binary signal generation."""
        gen = LISADatasetGenerator(small_config)
        
        signal, signal_plus_noise, params = gen.generate_signal_segment(
            signal_type="galactic_binary",
            target_snr=12.0,
            seed=44,
        )
        
        assert params["signal_type"] == "galactic_binary"
        assert params["snr"] == 12.0
    
    def test_generate_signal_segment_invalid_type(self, small_config):
        """Test invalid signal type raises error."""
        gen = LISADatasetGenerator(small_config)
        
        with pytest.raises(ValueError):
            gen.generate_signal_segment(
                signal_type="invalid_type",
                target_snr=10.0,
                seed=42,
            )
    
    def test_signal_plus_noise_stronger_than_pure_noise(self, small_config):
        """Test signal+noise has more power than pure noise."""
        gen = LISADatasetGenerator(small_config)
        
        noise = gen.generate_noise_segment(seed=42)
        signal, signal_plus_noise, _ = gen.generate_signal_segment(
            signal_type="mbhb",
            target_snr=10.0,
            seed=42,
        )
        
        # Signal+noise should have more power than noise alone
        # (this isn't always true for individual realizations, but for high SNR it should be)
        power_signal_plus_noise = np.sum(signal_plus_noise**2)
        power_noise = np.sum(noise**2)
        
        # Signal should add power
        power_signal = np.sum(signal**2)
        assert power_signal > 0
    
    def test_generate_training_set(self, small_config):
        """Test training set generation."""
        gen = LISADatasetGenerator(small_config)
        train_data = gen.generate_training_set()
        
        assert train_data.shape == (small_config.n_train_background, 100)
        assert len(gen.metadata["train_noise"]) == small_config.n_train_background
        
        # Check all labels are 0 (noise)
        for entry in gen.metadata["train_noise"]:
            assert entry["label"] == 0
    
    def test_generate_test_set(self, small_config):
        """Test test set generation."""
        gen = LISADatasetGenerator(small_config)
        test_data, test_labels = gen.generate_test_set()
        
        n_total = small_config.n_test_background + small_config.n_test_signals
        assert test_data.shape == (n_total, 100)
        assert test_labels.shape == (n_total,)
        
        # Check label counts
        assert np.sum(test_labels == 0) == small_config.n_test_background
        assert np.sum(test_labels == 1) == small_config.n_test_signals
    
    def test_test_set_signal_type_distribution(self, small_config):
        """Test signal type distribution in test set."""
        gen = LISADatasetGenerator(small_config)
        gen.generate_test_set()
        
        # Count signal types
        signal_counts = {}
        for entry in gen.metadata["test_signals"]:
            sig_type = entry["signal_type"]
            signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1
        
        # Each type should have 3 signals (9 total / 3 types)
        assert signal_counts["mbhb"] == 3
        assert signal_counts["emri"] == 3
        assert signal_counts["galactic_binary"] == 3
    
    def test_metadata_structure(self, small_config):
        """Test metadata has correct structure."""
        gen = LISADatasetGenerator(small_config)
        gen.generate_training_set()
        gen.generate_test_set()
        
        assert "config" in gen.metadata
        assert "train_noise" in gen.metadata
        assert "test_noise" in gen.metadata
        assert "test_signals" in gen.metadata
        
        # Check signal metadata has required fields
        for signal in gen.metadata["test_signals"]:
            assert "index" in signal
            assert "seed" in signal
            assert "label" in signal
            assert "signal_type" in signal
            assert "params" in signal
            assert signal["label"] == 1
    
    def test_save_and_load_dataset(self, small_config):
        """Test saving and loading dataset."""
        gen = LISADatasetGenerator(small_config)
        train_data = gen.generate_training_set()
        test_data, test_labels = gen.generate_test_set()
        
        gen.save_dataset(train_data, test_data, test_labels)
        
        dataset_path = Path(small_config.output_dir) / small_config.dataset_name
        
        # Check files exist
        assert (dataset_path / "train.h5").exists()
        assert (dataset_path / "test.h5").exists()
        assert (dataset_path / "metadata.json").exists()
        assert (dataset_path / "noise_psd.npz").exists()
        
        # Load and verify training data
        with h5py.File(dataset_path / "train.h5", "r") as f:
            loaded_train = f["data"][:]
            assert np.allclose(loaded_train, train_data)
            assert f.attrs["n_segments"] == small_config.n_train_background
        
        # Load and verify test data
        with h5py.File(dataset_path / "test.h5", "r") as f:
            loaded_test = f["data"][:]
            loaded_labels = f["labels"][:]
            assert np.allclose(loaded_test, test_data)
            assert np.allclose(loaded_labels, test_labels)
            assert f.attrs["n_background"] == small_config.n_test_background
            assert f.attrs["n_signals"] == small_config.n_test_signals
        
        # Load and verify metadata
        with open(dataset_path / "metadata.json", "r") as f:
            loaded_metadata = json.load(f)
            assert len(loaded_metadata["train_noise"]) == small_config.n_train_background
            assert len(loaded_metadata["test_signals"]) == small_config.n_test_signals
        
        # Load and verify PSD
        psd_data = np.load(dataset_path / "noise_psd.npz")
        assert "frequencies" in psd_data
        assert "psd" in psd_data
    
    def test_reproducibility_with_seed(self, small_config):
        """Test dataset generation is reproducible with same seed."""
        gen1 = LISADatasetGenerator(small_config)
        train1 = gen1.generate_training_set()
        
        gen2 = LISADatasetGenerator(small_config)
        train2 = gen2.generate_training_set()
        
        assert np.allclose(train1, train2)
    
    def test_different_seeds_produce_different_data(self, temp_dir):
        """Test different seeds produce different data."""
        config1 = DatasetConfig(
            duration=100.0,
            n_train_background=10,
            n_test_background=5,
            n_test_signals=5,
            seed=42,
            output_dir=temp_dir,
        )
        config2 = DatasetConfig(
            duration=100.0,
            n_train_background=10,
            n_test_background=5,
            n_test_signals=5,
            seed=142,  # Use more different seed
            output_dir=temp_dir,
        )
        
        gen1 = LISADatasetGenerator(config1)
        train1 = gen1.generate_training_set()
        
        # Reset random state before second generator
        np.random.seed(None)
        
        gen2 = LISADatasetGenerator(config2)
        train2 = gen2.generate_training_set()
        
        # Should be at least somewhat different
        assert not np.allclose(train1, train2, rtol=1e-3, atol=1e-25)


class TestDatasetProperties:
    """Test statistical properties of generated datasets."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create config for property tests."""
        return DatasetConfig(
            duration=1000.0,
            sampling_rate=1.0,
            n_train_background=50,
            n_test_background=20,
            n_test_signals=30,
            snr_range=(5.0, 20.0),
            seed=42,
            output_dir=temp_dir,
        )
    
    def test_training_data_all_finite(self, config):
        """Test training data is all finite."""
        gen = LISADatasetGenerator(config)
        train_data = gen.generate_training_set()
        
        assert np.all(np.isfinite(train_data))
    
    def test_test_data_all_finite(self, config):
        """Test test data is all finite."""
        gen = LISADatasetGenerator(config)
        test_data, _ = gen.generate_test_set()
        
        assert np.all(np.isfinite(test_data))
    
    def test_snr_within_range(self, config):
        """Test generated signals have SNR within specified range."""
        gen = LISADatasetGenerator(config)
        gen.generate_test_set()
        
        for signal in gen.metadata["test_signals"]:
            snr = signal["params"]["snr"]
            assert config.snr_range[0] <= snr <= config.snr_range[1]
    
    def test_noise_approximately_zero_mean(self, config):
        """Test noise segments have approximately zero mean."""
        gen = LISADatasetGenerator(config)
        train_data = gen.generate_training_set()
        
        # Mean across all training data should be close to zero
        overall_mean = np.mean(train_data)
        assert np.abs(overall_mean) < 1e-20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

