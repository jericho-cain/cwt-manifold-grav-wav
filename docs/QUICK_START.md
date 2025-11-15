# Quick Start Guide

## Setup

1. **Create virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Generation

### Generate Default Dataset

Generate a LISA dataset with default configuration:

```bash
python scripts/data_generation/generate_lisa_data.py --config config/data_generation.yaml
```

This creates:
- 1000 training noise segments
- 300 test noise segments
- 300 test signal segments (40% MBHB, 40% EMRI, 20% Galactic Binaries)
- Each segment is 1 hour long at 1 Hz sampling rate

### Visualize Generated Data

```bash
python scripts/data_generation/visualize_data.py --dataset data/raw/lisa_dataset --output results
```

This produces:
- `results/noise_samples.png` - Example noise time series
- `results/signal_samples.png` - Example signal time series by type
- `results/noise_psd.png` - LISA noise PSD
- `results/snr_distribution.png` - SNR distribution across signal types

### Customize Dataset

Edit `config/data_generation.yaml` to customize:

```yaml
# Dataset sizes
n_train_noise: 2000    # Increase training data
n_test_signals: 500    # More test signals

# Signal diversity
signal_fractions:
  mbhb: 0.5
  emri: 0.3
  galactic_binary: 0.2

# SNR range
snr_range: [3.0, 25.0]  # Wider SNR range
```

## Data Structure

Generated datasets are stored in HDF5 format:

```
data/raw/lisa_dataset/
├── train.h5              # Training data (pure noise)
├── test.h5               # Test data (noise + signals)
├── metadata.json         # Detailed metadata for all samples
└── noise_psd.npz         # LISA noise PSD
```

### Loading Data in Python

```python
import h5py
import numpy as np
import json

# Load training data
with h5py.File("data/raw/lisa_dataset/train.h5", "r") as f:
    train_data = f["data"][:]
    print(f"Shape: {train_data.shape}")  # (1000, 3600)

# Load test data
with h5py.File("data/raw/lisa_dataset/test.h5", "r") as f:
    test_data = f["data"][:]
    test_labels = f["labels"][:]
    print(f"Data shape: {test_data.shape}")    # (600, 3600)
    print(f"Labels shape: {test_labels.shape}") # (600,)
    print(f"Noise samples: {np.sum(test_labels == 0)}")
    print(f"Signal samples: {np.sum(test_labels == 1)}")

# Load metadata
with open("data/raw/lisa_dataset/metadata.json", "r") as f:
    metadata = json.load(f)
    
# Access signal parameters
for signal in metadata["test_signals"][:3]:
    print(f"Signal {signal['index']}: {signal['signal_type']}, SNR={signal['params']['snr']:.1f}")
```

## LISA Signal Types

### 1. Massive Black Hole Binaries (MBHBs)
- Mass range: 10⁴ - 10⁷ M☉
- Frequency: ~10⁻⁴ - 10⁻² Hz
- Inspiral and merger signatures
- Most similar to LIGO signals but at lower frequencies

### 2. Extreme Mass Ratio Inspirals (EMRIs)
- Central BH: 10⁵ - 10⁷ M☉
- Small body: 1 - 100 M☉
- Complex orbital dynamics
- Rich harmonic structure due to eccentricity

### 3. Galactic Binaries
- Nearly monochromatic signals
- Slow frequency evolution
- Many overlapping sources in real LISA data
- Simplest signal type

## LISA Noise Model

The noise model includes:

1. **Acceleration noise** (low frequency)
   - Dominates at f < 1 mHz
   - From residual forces on test masses

2. **Optical metrology noise** (high frequency)
   - Dominates at f > 10 mHz
   - From laser interferometry precision

Combined characteristic strain: ~10⁻²⁰ Hz⁻¹/² at peak sensitivity (~3 mHz)

## Next Steps

After data generation:

1. **Preprocess data** - Implement whitening, normalization
2. **Train autoencoder** - Migrate autoencoder from LIGO work
3. **Build manifold** - Extract latent codes and construct k-NN manifold
4. **Evaluate** - Test AE-only vs AE+Manifold scoring

See `docs/MANIFOLD_FINAL_SUMMARY.md` for background on the manifold learning approach from LIGO work.

