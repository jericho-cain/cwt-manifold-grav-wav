# Test Suite

Comprehensive tests for LISA data generation code.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Test Files

- **test_lisa_noise.py** - LISA noise model (PSD, time-domain generation, SNR)
- **test_lisa_waveforms.py** - Waveform generators (MBHB, EMRI, Galactic Binary)
- **test_dataset_generator.py** - Full dataset pipeline (generation, saving, loading)

## Test Statistics

- **Total tests**: 60+
- **Target coverage**: >90%

See `docs/TESTING.md` for detailed documentation.

