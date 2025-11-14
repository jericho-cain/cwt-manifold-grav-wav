# Testing Guide

## Overview

This project uses `pytest` for testing. The test suite covers:

- **LISA Noise Model** (`test_lisa_noise.py`)
- **Waveform Generators** (`test_lisa_waveforms.py`)
- **Dataset Generator** (`test_dataset_generator.py`)

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest
```

Or use the test runner script:

```bash
python scripts/run_tests.py
```

### Run Specific Test Module

```bash
# Test noise model
pytest tests/test_lisa_noise.py

# Test waveforms
pytest tests/test_lisa_waveforms.py

# Test dataset generator
pytest tests/test_dataset_generator.py
```

Or with the runner:

```bash
python scripts/run_tests.py --module lisa_noise
python scripts/run_tests.py --module lisa_waveforms
python scripts/run_tests.py --module dataset_generator
```

### Run Specific Test

```bash
# By test name
pytest tests/test_lisa_noise.py::TestLISANoise::test_initialization

# By pattern matching
pytest -k "noise"
pytest -k "mbhb"
```

Or with the runner:

```bash
python scripts/run_tests.py --test "noise"
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

Or:

```bash
python scripts/run_tests.py --coverage
```

This generates:
- Terminal coverage summary
- HTML report in `htmlcov/index.html`

### Verbose Output

```bash
pytest -vv
```

Or:

```bash
python scripts/run_tests.py --verbose
```

## Test Organization

### Unit Tests

Tests are organized by module:

```
tests/
├── test_lisa_noise.py          # LISA noise model tests
├── test_lisa_waveforms.py      # Waveform generator tests
└── test_dataset_generator.py   # Dataset generation tests
```

### Test Structure

Each test file contains multiple test classes:

**test_lisa_noise.py:**
- `TestLISANoise` - Basic functionality
- `TestLISANoisePSDShape` - PSD frequency behavior

**test_lisa_waveforms.py:**
- `TestLISAWaveformGenerator` - Generator initialization
- `TestMBHBGeneration` - MBHB waveforms
- `TestEMRIGeneration` - EMRI waveforms
- `TestGalacticBinaryGeneration` - Galactic Binary waveforms
- `TestWaveformPhysicalProperties` - Physical consistency

**test_dataset_generator.py:**
- `TestDatasetConfig` - Configuration validation
- `TestLISADatasetGenerator` - Dataset generation
- `TestDatasetProperties` - Statistical properties

## Test Categories

### Smoke Tests
Basic functionality and initialization tests.

```bash
pytest -k "initialization or basic"
```

### Physical Validation
Tests that verify physical properties (positive PSD, finite values, etc.).

```bash
pytest -k "physical or finite or positive"
```

### Reproducibility Tests
Tests that verify seeded random generation is reproducible.

```bash
pytest -k "reproducibility or seed"
```

### Integration Tests
Tests that verify full pipeline (generate → save → load).

```bash
pytest -k "save_and_load"
```

## What Each Test Suite Covers

### LISA Noise Tests (test_lisa_noise.py)

✅ Initialization with default/custom parameters  
✅ PSD is positive and correct shape  
✅ ASD is sqrt(PSD)  
✅ Time-domain noise generation  
✅ Noise statistics (zero mean, finite variance)  
✅ Reproducibility with seeds  
✅ SNR computation  
✅ PSD frequency-dependent behavior  

**Key Tests:**
- `test_psd_minimum_at_sweet_spot` - Verifies LISA sensitivity peak at ~3 mHz
- `test_noise_statistics` - Checks noise is zero mean with reasonable variance
- `test_reproducibility_with_seed` - Ensures deterministic generation

### Waveform Tests (test_lisa_waveforms.py)

✅ MBHB waveform generation  
✅ EMRI waveform generation  
✅ Galactic Binary waveform generation  
✅ Random parameter generation  
✅ Physical properties (finite, non-zero power)  
✅ Amplitude scaling with distance  
✅ Frequency content (harmonics for EMRI, monochromatic for GB)  

**Key Tests:**
- `test_mbhb_amplitude_scales_with_distance` - Verifies 1/r amplitude scaling
- `test_emri_has_harmonics` - Checks EMRI has multiple frequency components
- `test_gb_nearly_monochromatic` - Verifies GB is single-frequency
- `test_waveform_amplitudes_reasonable` - Ensures LISA-appropriate strain levels

### Dataset Generator Tests (test_dataset_generator.py)

✅ Configuration validation  
✅ Training set generation (pure noise)  
✅ Test set generation (noise + signals)  
✅ Signal type distribution  
✅ SNR injection  
✅ Metadata tracking  
✅ File I/O (HDF5 and JSON)  
✅ Reproducibility  

**Key Tests:**
- `test_signal_fractions_validation` - Ensures fractions sum to 1.0
- `test_test_set_signal_type_distribution` - Verifies correct signal mix
- `test_save_and_load_dataset` - Full pipeline: generate → save → load
- `test_snr_within_range` - Checks injected signals have target SNR

## Continuous Integration

To set up CI (e.g., GitHub Actions), create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=src --cov-report=term-missing
```

## Writing New Tests

### Template

```python
import pytest
from src.your_module import YourClass

class TestYourClass:
    """Test suite for YourClass."""
    
    @pytest.fixture
    def instance(self):
        """Create instance for testing."""
        return YourClass()
    
    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result is not None
    
    def test_edge_case(self, instance):
        """Test edge case behavior."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

### Best Practices

1. **One concept per test** - Each test should verify one thing
2. **Use descriptive names** - `test_noise_has_zero_mean` not `test_1`
3. **Use fixtures** - Avoid repetitive setup code
4. **Test edge cases** - Empty inputs, invalid values, boundary conditions
5. **Check both success and failure** - Test error handling
6. **Use assertions effectively** - `assert` with clear failure messages

## Troubleshooting

### Tests Are Slow

Skip slow tests:
```bash
python scripts/run_tests.py --fast
```

Or run specific fast tests:
```bash
pytest tests/test_lisa_noise.py::TestLISANoise::test_initialization
```

### Import Errors

Make sure you're in the project root and `src/` is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Fixture Errors

Clear pytest cache:
```bash
pytest --cache-clear
```

### Test Failures on Windows

Some tests may have platform-specific floating point differences. Adjust tolerances:
```python
assert np.allclose(a, b, rtol=1e-5, atol=1e-8)
```

## Coverage Goals

Target coverage: **>90%** for core modules

Check current coverage:
```bash
python scripts/run_tests.py --coverage
open htmlcov/index.html  # View detailed report
```

## Next Steps

After migrating autoencoder and manifold code, add:
- `tests/test_autoencoder.py` - Model architecture and training
- `tests/test_latent_manifold.py` - Manifold construction and geometry
- `tests/test_manifold_scorer.py` - Combined scoring
- `tests/test_evaluation.py` - End-to-end evaluation

---

**Summary:** Well-tested code is reliable code. Run tests often! 🧪

