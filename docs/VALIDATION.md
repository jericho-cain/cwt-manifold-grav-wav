# LISA Data Validation

## How Do We Know Our LISA Data is Realistic?

This document explains validation of our synthetic LISA data against published literature and physical expectations.

## Table of Contents

1. [Validation Approach](#validation-approach)
2. [LISA Noise Model](#lisa-noise-model)
3. [Waveform Generators](#waveform-generators)
4. [Running Validation](#running-validation)
5. [References](#references)

---

## Validation Approach

We validate our LISA data generation in three ways:

### 1. **Literature Comparison**
Compare our implementations directly to published formulas and sensitivity curves

### 2. **Physical Consistency**
Verify physical scaling laws (amplitude ∝ 1/distance, frequency evolution, etc.)

### 3. **Unit Testing**
58 automated tests ensure correctness of individual components

---

## LISA Noise Model

### Reference Implementation

Our noise model follows **Cornish & Robson (2017)**, arXiv:1703.09858, Equations 13-14.

The LISA strain noise PSD has two components:

**1. Acceleration Noise (dominates at low frequency):**
```
S_acc(f) = (3×10⁻¹⁵ m/s²/√Hz)² × [1 + (0.4 mHz/f)²] × [1 + (f/8 mHz)⁴]
```

**2. Optical Metrology System Noise (dominates at high frequency):**
```
S_oms(f) = (15×10⁻¹² m/√Hz)²
```

**Combined:**
```
S_n(f) = (10/3L²) × [S_oms + (3 + cos(2f/f*))S_acc/(2πf)²]
```

where:
- L = 2.5×10⁹ m (arm length)
- f* = 19.09 mHz (transfer frequency)

### Validation Checks

✅ **Exact Match**: Our implementation matches Cornish & Robson to machine precision  
✅ **Sensitivity Peak**: ~2-4 mHz (expected for LISA)  
✅ **Low-frequency scaling**: S_n ∝ f⁻⁴ (acceleration noise)  
✅ **High-frequency scaling**: S_n ∝ f² (shot noise)  
✅ **Amplitude**: Characteristic strain ~10⁻²⁰ Hz⁻¹/² at peak  

### Run Validation

```bash
python scripts/validation/validate_lisa_noise.py
```

This will:
- Compare to literature implementation
- Check sensitivity sweet spot location
- Verify frequency scaling
- Generate validation plots

**Expected output:**
```
✅ EXCELLENT: Implementation matches literature to machine precision
✅ PASS: Sweet spot in expected range (1-10 mHz)
✅ PASS: Characteristic strain in expected range
```

---

## Waveform Generators

### 1. Massive Black Hole Binaries (MBHBs)

**Implementation:** Post-Newtonian approximation (leading order)

**Expected Strain:**
```
h ~ (GM_chirp/c²)^(5/3) × (πf/c)^(2/3) / D
```

where M_chirp = (m₁m₂)^(3/5)/(m₁+m₂)^(1/5)

**Validation Checks:**

✅ **Amplitude Scaling**:
- Typical MBHB (10⁶/5×10⁵ M☉ @ 5 Gpc, 3 mHz): h ~ 10⁻²⁰  
- Strain ∝ 1/distance (tested)
- Strain ∝ M_chirp^(5/3) (implicit in formula)

✅ **Frequency Evolution**:
- Chirp rate: df/dt ∝ f^(11/3) (PN formula)
- Coalescence time realistic for chosen parameters

✅ **LISA Range**:
- Generated strains: 10⁻²³ to 10⁻¹⁸ ✓
- Frequencies: 10⁻⁴ to 10⁻¹ Hz ✓

### 2. Extreme Mass Ratio Inspirals (EMRIs)

**Implementation:** Quasi-circular orbit with eccentricity harmonics

**Expected Behavior:**
- Multiple harmonics due to eccentricity
- Slow inspiral due to radiation reaction
- Orbital frequency sets GW frequency (f_GW = 2 × f_orb for quadrupole)

**Validation Checks:**

✅ **Harmonic Structure**:
- Multiple peaks in FFT (3-5 harmonics detected)
- Characteristic of eccentric orbits

✅ **Amplitude**:
- Strain levels: 10⁻²² to 10⁻²⁰ for typical EMRIs ✓
- Scales with mass ratio and distance

✅ **Frequency**:
- GW frequency from Kepler's law
- In LISA band (mHz range) ✓

### 3. Galactic Binaries

**Implementation:** Nearly monochromatic with slow chirp

**Expected Behavior:**
- Nearly constant frequency
- Very slow frequency evolution (df/dt ~ 10⁻¹⁵ Hz/s)
- Smallest strain amplitudes

**Validation Checks:**

✅ **Monochromaticity**:
- FFT shows single dominant peak
- Peak within FFT resolution of specified frequency

✅ **Amplitude**:
- Strain levels: 10⁻²³ to 10⁻²⁰ ✓
- Matches specified amplitude (within inclination angle factors)

✅ **Frequency**:
- Specified frequency recovered from FFT
- In LISA band (sub-mHz to mHz range) ✓

### Run Validation

```bash
python scripts/validation/validate_lisa_waveforms.py
```

This will:
- Check strain amplitudes against theoretical predictions
- Verify frequency content
- Compute SNRs in LISA noise
- Generate waveform comparison plots

**Expected output:**
```
✅ PASS: Amplitude within factor of 10 of expected
✅ PASS: Multiple harmonics present (EMRI)
✅ PASS: Peak at expected frequency (GB)
✅ Detectable by LISA (SNR > 5)
```

---

## Signal-to-Noise Ratios

### Expected SNRs for LISA

Typical SNRs for our generated signals:

| Source Type | Parameters | Typical SNR |
|-------------|------------|-------------|
| MBHB | 10⁶/5×10⁵ M☉ @ 5 Gpc | 10-100 |
| EMRI | 10⁶ M☉ + 10 M☉ @ 3 Gpc | 5-50 |
| Galactic Binary | Nearby, f ~ 1 mHz | 1-10 |

Our generated signals have SNRs in these ranges when computed with LISA noise PSD.

### Validation

The matched-filter SNR is computed as:

```
SNR² = 4 ∫ |h̃(f)|²/S_n(f) df
```

We verify:
- SNRs are reasonable for typical sources
- Distance scaling: SNR ∝ 1/D
- Frequency dependence through S_n(f)

---

## Validation Test Suite

### Automated Tests (58 total)

Our test suite automatically validates:

**Noise Model (17 tests):**
- PSD always positive
- Correct shape and frequency range
- Reproducibility with seeds
- Zero mean, finite variance
- SNR computation

**Waveforms (18 tests):**
- Strain amplitudes in LISA range (10⁻²³ to 10⁻¹⁸)
- Correct frequency content
- Distance scaling
- Physical properties (finite, real-valued)
- Random parameter generation

**Dataset Generator (23 tests):**
- Correct signal type distribution
- SNR injection accuracy
- Metadata completeness
- File I/O integrity

### Run Tests

```bash
pytest tests/ --cov=src
```

**Current coverage: 79%**

---

## What Makes Data "Realistic"?

### 1. **Physics-Based**
All waveforms follow established approximations (PN theory, Newtonian orbits)

### 2. **Literature-Validated**
Noise model matches published LISA sensitivity curves exactly

### 3. **Scale-Appropriate**
- Masses: 10³-10⁷ M☉ (LISA-observable range)
- Distances: 0.5-20 Gpc (cosmological)
- Frequencies: 10⁻⁴-10⁻¹ Hz (LISA band)
- Strains: 10⁻²³-10⁻¹⁸ (LISA-detectable)

### 4. **Phenomenologically Correct**
- MBHBs chirp (increasing frequency)
- EMRIs have harmonics (eccentricity)
- Galactic binaries are monochromatic
- Noise has acceleration + shot noise components

### 5. **Statistically Validated**
- Reproducible with seeds
- Zero-mean noise
- Correct SNR distributions

---

## Limitations

### What Our Synthetic Data Does NOT Include

1. **Full numerical relativity**
   - We use analytical approximations, not NR waveforms
   - Sufficient for testing anomaly detectors

2. **Detector response**
   - No time-varying antenna pattern
   - No Doppler shifts from orbital motion
   - Single channel (not multi-channel TDI)

3. **Astrophysical realism**
   - Mass distributions are uniform, not astrophysical
   - No correlated signals or backgrounds
   - No galactic confusion noise

4. **Instrumental effects**
   - No glitches or data gaps
   - Perfect stationarity
   - No calibration errors

**These simplifications are acceptable** for testing manifold-based anomaly detection methods. The goal is to test whether geometric methods work on diverse signal populations, not to simulate real LISA data exactly.

---

## Comparison to Real LIGO Data

Our previous LIGO work used **real GWOSC data**. For LISA:

### Why Synthetic Data?

1. **LISA hasn't launched yet** (planned for 2030s)
2. **No real data available** for testing
3. **Synthetic data is controlled** - we know ground truth labels
4. **Can generate diverse signals** easily

### Validation Strategy

Our approach:
1. Use **published models** (Cornish & Robson for noise)
2. **Match literature formulas** exactly
3. **Verify physical scaling**
4. **Extensive unit testing**

This gives us confidence that:
- If our method works on synthetic data → worth trying on real LISA when available
- If it fails on synthetic data → no point waiting for real data

---

## References

### LISA Mission

1. **LISA Science Requirements**  
   ESA/SRE(2018)1  
   https://www.cosmos.esa.int/documents/678316/1700384/SciRD.pdf

2. **Cornish & Robson (2017)**  
   *The construction and use of LISA sensitivity curves*  
   arXiv:1703.09858  
   https://arxiv.org/abs/1703.09858

### Waveform Physics

3. **Peters & Mathews (1963)**  
   *Gravitational Radiation from Point Masses*  
   Physical Review 131, 435

4. **Blanchet (2014)**  
   *Gravitational Radiation from Post-Newtonian Sources*  
   Living Rev. Relativity 17, 2  
   https://link.springer.com/article/10.12942/lrr-2014-2

### LISA Sources

5. **Amaro-Seoane et al. (2017)**  
   *Laser Interferometer Space Antenna*  
   arXiv:1702.00786 (LISA proposal)

6. **Babak et al. (2017)**  
   *Science with the space-based interferometer LISA*  
   Physical Review D 95, 103012

---

## Validation for Approach B (Realistic LISA)

### New Validator: Confusion Noise

**`scripts/validation/validate_confusion_noise.py`** - Specific to Approach B!

```bash
python scripts/validation/validate_confusion_noise.py
```

**What it validates:**
- ✅ Confusion background has realistic statistical properties
- ✅ Resolvable sources have SNR >> confusion SNR  
- ✅ Combined signal+background behaves correctly
- ✅ Frequency content appropriate for LISA

**This is critical** because it validates our core setup:
- Training on background (not pure noise) is realistic
- Resolvable sources are actually detectable above confusion
- The problem is well-posed

### Validation Checklist

Use this checklist when generating new datasets:

- [ ] Run noise validation: `python scripts/validation/validate_lisa_noise.py`
- [ ] Run waveform validation: `python scripts/validation/validate_lisa_waveforms.py`
- [ ] **NEW**: Run confusion validation: `python scripts/validation/validate_confusion_noise.py`
- [ ] Run test suite: `pytest tests/`
- [ ] Check SNR distribution in metadata
- [ ] Visualize sample waveforms
- [ ] Verify signal type fractions match config
- [ ] **NEW**: Check confusion parameters are reasonable

If all checks pass → data is validated and ready for use! ✅

---

## Contact

Questions about validation? See:
- `docs/TESTING.md` for test documentation
- `docs/QUICK_START.md` for usage
- `docs/PROJECT_OVERVIEW.md` for context

