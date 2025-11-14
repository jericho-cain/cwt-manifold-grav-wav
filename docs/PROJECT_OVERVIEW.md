# Project Overview: LISA Manifold Gravitational Wave Detection

## Objective

Test whether manifold-based anomaly detection provides additional discriminative power beyond reconstruction error when applied to **diverse** gravitational wave signals.

### Hypothesis

The LIGO experiment showed manifold geometry provided no benefit (β=0) due to limited signal diversity. LISA's richer source population should provide the complexity where geometric methods can add value.

## Implementation Status

### ✅ Phase 1: Project Structure & Data Generation (COMPLETE)

**Project Structure:**
```
cwt-manifold-grav-wav/
├── src/
│   ├── data/              # Data generation
│   │   ├── lisa_noise.py           # LISA noise PSD model
│   │   ├── lisa_waveforms.py       # Waveform generators (MBHB, EMRI, GB)
│   │   └── dataset_generator.py    # Complete data pipeline
│   ├── models/            # Autoencoder (to be migrated)
│   ├── geometry/          # Manifold learning (to be migrated)
│   └── evaluation/        # Scoring and metrics (to be migrated)
├── scripts/
│   ├── data_generation/
│   │   ├── generate_lisa_data.py   # Generate datasets
│   │   └── visualize_data.py       # Visualize results
│   ├── training/          # Training scripts (pending)
│   ├── evaluation/        # Evaluation scripts (pending)
│   └── geometry/          # Manifold scripts (pending)
├── config/
│   └── data_generation.yaml        # Data generation config
├── docs/
│   ├── MANIFOLD_FINAL_SUMMARY.md   # LIGO experiment results
│   ├── QUICK_START.md              # Usage guide
│   └── PROJECT_OVERVIEW.md         # This file
├── requirements.txt
├── .gitignore
└── README.md
```

**Data Generation Components:**

1. **LISA Noise Model** (`lisa_noise.py`)
   - Acceleration noise + optical metrology noise
   - Characteristic strain ~10⁻²⁰ Hz⁻¹/² at peak sensitivity
   - Frequency range: 10⁻⁴ - 10⁻¹ Hz

2. **Waveform Generators** (`lisa_waveforms.py`)
   - **MBHB**: Post-Newtonian inspiral, mass range 10⁴-10⁷ M☉
   - **EMRI**: Quasi-circular orbits with eccentricity harmonics
   - **Galactic Binary**: Nearly monochromatic with slow chirp
   - Random parameter generation for each type

3. **Dataset Pipeline** (`dataset_generator.py`)
   - Generates training (noise only) and test (noise + signals)
   - Configurable signal type fractions
   - Target SNR injection
   - Saves to HDF5 with comprehensive metadata

### 🔄 Phase 2: Autoencoder Migration (PENDING)

**Tasks:**
- [ ] Migrate LSTM autoencoder from LIGO repo
- [ ] Adapt for LISA time series characteristics
- [ ] Implement preprocessing (whitening, normalization)
- [ ] Create training pipeline
- [ ] Add checkpointing and logging

**Expected changes:**
- May need different architecture hyperparameters
- LISA signals are longer duration, lower frequency
- Could benefit from different preprocessing

### 🔄 Phase 3: Manifold Framework (PENDING)

**Tasks:**
- [ ] Copy manifold code from LIGO work
  - `geometry/latent_manifold.py` - k-NN + tangent space
  - `evaluation/manifold_scorer.py` - Combined scoring
- [ ] Create latent extraction scripts
- [ ] Implement grid search over α, β weights
- [ ] Add per-source-type analysis

**Key parameters (from LIGO):**
- k_neighbors: 32
- tangent_dim: 8
- latent_dim: 32
- α ∈ [0.5, 1, 2, 5, 10, 20]
- β ∈ [0, 0.01, 0.05, 0.1, 0.5, 1, 2]

### 🔄 Phase 4: Evaluation (PENDING)

**Tasks:**
- [ ] Baseline evaluation (AE reconstruction error only)
- [ ] Manifold evaluation (AE + manifold scoring)
- [ ] Grid search to find optimal weights
- [ ] Performance metrics: Precision, Recall, F1, ROC-AUC
- [ ] Per-signal-type analysis
- [ ] Comparison plots and tables

**Success criteria:**
- If β > 0: Manifold geometry helps → diversity hypothesis validated
- If β = 0: Need better geometric methods or more diversity

## LISA vs LIGO: Key Differences

| Aspect | LIGO | LISA |
|--------|------|------|
| **Frequency** | 10 - 1000 Hz | 10⁻⁴ - 10⁻¹ Hz |
| **Signal duration** | 0.1 - 100 s | Hours to years |
| **Primary sources** | BBH (5-100 M☉) | MBHB (10⁴-10⁷ M☉), EMRIs, GBs |
| **Signal diversity** | Low (narrow mass range) | High (multiple source types) |
| **Waveform complexity** | Inspiral-merger-ringdown | Rich orbital dynamics, many harmonics |
| **Manifold result** | β = 0 (no benefit) | **To be tested** |

## Scientific Questions

1. **Does signal diversity enable manifold methods?**
   - LIGO: Simple manifold → β = 0
   - LISA: Complex manifold → β > 0?

2. **Which signal types benefit most from geometry?**
   - Are EMRIs geometrically distinct from MBHBs?
   - Do galactic binaries cluster separately?

3. **What is the role of SNR?**
   - Does manifold help more at low SNR?
   - Is there an SNR threshold for geometric benefit?

4. **Can we learn preprocessing-invariant features?**
   - LIGO had cross-run batch effects
   - Can manifold capture intrinsic geometry?

## Expected Timeline

- [x] **Week 1**: Project structure + data generation ✅
- [ ] **Week 2**: Autoencoder migration + training
- [ ] **Week 3**: Manifold framework + evaluation
- [ ] **Week 4**: Analysis + results + documentation

## Data Specifications

### Default Dataset (config/data_generation.yaml)

- **Training**: 1000 pure noise segments
- **Test**: 300 noise + 300 signals
  - 40% MBHB (120 signals)
  - 40% EMRI (120 signals)
  - 20% GB (60 signals)
- **Duration**: 1 hour per segment
- **Sampling**: 1 Hz
- **SNR**: 5-20 (uniform)

### Disk Usage Estimate

Each segment: 3600 samples × 8 bytes = ~29 KB

Total dataset: ~47 MB (uncompressed), ~10-15 MB (compressed HDF5)

## References

### LISA Mission
- ESA Science Requirements: ESA/SRE(2018)1
- Cornish & Robson (2017), arXiv:1703.09858

### LIGO Background
- See `docs/MANIFOLD_FINAL_SUMMARY.md`
- Previous work established null result: β = 0 optimal

### Manifold Learning
- k-NN + PCA for tangent space estimation
- Normal-bundle deviation scoring
- Grid search for optimal weight combination

## Notes

- This is a **controlled experiment** with synthetic data
- Real LISA will have additional complications:
  - Many overlapping signals (especially galactic binaries)
  - Instrumental glitches
  - Time-varying detector response
- Positive result here → worth trying on real LISA data
- Negative result here → need new geometric approaches

---

**Status**: Phase 1 complete, ready for Phase 2 (autoencoder migration)
**Last updated**: November 14, 2024

