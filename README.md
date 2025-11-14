<div align="center">

# LISA Manifold Gravitational Wave Detection

<img src="assets/logo.png" alt="Project Logo" width="300"/>

### Testing manifold-based detection of resolvable sources in galactic confusion noise

</div>

---

## The LISA Challenge

Unlike LIGO (sparse signals in high noise), LISA faces:
- **Low instrumental noise** (space-based)
- **~10⁷ galactic binaries** creating confusion "background"
- **Challenge**: Detect rare, loud sources (MBHBs, EMRIs) in this sea of signals

## Our Approach

**Can manifold geometry help separate resolvable sources from confusion background?**

### Training
- Learn latent manifold of "typical LISA background"
- Background = instrumental noise + 50 unresolved galactic binaries
- Manifold captures structure of confusion noise

### Testing  
- Detect resolvable sources (MBHBs, EMRIs) as deviations from background manifold
- Compare AE reconstruction error vs. AE + manifold geometry
- Test if geometric structure helps disentangle overlapping signals

### Why This is Different from LIGO

**Previous LIGO work:**
- β=0 (manifold didn't help)
- Signals too homogeneous, low diversity

**LISA approach:**
- High signal diversity + overlapping sources
- Tests if manifold can separate sources, not just detect them
- More realistic astrophysical problem

## Project Structure

```
.
├── src/                    # Source code
│   ├── data/              # Data generation and preprocessing
│   ├── models/            # Autoencoder models
│   ├── geometry/          # Manifold learning (from LIGO work)
│   └── evaluation/        # Scoring and evaluation
├── scripts/               # Executable scripts
│   ├── data_generation/   # Generate synthetic LISA data
│   ├── training/          # Train autoencoder
│   ├── evaluation/        # Evaluate performance
│   └── geometry/          # Manifold extraction and analysis
├── config/                # Configuration files
├── data/                  # Data directory (gitignored)
│   ├── raw/              # Raw generated waveforms
│   └── processed/         # Preprocessed for training
├── models/                # Saved model checkpoints (gitignored)
├── results/               # Experiment results
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Workflow

### 1. Generate Realistic LISA Data
```bash
python scripts/data_generation/generate_lisa_data.py --config config/data_generation.yaml
```

Creates:
- **Training**: 1000 segments of background (noise + 50 unresolved GBs each)
- **Test**: 200 background + 400 with resolvable sources (MBHBs, EMRIs, bright GBs)

### 2. Train Autoencoder on Background
```bash
python scripts/training/train_autoencoder.py --config config/training.yaml
```

Learns to reconstruct typical LISA background (confusion noise).

### 3. Build Manifold from Training Background
```bash
python scripts/geometry/build_manifold.py --model models/best_model.pth
```

Constructs k-NN manifold in latent space representing confusion structure.

### 4. Evaluate: Can We Detect Resolvable Sources?
```bash
python scripts/evaluation/evaluate_manifold.py --config config/evaluation.yaml
```

Tests if manifold geometry (β) helps distinguish resolvable sources from background.

## Key Parameters (from LIGO work)

- **Latent dimension**: 32
- **k-neighbors**: 32
- **Tangent space dimension**: 8
- **Weight grid search**: α ∈ [0.5, 1, 2, 5, 10, 20], β ∈ [0, 0.01, 0.05, 0.1, 0.5, 1, 2]

## Expected Outcome

### Success Criteria

**If β > 0** (manifold geometry helps):
- Geometric structure distinguishes loud sources from confusion
- Manifold learning useful for source separation
- Novel contribution to LISA data analysis

**If β = 0** (manifold doesn't help):
- Even with overlapping signals, geometry provides no benefit
- Need different approaches (ICA, blind source separation)
- Important negative result for the field

## The Scientific Question

**Can geometric priors from manifold learning improve detection/separation of interesting sources in realistic LISA confusion noise?**

This is fundamentally different from LIGO:
- Not "signal vs. noise" but "interesting signal vs. background signals"
- Tests if manifold structure can disentangle overlapping sources
- More ambitious than diversity hypothesis alone

## Project Status

✅ **Phase 1: Data Generation** (Complete)
- LISA noise model implemented
- Waveform generators (MBHB, EMRI, GB)
- Confusion noise generation (50 unresolved GBs)
- Dataset pipeline with HDF5 storage
- **Tests**: 74 passing (58 unit + 16 integration)

✅ **Phase 2: Preprocessing** (Complete)
- CWT adapted from LIGO legacy code
- Frequency range: 0.1-100 mHz (vs LIGO 20-512 Hz)
- Global normalization (prevents batch effects)
- Log transform + per-segment normalization
- **Tests**: 11 passing

⏳ **Phase 3: Autoencoder Training** (Next - ~2 weeks)
- Migrate autoencoder from LIGO repo → ✅ Legacy code reviewed
- Preprocess LISA dataset to CWT format
- Train on confusion background
- Build k-NN manifold in latent space
- Grid search α, β coefficients
- **Measure β for LISA!**

⏳ **Phase 4: Manifold Geometry**
- Build k-NN manifold in latent space
- Extract tangent space geometry
- Compute off-manifold scores

⏳ **Phase 5: Evaluation**
- Compare AE vs AE+manifold (β coefficient)
- ROC curves, confusion matrices
- Statistical significance testing

## License

MIT

## References

See `docs/MANIFOLD_FINAL_SUMMARY.md` for complete LIGO experiment documentation.

