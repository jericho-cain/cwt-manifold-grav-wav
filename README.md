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
- Background = instrumental noise + 1000 unresolved galactic binaries per segment
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
│   ├── geometry/          # Manifold learning and latent space analysis
│   └── evaluation/        # Scoring and evaluation
├── scripts/               # Executable scripts
│   ├── data_generation/   # Generate synthetic LISA data
│   ├── training/          # Train autoencoder
│   ├── analysis/          # Visualization and analysis
│   └── run_end_to_end_test.py  # Complete pipeline
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

**Requirements**: Python 3.8 or higher

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Replicating Paper Results (Run 3)

The paper is based on **Run 3**, which uses 5000 training samples with optimized CWT parameters. To replicate the paper's results:

```bash
# Run the complete pipeline (data generation, training, manifold construction, evaluation)
python scripts/run_end_to_end_test.py --config config/run_3_5000_samples.yaml
```

This will:
1. Generate 5000 training background segments and 600 test segments (200 background + 400 signals)
2. Preprocess data with CWT (140 scales, 100×3600 output dimensions)
3. Train the autoencoder for up to 100 epochs (with early stopping)
4. Build the k-NN manifold from training latents
5. Evaluate performance with grid search over α and β coefficients
6. Save results to `results/run_3_5000_samples/`

**Expected results** (from paper):
- Best configuration: α=0.5, β=2.0
- AUC: 0.752 (vs 0.559 for AE-only baseline)
- Precision: 0.81, Recall: 0.61

**Note**: This is a long-running process (~14+ hours). The script will save checkpoints and can be monitored via logs.

## General Workflow

### 1. Generate Realistic LISA Data
```bash
python scripts/data_generation/generate_lisa_data.py --config config/data_generation.yaml
```

Creates:
- **Training**: Background segments (noise + unresolved GBs)
- **Test**: Background + resolvable sources (MBHBs, EMRIs, bright GBs)

### 2. Train Autoencoder on Background
```bash
python scripts/training/train_lisa_ae.py --config config/training_lisa.yaml
```

Learns to reconstruct typical LISA background (confusion noise).

### 3. Build Manifold from Training Background
The manifold is automatically built during the end-to-end pipeline. To build separately:
```bash
python scripts/run_end_to_end_test.py --config config/run_3_5000_samples.yaml --skip-data --skip-training
```

Constructs k-NN manifold in latent space representing confusion structure.

### 4. Evaluate: Can We Detect Resolvable Sources?
Evaluation is included in the end-to-end pipeline. To evaluate separately:
```bash
python scripts/run_end_to_end_test.py --config config/run_3_5000_samples.yaml --skip-data --skip-training --skip-manifold
```

Tests if manifold geometry (β) helps distinguish resolvable sources from background.

## Key Parameters

- **Latent dimension**: 32
- **k-neighbors**: 32
- **Tangent space dimension**: 8
- **Weight grid search**: α ∈ [0.5, 1, 2, 5, 10], β ∈ [0, 0.01, 0.05, 0.1, 0.5, 1, 2]

## Results

Our experiments demonstrate that manifold geometry significantly improves detection performance:

- **Optimal configuration**: α=0.5, β=2.0
- **AUC**: 0.752 (vs 0.559 for AE-only baseline)
- **35% relative improvement** over autoencoder-only detection
- **Precision**: 0.81, **Recall**: 0.61

These results show that geometric structure in the latent space provides complementary information to reconstruction error, enabling better discrimination of resolvable sources from confusion background.

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
- Confusion noise generation (1000 unresolved GBs per segment)
- Dataset pipeline with HDF5 storage
- **Tests**: 74 passing (58 unit + 16 integration)

✅ **Phase 2: Preprocessing** (Complete)
- Continuous Wavelet Transform (CWT) adapted for LISA frequency range
- Frequency range: 0.1-100 mHz (vs LIGO 20-512 Hz)
- Global normalization (prevents batch effects)
- Log transform + per-segment normalization
- Optimized parameters: 140 scales, 100×3600 output
- **Tests**: 11 passing

✅ **Phase 3: Autoencoder Training** (Complete)
- CNN-based autoencoder architecture
- Trained on confusion background (5000 samples)
- 32-dimensional latent space
- **Results**: Best model achieves low reconstruction error on background

✅ **Phase 4: Manifold Geometry** (Complete)
- k-NN manifold built in latent space (k=32)
- Tangent space estimation (D=8 intrinsic dimension)
- Off-manifold distance computation
- **Results**: Signals show clear geometric separation from background

✅ **Phase 5: Evaluation** (Complete)
- Grid search over α, β coefficients
- **Key Finding**: β=2.0 optimal (manifold geometry significantly helps!)
- AUC: 0.752 (vs 0.559 for AE-only)
- Precision: 0.81, Recall: 0.61
- **35% relative improvement** over baseline

## License

MIT

## Visualization

To generate figures from the paper:

**Latent space manifold visualizations (2D and 3D):**
```bash
python scripts/analysis/visualize_manifold_results.py --config config/run_3_5000_samples.yaml --model models/run_3_5000_samples/best_model.pth --manifold models/run_3_5000_samples/manifold.npz --test-data data/raw/run_3_5000_samples/test.h5 --output-dir results/figures/run_3 --dims 2
```

**ROC and Precision-Recall curves:**
```bash
python scripts/analysis/plot_roc_pr_curves.py --config config/run_3_5000_samples.yaml --model models/run_3_5000_samples/best_model.pth --manifold models/run_3_5000_samples/manifold.npz --test-data data/raw/run_3_5000_samples/test.h5 --output-dir results/figures/run_3 --grid-search-results results/run_3_5000_samples/grid_search_results.json
```

**Architecture diagram:**
```bash
python scripts/analysis/create_architecture_diagram.py --output results/figures/run_3/architecture_diagram.png
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Manifold Learning for Source Separation in LISA Gravitational Wave Data},
  author={...},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## References

See `docs/QUICK_START.md` for detailed setup and usage instructions.

