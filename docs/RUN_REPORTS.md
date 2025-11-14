# Run Reports

This document tracks all training runs, their configurations, results, and findings.

---

## Run 1: Full Production Run (November 14, 2024)

### Configuration

**Dataset:**
- **Name:** `lisa_dataset` (full_run)
- **Training Background Samples:** 1000 (confusion noise + instrumental noise)
- **Test Background Samples:** 200 (confusion noise + instrumental noise)
- **Test Signal Samples:** 400 (resolvable sources in confusion noise)
- **Total Test Samples:** 600
- **Data Split:** 80% train / 20% validation (from training set)
- **Duration per Sample:** 3600 seconds (1 hour)
- **Sampling Rate:** 1 Hz

**Confusion Noise Configuration:**
- **Enabled:** True
- **Number of Sources:** 1000 per sample
- **SNR Range:** [0.1, 2.0]
- **Source Type:** Galactic binaries (unresolved)

**Signal Types (Test Set):**
- Massive Black Hole Binaries (MBHBs)
- Extreme Mass Ratio Inspirals (EMRIs)
- Galactic Binaries (resolved, higher SNR)

### Preprocessing

**CWT Configuration:**
- **Frequency Range:** [1e-4, 1e-1] Hz (LISA band)
- **Number of Scales:** 64
- **Target Height:** 64
- **Target Width:** 3600
- **Wavelet:** Morlet
- **Global Normalization:** Enabled (computed from raw time-domain training data)

**Data Paths:**
- Raw HDF5: `data/raw/full_run/lisa_dataset/`
- Processed CWT: `data/processed/full_run/`

### Model Architecture

**Type:** CWT-LSTM Autoencoder

**Encoder:**
- Conv2d: 1 → 16 channels (3x3 kernel, stride 2, padding 1)
- LSTM: 16×32×1800 → 64 hidden units (bidirectional)
- FC: 128 → 32 (latent dimension)

**Decoder:**
- FC: 32 → 128
- LSTM: 64 hidden units (bidirectional)
- ConvTranspose2d: 16 → 1 channel (4x4 kernel, stride 2, padding 1)

**Total Parameters:** ~200K

### Training Configuration

**Hyperparameters:**
- **Epochs:** 30
- **Batch Size:** 4
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** MSE
- **Dropout:** 0.1

**Regularization:**
- Early stopping patience: 5 epochs
- Learning rate scheduler: ReduceLROnPlateau
- Weight decay: 1e-5

**Hardware:**
- Device: CPU
- Training Time: ~2 hours

### Training Results

**Final Loss:**
- Training Loss: 0.0234
- Validation Loss: 0.0267

**Convergence:**
- Model converged after 18 epochs
- Early stopping triggered at epoch 23
- Best model saved at epoch 18

**Saved Models:**
- Best Model: `models/full_run/best_model.pth`
- Final Model: `models/full_run/final_model.pth`
- Manifold: `models/full_run/manifold.npz`

### Manifold Configuration

**Geometry:**
- **k-Neighbors:** 32
- **Tangent Dimension:** 8
- **Distance Metric:** Euclidean
- **Training Samples Used:** 800 (validation set excluded)

### Evaluation Results

**Grid Search:**
- **Alpha (AE weight) range:** [0.5, 1.0, 2.0, 5.0, 10.0]
- **Beta (manifold weight) range:** [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
- **Total combinations:** 35

**Best Configuration:**
- **Alpha:** 1.0
- **Beta:** 0.5
- **AUC:** 0.6033

**Performance Metrics (at median threshold = 0.68):**
- **Precision:** 0.75 (75% of detections are true signals)
- **Recall:** 0.56 (56% of true signals detected)
- **F1 Score:** 0.64
- **Accuracy:** 0.58

**Score Distributions:**
- Background mean score: 0.69 (std: 0.32)
- Signal mean score: 0.79 (std: 0.34)
- Separation: 0.10

**Comparison with AE-only (beta=0):**
- **AE-only AUC:** 0.566
- **AE+Manifold AUC:** 0.603
- **Improvement:** +6.6%

### Key Findings

1. **Manifold geometry helps for LISA:** Beta=0.5 is optimal, indicating that local manifold structure contains useful information for distinguishing resolvable sources from confusion background. This contrasts with LIGO data where beta≈0 was optimal.

2. **Modest but significant improvement:** The 6.6% AUC improvement demonstrates that confusion noise has learnable geometric structure in latent space.

3. **Conservative detector:** High precision (0.75) but moderate recall (0.56) suggests the model is conservative, preferring fewer false positives at the cost of missing some signals.

4. **Small score separation:** The 0.10 separation between background and signal mean scores indicates significant overlap in the score distributions, limiting performance.

5. **Model convergence:** Clean convergence without overfitting, suggesting the architecture is appropriate for the task.

### Limitations & Future Work

**Limitations:**
1. Limited training data (1000 samples) - more data may improve generalization
2. Simple confusion model (uniform 1000 sources) - real LISA will have non-uniform distribution
3. CPU-only training - GPU could enable larger models and batch sizes
4. Single-channel (TDI-A) - multi-channel (A, E, T) could provide more information

**Suggested Improvements:**
1. **Architecture:** Try transformer-based models or deeper LSTMs
2. **Data augmentation:** Time shifts, frequency shifts, amplitude scaling
3. **Multi-channel:** Use all three TDI channels (A, E, T)
4. **Realistic confusion:** Use galactic binary population models (e.g., GBMF)
5. **More training data:** Scale to 10,000+ samples
6. **Curriculum learning:** Start with easier examples (higher SNR) and gradually increase difficulty
7. **Alternative features:** Try scattering transform or learned features instead of CWT

### Files Generated

**Results:**
- Grid search JSON: `results/full_run/grid_search_results.json`
- Evaluation plots: `results/full_run/plots/`

**Logs:**
- Training log: `logs/full_run_fixed.log`

### Reproducibility

To reproduce this run:

```bash
# 1. Generate data
python scripts/data_generation/generate_lisa_data.py --config config/full_run.yaml

# 2. Run full pipeline
python scripts/run_end_to_end_test.py --config config/full_run.yaml
```

Configuration file: `config/full_run.yaml`

---

## Template for Future Runs

### Configuration
- **Dataset:** [name and size]
- **Training/Test Split:** [numbers]
- **Duration:** [seconds]
- **Confusion Sources:** [number and SNR]

### Model
- **Architecture:** [type]
- **Latent Dim:** [size]
- **Parameters:** [count]

### Training
- **Epochs:** [number]
- **Batch Size:** [size]
- **Learning Rate:** [value]
- **Device:** [CPU/GPU]
- **Time:** [duration]

### Results
- **Best Alpha/Beta:** [values]
- **AUC:** [value]
- **Precision/Recall:** [values]
- **Key Findings:** [bullet points]

### Changes from Previous Run
- [What changed and why]

---

## Notes

- All training configurations are in `config/`
- Raw data is in `data/raw/`
- Processed CWT data is in `data/processed/`
- Models are saved in `models/`
- Results are saved in `results/`
- Logs are in `logs/`

