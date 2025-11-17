#!/usr/bin/env python3
"""
Generate ROC and Precision-Recall curves comparing AE-only vs manifold-enhanced detection.

This script creates two plots:
1. Precision-Recall curve: AE-only (alpha=0.5, beta=0) vs optimal (alpha=0.5, beta=2.0)
2. ROC curve: Same comparison
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import torch
import yaml
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import load_model, CWTAutoencoder
from src.preprocessing import LISACWTTransform, CWTConfig
from src.geometry import LatentManifold
from src.evaluation import ManifoldScorer, ManifoldScorerConfig
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score
from tqdm import tqdm

# Set style for publication quality
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_data_and_compute_scores(config_path, model_path, manifold_path, test_data_path):
    """Load model, data, and compute scores for both cases."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_config = config['model']
    model, metadata = load_model(
        Path(model_path),
        CWTAutoencoder,
        latent_dim=model_config['latent_dim'],
        lstm_hidden=model_config.get('lstm_hidden', 64),
        dropout=model_config.get('dropout', 0.1)
    )
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Load test data
    with h5py.File(test_data_path, 'r') as f:
        test_data = f['data'][:]
        test_labels = f['labels'][:]
    
    # Preprocess with CWT
    cwt_cfg = config['preprocessing']['cwt']
    cwt_config = CWTConfig(
        fmin=cwt_cfg.get('fmin', 1e-4),
        fmax=cwt_cfg.get('fmax', 1e-1),
        n_scales=cwt_cfg.get('n_scales', 64),
        target_height=cwt_cfg.get('target_height', 64),
        target_width=cwt_cfg.get('target_width', 3600),
        sampling_rate=1.0,
        wavelet=cwt_cfg.get('wavelet', 'morl'),
        use_global_norm=cwt_cfg.get('use_global_norm', True),
        global_mean=cwt_cfg.get('global_mean'),
        global_std=cwt_cfg.get('global_std'),
    )
    
    # Compute global normalization if needed
    if cwt_config.use_global_norm and cwt_config.global_mean is None:
        n_norm = min(100, len(test_data))
        norm_samples = test_data[:n_norm]
        global_mean = np.mean(norm_samples.flatten())
        global_std = np.std(norm_samples.flatten())
        cwt_config.global_mean = float(global_mean)
        cwt_config.global_std = float(global_std)
    
    cwt_transform = LISACWTTransform(cwt_config)
    
    # Apply CWT
    print("Applying CWT to test data...")
    test_cwt = []
    for signal in tqdm(test_data, desc="CWT"):
        cwt = cwt_transform.transform(signal)
        if cwt is not None:
            test_cwt.append(cwt)
    test_cwt = np.array(test_cwt)
    test_cwt = test_cwt[:, np.newaxis, :, :]  # Add channel dimension
    
    # Extract latents and reconstruction errors
    print("Extracting latents and reconstruction errors...")
    test_latents = []
    recon_errors = []
    batch_size = config['training'].get('batch_size', 4)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_cwt), batch_size), desc="Model inference"):
            batch = test_cwt[i:i + batch_size]
            x = torch.FloatTensor(batch).to(device)
            recon, latent = model(x)
            
            error = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
            test_latents.append(latent.cpu().numpy())
            recon_errors.append(error.cpu().numpy())
    
    test_latents = np.vstack(test_latents)
    recon_errors = np.concatenate(recon_errors)
    
    # Load manifold
    manifold = LatentManifold.load(manifold_path)
    
    # Compute scores for AE-only (alpha=0.5, beta=0)
    print("Computing AE-only scores (alpha=0.5, beta=0)...")
    scorer_ae_only = ManifoldScorer(
        manifold,
        ManifoldScorerConfig(
            mode='ae_plus_manifold',
            alpha_ae=0.5,
            beta_manifold=0.0,
            use_density=False
        )
    )
    scores_ae_only = scorer_ae_only.score_batch(recon_errors, test_latents)
    scores_ae_only = scores_ae_only['combined']
    
    # Compute scores for optimal (alpha=0.5, beta=2.0)
    print("Computing optimal scores (alpha=0.5, beta=2.0)...")
    scorer_optimal = ManifoldScorer(
        manifold,
        ManifoldScorerConfig(
            mode='ae_plus_manifold',
            alpha_ae=0.5,
            beta_manifold=2.0,
            use_density=False
        )
    )
    scores_optimal = scorer_optimal.score_batch(recon_errors, test_latents)
    scores_optimal = scores_optimal['combined']
    
    return test_labels, scores_ae_only, scores_optimal


def plot_roc_curves(labels, scores_ae_only, scores_optimal, save_path=None, auc_ae=None, auc_opt=None):
    """Plot ROC curves for both cases.
    
    Parameters
    ----------
    auc_ae : float, optional
        Exact AUC value for AE-only (if None, will compute from scores)
    auc_opt : float, optional
        Exact AUC value for optimal case (if None, will compute from scores)
    """
    # Compute ROC curves
    fpr_ae, tpr_ae, _ = roc_curve(labels, scores_ae_only)
    fpr_opt, tpr_opt, _ = roc_curve(labels, scores_optimal)
    
    # Compute AUC (use provided values if available, otherwise compute)
    if auc_ae is None:
        auc_ae = roc_auc_score(labels, scores_ae_only)
    if auc_opt is None:
        auc_opt = roc_auc_score(labels, scores_optimal)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot curves
    ax.plot(fpr_ae, tpr_ae, linewidth=2, label=f'AE-only (alpha=0.5, beta=0), AUC={auc_ae:.3f}', color='blue', linestyle='--')
    ax.plot(fpr_opt, tpr_opt, linewidth=2, label=f'Manifold-enhanced (alpha=0.5, beta=2.0), AUC={auc_opt:.3f}', color='red', linestyle='-')
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random classifier (AUC=0.5)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: AE-only vs Manifold-enhanced', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pr_curves(labels, scores_ae_only, scores_optimal, save_path=None, ap_ae=None, ap_opt=None):
    """Plot Precision-Recall curves for both cases.
    
    Parameters
    ----------
    ap_ae : float, optional
        Exact AP value for AE-only (if None, will compute from scores)
    ap_opt : float, optional
        Exact AP value for optimal case (if None, will compute from scores)
    """
    # Compute PR curves
    precision_ae, recall_ae, _ = precision_recall_curve(labels, scores_ae_only)
    precision_opt, recall_opt, _ = precision_recall_curve(labels, scores_optimal)
    
    # Compute Average Precision (AP) (use provided values if available, otherwise compute)
    if ap_ae is None:
        ap_ae = average_precision_score(labels, scores_ae_only)
    if ap_opt is None:
        ap_opt = average_precision_score(labels, scores_optimal)
    
    # Baseline (random classifier)
    baseline = np.sum(labels) / len(labels)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot curves
    ax.plot(recall_ae, precision_ae, linewidth=2, label=f'AE-only (alpha=0.5, beta=0), AP={ap_ae:.3f}', color='blue', linestyle='--')
    ax.plot(recall_opt, precision_opt, linewidth=2, label=f'Manifold-enhanced (alpha=0.5, beta=2.0), AP={ap_opt:.3f}', color='red', linestyle='-')
    
    # Baseline line
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f'Random classifier (AP={baseline:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves: AE-only vs Manifold-enhanced', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate ROC and PR curves')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--manifold', type=str, required=True, help='Path to manifold file')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data HDF5')
    parser.add_argument('--output-dir', type=str, default='results/figures', help='Output directory')
    parser.add_argument('--grid-search-results', type=str, help='Path to grid search results JSON (optional, for exact AUC values)')
    
    args = parser.parse_args()
    
    # Load exact AUC values from grid search results if provided
    auc_ae = None
    auc_opt = None
    ap_ae = None
    ap_opt = None
    
    if args.grid_search_results and Path(args.grid_search_results).exists():
        with open(args.grid_search_results, 'r') as f:
            grid_results = json.load(f)
            # Find AE-only result (beta=0)
            for result in grid_results.get('all_results', []):
                if result.get('beta') == 0.0:
                    auc_ae = result.get('auc')
                    break
            # Find optimal result
            best_alpha = grid_results.get('best_alpha')
            best_beta = grid_results.get('best_beta')
            for result in grid_results.get('all_results', []):
                if result.get('alpha') == best_alpha and result.get('beta') == best_beta:
                    auc_opt = result.get('auc')
                    ap_ae = grid_results.get('all_results', [{}])[0].get('precision')  # Will compute AP from curve
                    ap_opt = result.get('precision')  # Will compute AP from curve
                    break
        if auc_ae is not None and auc_opt is not None:
            print(f"Loaded exact AUC values from grid search: AE-only={auc_ae:.3f}, Optimal={auc_opt:.3f}")
    
    # Load data and compute scores
    labels, scores_ae_only, scores_optimal = load_data_and_compute_scores(
        args.config, args.model, args.manifold, args.test_data
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_roc_curves(
        labels, scores_ae_only, scores_optimal,
        save_path=str(output_dir / 'roc_curves_comparison.png'),
        auc_ae=auc_ae,
        auc_opt=auc_opt
    )
    plot_pr_curves(
        labels, scores_ae_only, scores_optimal,
        save_path=str(output_dir / 'pr_curves_comparison.png'),
        ap_ae=ap_ae,
        ap_opt=ap_opt
    )
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()

