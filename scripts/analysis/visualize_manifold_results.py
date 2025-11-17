#!/usr/bin/env python3
"""
Visualize manifold learning results for paper figures.

This script generates visualizations showing:
1. Latent space manifold structure (t-SNE/UMAP projection)
2. Reconstruction error vs off-manifold distance
3. Beta coefficient analysis (AUC vs beta)
4. ROC curve comparison
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import h5py
import torch

# Visualization imports
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("Warning: t-SNE not available. Install scikit-learn for manifold visualization.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install umap-learn for manifold visualization.")

from sklearn.decomposition import PCA

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import create_model, load_model, CWTAutoencoder
from src.preprocessing import LISACWTTransform, CWTConfig
from src.geometry import LatentManifold


def load_latents_and_labels(model_path, data_path, config, device='cpu'):
    """Load trained model and extract latents from test data."""
    # Load model using proper load_model function
    model_config = config['model']
    
    # Determine model class based on type
    if model_config['type'] == 'cwt_ae':
        model_class = CWTAutoencoder
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    
    # Load model - load_model will use model_info from saved file
    model, metadata = load_model(
        Path(model_path),
        model_class,
        latent_dim=model_config['latent_dim'],
        lstm_hidden=model_config.get('lstm_hidden', 64),
        dropout=model_config.get('dropout', 0.1)
    )
    model.eval()
    model.to(device)
    
    # Load test data
    with h5py.File(data_path, 'r') as f:
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
    test_cwt = []
    for signal in test_data:
        cwt = cwt_transform.transform(signal)
        if cwt is not None:
            test_cwt.append(cwt)
    test_cwt = np.array(test_cwt)
    test_cwt = test_cwt[:, np.newaxis, :, :]  # Add channel dimension
    
    # Extract latents
    latents = []
    recon_errors = []
    batch_size = config['training'].get('batch_size', 4)
    
    with torch.no_grad():
        for i in range(0, len(test_cwt), batch_size):
            batch = test_cwt[i:i + batch_size]
            x = torch.FloatTensor(batch).to(device)
            recon, latent = model(x)
            
            error = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
            latents.append(latent.cpu().numpy())
            recon_errors.append(error.cpu().numpy())
    
    latents = np.vstack(latents)
    recon_errors = np.concatenate(recon_errors)
    
    return latents, recon_errors, test_labels


def plot_latent_space_manifold(train_latents, test_latents, test_labels, method='umap', save_path=None, dims=2):
    """Visualize latent space manifold structure using t-SNE or UMAP.
    
    Parameters
    ----------
    dims : int
        Number of dimensions for projection (2 or 3)
    """
    # Combine all latents
    all_latents = np.vstack([train_latents, test_latents])
    
    # Project to 2D or 3D
    if method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=dims, random_state=42, n_neighbors=15, min_dist=0.1)
        projection = reducer.fit_transform(all_latents)
    elif method == 'tsne' and TSNE_AVAILABLE:
        reducer = TSNE(n_components=dims, random_state=42, perplexity=30)
        projection = reducer.fit_transform(all_latents)
    else:
        # Fallback to PCA
        print(f"Warning: {method} not available, using PCA instead.")
        reducer = PCA(n_components=dims, random_state=42)
        projection = reducer.fit_transform(all_latents)
    
    # Split back
    n_train = len(train_latents)
    train_proj = projection[:n_train]
    test_proj = projection[n_train:]
    
    # Create plot
    if dims == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot training data (background) - on manifold
    if dims == 3:
        ax.scatter(train_proj[:, 0], train_proj[:, 1], train_proj[:, 2],
                   c='blue', alpha=0.4, s=15, label='Background (training)', marker='o')
    else:
        ax.scatter(train_proj[:, 0], train_proj[:, 1], 
                   c='blue', alpha=0.5, s=20, label='Background (training)', marker='o')
    
    # Plot test background
    test_bg_mask = test_labels == 0
    if dims == 3:
        ax.scatter(test_proj[test_bg_mask, 0], test_proj[test_bg_mask, 1], test_proj[test_bg_mask, 2],
                   c='cyan', alpha=0.6, s=30, label='Background (test)', marker='s', edgecolors='blue')
    else:
        ax.scatter(test_proj[test_bg_mask, 0], test_proj[test_bg_mask, 1],
                   c='cyan', alpha=0.6, s=30, label='Background (test)', marker='s', edgecolors='blue')
    
    # Plot test signals (off-manifold)
    test_signal_mask = test_labels == 1
    if dims == 3:
        ax.scatter(test_proj[test_signal_mask, 0], test_proj[test_signal_mask, 1], test_proj[test_signal_mask, 2],
                   c='red', alpha=0.8, s=50, label='Signals (test)', marker='^', edgecolors='darkred')
    else:
        ax.scatter(test_proj[test_signal_mask, 0], test_proj[test_signal_mask, 1],
                   c='red', alpha=0.7, s=40, label='Signals (test)', marker='^', edgecolors='darkred')
    
    if dims == 3:
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.set_zlabel(f'{method.upper()} Component 3', fontsize=12)
        ax.set_title('Latent Space Manifold Structure (3D)', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.set_title('Latent Space Manifold Structure', fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    if dims == 2:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latent space visualization ({dims}D) to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_reconstruction_vs_manifold(recon_errors, manifold_norms, labels, save_path=None):
    """Plot reconstruction error vs off-manifold distance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot background
    bg_mask = labels == 0
    ax.scatter(recon_errors[bg_mask], manifold_norms[bg_mask],
               c='blue', alpha=0.5, s=30, label='Background', marker='o')
    
    # Plot signals
    signal_mask = labels == 1
    ax.scatter(recon_errors[signal_mask], manifold_norms[signal_mask],
               c='red', alpha=0.7, s=40, label='Signals', marker='^')
    
    ax.set_xlabel('Reconstruction Error', fontsize=12)
    ax.set_ylabel('Off-Manifold Distance', fontsize=12)
    ax.set_title('Reconstruction Error vs Off-Manifold Distance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction vs manifold plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_off_manifold_distance_distribution(manifold_norms, labels, save_path=None):
    """Plot histogram of off-manifold distances for signals vs background.
    
    This visualization directly shows how signals are separated from the manifold.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bg_mask = labels == 0
    signal_mask = labels == 1
    
    bg_distances = manifold_norms[bg_mask]
    signal_distances = manifold_norms[signal_mask]
    
    # Create histogram
    bins = np.logspace(np.log10(manifold_norms.min()), np.log10(manifold_norms.max()), 50)
    
    ax.hist(bg_distances, bins=bins, alpha=0.6, color='blue', label=f'Background (n={len(bg_distances)})', 
            density=True, edgecolor='black', linewidth=0.5)
    ax.hist(signal_distances, bins=bins, alpha=0.7, color='red', label=f'Signals (n={len(signal_distances)})',
            density=True, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for medians
    bg_median = np.median(bg_distances)
    signal_median = np.median(signal_distances)
    ax.axvline(bg_median, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'Background median: {bg_median:.3f}')
    ax.axvline(signal_median, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Signal median: {signal_median:.3f}')
    
    ax.set_xlabel('Off-Manifold Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Off-Manifold Distances', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved off-manifold distance distribution to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_beta_analysis(grid_search_results, save_path=None):
    """Plot AUC vs beta coefficient."""
    # Extract results for beta analysis
    beta_values = sorted(set(r['beta'] for r in grid_search_results))
    auc_by_beta = {}
    
    for beta in beta_values:
        beta_results = [r for r in grid_search_results if r['beta'] == beta]
        best_result = max(beta_results, key=lambda x: x['auc'])
        auc_by_beta[beta] = best_result['auc']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    betas = sorted(beta_values)
    aucs = [auc_by_beta[b] for b in betas]
    
    ax.plot(betas, aucs, 'o-', linewidth=2, markersize=8, color='blue')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='AE-only (β=0)')
    ax.axhline(y=auc_by_beta[0], color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Highlight optimal beta
    optimal_beta = max(auc_by_beta.items(), key=lambda x: x[1])[0]
    optimal_auc = auc_by_beta[optimal_beta]
    ax.plot(optimal_beta, optimal_auc, 'ro', markersize=12, label=f'Optimal (β={optimal_beta}, AUC={optimal_auc:.3f})')
    
    ax.set_xlabel('Beta (Manifold Weight)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC vs Beta Coefficient', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved beta analysis plot to {save_path}")
    else:
        plt.show()
    plt.close()




def main():
    parser = argparse.ArgumentParser(description="Visualize manifold learning results")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--manifold', type=str, required=True, help='Path to manifold file')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data HDF5')
    parser.add_argument('--train-latents', type=str, help='Path to training latents (npy file)')
    parser.add_argument('--results', type=str, help='Path to grid search results JSON (optional)')
    parser.add_argument('--output-dir', type=str, default='results/figures', help='Output directory for figures')
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne', 'pca'], 
                       help='Dimensionality reduction method')
    parser.add_argument('--dims', type=int, default=2, choices=[2, 3],
                       help='Number of dimensions for latent space visualization (2 or 3)')
    parser.add_argument('--plot-distance-dist', action='store_true',
                       help='Also plot off-manifold distance distribution histogram')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load results (optional)
    grid_search_results = []
    if args.results and Path(args.results).exists():
        with open(args.results, 'r') as f:
            results_data = json.load(f)
            grid_search_results = results_data.get('all_results', [])
    
    # Load manifold
    manifold = LatentManifold.load(args.manifold)
    
    # Load training latents (if provided) or extract from manifold
    if args.train_latents:
        train_latents = np.load(args.train_latents)
    else:
        train_latents = manifold.train_latents
    
    # Load test latents and labels
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_latents, recon_errors, test_labels = load_latents_and_labels(
        args.model, args.test_data, config, device
    )
    
    # Compute off-manifold distances
    manifold_norms = manifold.batch_normal_deviation(test_latents)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Latent space manifold structure (2D)
    print(f"1. Plotting latent space manifold structure ({args.dims}D)...")
    dim_suffix = f"_{args.dims}d" if args.dims == 3 else ""
    plot_latent_space_manifold(
        train_latents, test_latents, test_labels,
        method=args.method,
        dims=args.dims,
        save_path=str(output_dir / f'latent_space_manifold{dim_suffix}.png')
    )
    
    # 1b. Also generate 3D if 2D was requested (for comparison)
    if args.dims == 2:
        print("1b. Also generating 3D visualization for comparison...")
        plot_latent_space_manifold(
            train_latents, test_latents, test_labels,
            method=args.method,
            dims=3,
            save_path=str(output_dir / 'latent_space_manifold_3d.png')
        )
    
    # 2. Reconstruction vs manifold distance
    print("2. Plotting reconstruction error vs off-manifold distance...")
    plot_reconstruction_vs_manifold(
        recon_errors, manifold_norms, test_labels,
        save_path=str(output_dir / 'reconstruction_vs_manifold.png')
    )
    
    # 3. Off-manifold distance distribution (if requested)
    if args.plot_distance_dist:
        print("3. Plotting off-manifold distance distribution...")
        plot_off_manifold_distance_distribution(
            manifold_norms, test_labels,
            save_path=str(output_dir / 'off_manifold_distance_distribution.png')
        )
    else:
        print("3. Skipping distance distribution plot (use --plot-distance-dist to enable)")
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

