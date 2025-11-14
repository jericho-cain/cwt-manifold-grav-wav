#!/usr/bin/env python
"""
Visualize generated LISA dataset.

Usage:
    python scripts/data_generation/visualize_data.py --dataset data/raw/lisa_dataset
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def plot_noise_samples(data: np.ndarray, n_samples: int = 3, save_path: str = None):
    """Plot example noise time series."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 8))
    
    if n_samples == 1:
        axes = [axes]
    
    duration = data.shape[1]
    times = np.arange(duration)
    
    for i, ax in enumerate(axes):
        ax.plot(times[:600], data[i, :600])  # Show first 600 seconds
        ax.set_ylabel('Strain')
        ax.set_title(f'Noise Sample {i+1}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def plot_signal_samples(
    data: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
    n_per_type: int = 2,
    save_path: str = None,
):
    """Plot example signal time series."""
    # Get signal indices by type
    signal_types = {}
    for entry in metadata["test_signals"]:
        sig_type = entry["signal_type"]
        if sig_type not in signal_types:
            signal_types[sig_type] = []
        signal_types[sig_type].append(entry["index"])
    
    n_types = len(signal_types)
    fig, axes = plt.subplots(n_types, n_per_type, figsize=(14, 4*n_types))
    
    if n_types == 1:
        axes = axes.reshape(1, -1)
    
    duration = data.shape[1]
    times = np.arange(duration)
    
    for row, (sig_type, indices) in enumerate(signal_types.items()):
        for col in range(min(n_per_type, len(indices))):
            idx = indices[col]
            
            ax = axes[row, col]
            ax.plot(times[:600], data[idx, :600])  # Show first 600 seconds
            ax.set_ylabel('Strain')
            ax.set_title(f'{sig_type} - Sample {col+1}')
            ax.grid(True, alpha=0.3)
            
            if row == n_types - 1:
                ax.set_xlabel('Time [s]')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def plot_psd(dataset_path: Path, save_path: str = None):
    """Plot LISA noise PSD."""
    psd_data = np.load(dataset_path / "noise_psd.npz")
    frequencies = psd_data["frequencies"]
    psd = psd_data["psd"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(frequencies, np.sqrt(frequencies * psd), 'b-', linewidth=2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Characteristic Strain $\sqrt{f \cdot S_n(f)}$ [Hz$^{-1/2}$]')
    ax.set_title('LISA Noise Characteristic Strain')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def plot_snr_distribution(metadata: dict, save_path: str = None):
    """Plot SNR distribution across signal types."""
    signal_types = {}
    
    for entry in metadata["test_signals"]:
        sig_type = entry["signal_type"]
        snr = entry["params"]["snr"]
        
        if sig_type not in signal_types:
            signal_types[sig_type] = []
        signal_types[sig_type].append(snr)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = np.arange(len(signal_types))
    width = 0.6
    
    # Box plot
    data_to_plot = [signal_types[st] for st in signal_types.keys()]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=width, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(signal_types.keys())
    ax.set_ylabel('SNR')
    ax.set_title('SNR Distribution by Signal Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize LISA dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/lisa_dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for plots",
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    
    # Load data
    with h5py.File(dataset_path / "train.h5", "r") as f:
        train_data = f["data"][:]
        print(f"Training data shape: {train_data.shape}")
    
    with h5py.File(dataset_path / "test.h5", "r") as f:
        test_data = f["data"][:]
        test_labels = f["labels"][:]
        print(f"Test data shape: {test_data.shape}")
        print(f"Test labels shape: {test_labels.shape}")
    
    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    print("1. Plotting noise samples...")
    plot_noise_samples(
        train_data,
        n_samples=3,
        save_path=str(output_path / "noise_samples.png"),
    )
    
    print("2. Plotting signal samples...")
    plot_signal_samples(
        test_data,
        test_labels,
        metadata,
        n_per_type=2,
        save_path=str(output_path / "signal_samples.png"),
    )
    
    print("3. Plotting noise PSD...")
    plot_psd(dataset_path, save_path=str(output_path / "noise_psd.png"))
    
    print("4. Plotting SNR distribution...")
    plot_snr_distribution(metadata, save_path=str(output_path / "snr_distribution.png"))
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Training segments: {len(train_data)}")
    print(f"Test segments: {len(test_data)}")
    print(f"  - Noise: {np.sum(test_labels == 0)}")
    print(f"  - Signals: {np.sum(test_labels == 1)}")
    
    print("\nSignal type distribution:")
    signal_counts = {}
    for entry in metadata["test_signals"]:
        sig_type = entry["signal_type"]
        signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1
    
    for sig_type, count in signal_counts.items():
        print(f"  {sig_type}: {count}")
    
    print(f"\nAll plots saved to {output_path}")


if __name__ == "__main__":
    main()

