#!/usr/bin/env python3
"""
Create ML architecture diagram for CNN autoencoder with manifold learning.

This script generates a publication-quality diagram showing:
1. CNN autoencoder architecture (encoder-decoder)
2. Manifold learning branch (k-NN + PCA)
3. Combined scoring mechanism
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set style for publication quality
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def create_architecture_diagram(save_path='results/figures/architecture_diagram.png'):
    """Create the full architecture diagram."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    color_input = '#E8F4F8'  # Light blue
    color_encoder = '#B3D9E6'  # Medium blue
    color_latent = '#4A90E2'  # Blue
    color_decoder = '#E6F3FF'  # Light blue
    color_manifold = '#FFE6CC'  # Light orange
    color_score = '#FFB366'  # Orange
    color_output = '#FF6B6B'  # Red
    color_text = '#333333'
    
    # Box style
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', linewidth=1.5)
    
    # ========== INPUT ==========
    input_box = FancyBboxPatch((0.5, 8.5), 1.5, 1.0, 
                               boxstyle='round,pad=0.1', 
                               facecolor=color_input, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 9.2, 'CWT Scalogram', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.25, 8.8, '(64 × 3600)', ha='center', va='center', fontsize=9)
    
    # ========== ENCODER PATH ==========
    # Conv2d layers
    conv1_box = FancyBboxPatch((2.5, 8.5), 1.2, 1.0,
                               boxstyle='round,pad=0.1',
                               facecolor=color_encoder,
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(conv1_box)
    ax.text(3.1, 9.2, 'Conv2d', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.1, 8.8, '1→16, k=3', ha='center', va='center', fontsize=8)
    
    conv2_box = FancyBboxPatch((4.0, 8.5), 1.2, 1.0,
                               boxstyle='round,pad=0.1',
                               facecolor=color_encoder,
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(conv2_box)
    ax.text(4.6, 9.2, 'Conv2d', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.6, 8.8, '16→32, k=3', ha='center', va='center', fontsize=8)
    
    # Adaptive pooling
    pool_box = FancyBboxPatch((5.5, 8.5), 1.0, 1.0,
                              boxstyle='round,pad=0.1',
                              facecolor=color_encoder,
                              edgecolor='black', linewidth=1.5)
    ax.add_patch(pool_box)
    ax.text(6.0, 9.2, 'Adaptive', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(6.0, 8.8, 'Pool (4×4)', ha='center', va='center', fontsize=8)
    
    # Linear layers
    linear1_box = FancyBboxPatch((7.0, 8.5), 1.0, 1.0,
                                 boxstyle='round,pad=0.1',
                                 facecolor=color_encoder,
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(linear1_box)
    ax.text(7.5, 9.2, 'Linear', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.5, 8.8, '512→32', ha='center', va='center', fontsize=8)
    
    # ========== LATENT SPACE ==========
    latent_box = FancyBboxPatch((8.5, 7.5), 1.0, 2.0,
                                boxstyle='round,pad=0.1',
                                facecolor=color_latent,
                                edgecolor='black', linewidth=2.5)
    ax.add_patch(latent_box)
    ax.text(9.0, 9.0, 'Latent', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(9.0, 8.6, 'Space', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(9.0, 8.2, 'z in R^32', ha='center', va='center', fontsize=10, color='white')
    
    # ========== DECODER PATH ==========
    # Linear
    linear2_box = FancyBboxPatch((7.0, 6.5), 1.0, 1.0,
                                 boxstyle='round,pad=0.1',
                                 facecolor=color_decoder,
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(linear2_box)
    ax.text(7.5, 7.0, 'Linear', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.5, 6.7, '32→512', ha='center', va='center', fontsize=8)
    
    # ConvTranspose layers
    deconv1_box = FancyBboxPatch((5.5, 6.5), 1.0, 1.0,
                                 boxstyle='round,pad=0.1',
                                 facecolor=color_decoder,
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(deconv1_box)
    ax.text(6.0, 7.0, 'ConvTranspose2d', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(6.0, 6.7, '16→8', ha='center', va='center', fontsize=8)
    
    deconv2_box = FancyBboxPatch((4.0, 6.5), 1.2, 1.0,
                                 boxstyle='round,pad=0.1',
                                 facecolor=color_decoder,
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(deconv2_box)
    ax.text(4.6, 7.0, 'ConvTranspose2d', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.6, 6.7, '8→1', ha='center', va='center', fontsize=8)
    
    # Reconstructed output
    recon_box = FancyBboxPatch((2.5, 6.5), 1.2, 1.0,
                               boxstyle='round,pad=0.1',
                               facecolor=color_input,
                               edgecolor='black', linewidth=2)
    ax.add_patch(recon_box)
    ax.text(3.1, 7.0, 'Reconstructed', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.1, 6.7, 'CWT (64×3600)', ha='center', va='center', fontsize=8)
    
    # ========== RECONSTRUCTION ERROR ==========
    error_box = FancyBboxPatch((0.5, 6.0), 1.5, 0.8,
                               boxstyle='round,pad=0.1',
                               facecolor=color_score,
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(error_box)
    ax.text(1.25, 6.5, 'Reconstruction', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.25, 6.2, 'Error eps(x)', ha='center', va='center', fontsize=9)
    
    # ========== MANIFOLD LEARNING BRANCH ==========
    # k-NN
    knn_box = FancyBboxPatch((0.5, 4.5), 1.5, 0.8,
                             boxstyle='round,pad=0.1',
                             facecolor=color_manifold,
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(knn_box)
    ax.text(1.25, 5.0, 'k-NN Search', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.25, 4.7, 'k=32', ha='center', va='center', fontsize=8)
    
    # PCA
    pca_box = FancyBboxPatch((0.5, 3.5), 1.5, 0.8,
                             boxstyle='round,pad=0.1',
                             facecolor=color_manifold,
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(pca_box)
    ax.text(1.25, 4.0, 'Local PCA', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.25, 3.7, 'Tangent Space', ha='center', va='center', fontsize=8)
    
    # Off-manifold distance
    dist_box = FancyBboxPatch((0.5, 2.5), 1.5, 0.8,
                              boxstyle='round,pad=0.1',
                              facecolor=color_manifold,
                              edgecolor='black', linewidth=1.5)
    ax.add_patch(dist_box)
    ax.text(1.25, 3.0, 'Off-Manifold', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.25, 2.7, 'Distance delta_perp(z)', ha='center', va='center', fontsize=9)
    
    # ========== COMBINED SCORING ==========
    score_box = FancyBboxPatch((2.5, 2.5), 2.5, 1.2,
                               boxstyle='round,pad=0.15',
                               facecolor=color_output,
                               edgecolor='black', linewidth=2.5)
    ax.add_patch(score_box)
    ax.text(3.75, 3.4, 'Combined Score', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(3.75, 3.0, 's(x) = alpha*eps(x) + beta*delta_perp(z)', ha='center', va='center', fontsize=9, color='white', 
            family='monospace')
    ax.text(3.75, 2.7, 'alpha=0.5, beta=2.0', ha='center', va='center', fontsize=9, color='white')
    
    # ========== ARROWS ==========
    # Encoder path (left to right)
    arrow1 = FancyArrowPatch((2.0, 9.0), (2.5, 9.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((3.7, 9.0), (4.0, 9.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow2)
    arrow3 = FancyArrowPatch((5.2, 9.0), (5.5, 9.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((6.5, 9.0), (7.0, 9.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow4)
    arrow5 = FancyArrowPatch((8.0, 9.0), (8.5, 8.5), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow5)
    
    # Decoder path (right to left)
    arrow6 = FancyArrowPatch((8.5, 7.5), (8.0, 7.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow6)
    arrow7 = FancyArrowPatch((7.0, 7.0), (6.5, 7.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow7)
    arrow8 = FancyArrowPatch((5.5, 7.0), (5.2, 7.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow8)
    arrow9 = FancyArrowPatch((4.0, 7.0), (3.7, 7.0), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow9)
    arrow10 = FancyArrowPatch((2.5, 7.0), (2.0, 6.8), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow10)
    
    # Reconstruction error arrow
    arrow11 = FancyArrowPatch((2.0, 6.8), (2.0, 6.4), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow11)
    
    # Manifold learning branch (from latent space) - right angle path
    # First segment: straight down from latent space
    arrow12a = FancyArrowPatch((9.0, 7.5), (9.0, 5.5), 
                               arrowstyle='->', lw=2, color='#FF8C00', linestyle='--')
    ax.add_patch(arrow12a)
    # Second segment: horizontal to left
    arrow12b = FancyArrowPatch((9.0, 5.5), (2.0, 5.5), 
                               arrowstyle='->', lw=2, color='#FF8C00', linestyle='--')
    ax.add_patch(arrow12b)
    # Third segment: down to k-NN box
    arrow12c = FancyArrowPatch((2.0, 5.5), (2.0, 4.9), 
                               arrowstyle='->', lw=2, color='#FF8C00', linestyle='--')
    ax.add_patch(arrow12c)
    arrow13 = FancyArrowPatch((1.25, 4.5), (1.25, 4.3), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow13)
    arrow14 = FancyArrowPatch((1.25, 3.5), (1.25, 3.3), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow14)
    arrow15 = FancyArrowPatch((1.25, 2.5), (1.25, 2.3), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow15)
    
    # To combined score
    arrow16 = FancyArrowPatch((2.0, 3.0), (2.5, 3.1), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow16)
    arrow17 = FancyArrowPatch((2.0, 6.4), (2.5, 3.1), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow17)
    
    # ========== LABELS ==========
    # Section labels
    ax.text(5.0, 9.8, 'Encoder', ha='center', va='center', fontsize=12, fontweight='bold', color=color_text)
    ax.text(5.0, 5.8, 'Decoder', ha='center', va='center', fontsize=12, fontweight='bold', color=color_text)
    ax.text(0.15, 3.9, 'Manifold Learning', ha='left', va='center', fontsize=11, fontweight='bold', 
            rotation=90, color=color_text)
    
    # Training data note
    training_note = Rectangle((0.2, 0.5), 2.1, 1.2, 
                             facecolor='#F0F0F0', edgecolor='gray', linewidth=1, linestyle='--')
    ax.add_patch(training_note)
    ax.text(1.25, 1.5, 'Training Phase:', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.25, 1.2, 'Build k-NN index on', ha='center', va='center', fontsize=9)
    ax.text(1.25, 1.0, 'training latents', ha='center', va='center', fontsize=9)
    ax.text(1.25, 0.7, '(confusion background)', ha='center', va='center', fontsize=8, style='italic')
    
    # Title
    ax.text(5.0, 10.3, 'CNN Autoencoder with Manifold Learning Architecture', 
           ha='center', va='center', fontsize=14, fontweight='bold', color=color_text)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create ML architecture diagram')
    parser.add_argument('--output', type=str, default='results/figures/architecture_diagram.png',
                       help='Output path for the diagram')
    args = parser.parse_args()
    
    from pathlib import Path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_architecture_diagram(str(output_path))

