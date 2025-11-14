"""
End-to-End LISA Manifold Learning Test

Runs the complete pipeline:
1. Generate test dataset
2. Train autoencoder
3. Build manifold
4. Evaluate beta coefficient

Usage:
    python scripts/run_end_to_end_test.py --config config/test_run.yaml
    
    # Run in background (Windows)
    start /B python scripts/run_end_to_end_test.py --config config/test_run.yaml > test_run.log 2>&1
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import yaml
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_generator import LISADatasetGenerator, DatasetConfig
from src.training.trainer import train_lisa_autoencoder
from src.geometry.latent_manifold import LatentManifold, LatentManifoldConfig
from src.evaluation.manifold_scorer import ManifoldScorer, ManifoldScorerConfig
import numpy as np
import torch
import h5py
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def step1_generate_data(config: dict):
    """Step 1: Generate LISA dataset."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating LISA dataset")
    logger.info("=" * 60)
    
    # Extract data config, filtering out non-DatasetConfig fields
    data_dict = config['data'].copy()
    data_dict.pop('name', None)  # Remove 'name' if present (not in DatasetConfig)
    
    data_config = DatasetConfig(**data_dict)
    generator = LISADatasetGenerator(data_config)
    
    logger.info("Generating training data (confusion background)...")
    train_data = generator.generate_training_set()
    
    logger.info("Generating test data (background + resolvable sources)...")
    test_data, test_labels = generator.generate_test_set()
    
    logger.info("Saving dataset...")
    generator.save_dataset(train_data, test_data, test_labels)
    
    logger.info(f"Dataset saved to: {data_config.output_dir}")
    logger.info(f"  Training segments: {len(train_data)}")
    logger.info(f"  Test segments: {len(test_data)}")
    logger.info(f"  Resolvable sources: {sum(test_labels)}")
    
    return train_data, test_data, test_labels


def step2_train_autoencoder(config: dict):
    """Step 2: Train autoencoder on confusion background."""
    logger.info("=" * 60)
    logger.info("STEP 2: Training autoencoder")
    logger.info("=" * 60)
    
    trainer, results = train_lisa_autoencoder(config)
    
    logger.info("Training complete!")
    logger.info(f"  Best validation loss: {results['best_val_loss']:.6f}")
    logger.info(f"  Epochs trained: {results['epochs_trained']}")
    
    return trainer, results


def step3_build_manifold(config: dict, trainer):
    """Step 3: Build k-NN manifold from training latents."""
    logger.info("=" * 60)
    logger.info("STEP 3: Building manifold")
    logger.info("=" * 60)
    
    # Load training data - DatasetGenerator creates output_dir/name subdirectory
    output_dir = Path(config['data']['output_dir'])
    dataset_name = config['data'].get('name', 'lisa_dataset')
    data_dir = output_dir / dataset_name
    
    train_path = data_dir / "train.h5"
    with h5py.File(train_path, 'r') as f:
        train_data = f['data'][:]
    
    # Extract latents from training data (confusion background)
    logger.info("Extracting training latents...")
    # Need to preprocess to CWT first
    from src.preprocessing import LISACWTTransform, CWTConfig
    
    cwt_cfg = config.get('preprocessing', {}).get('cwt', {})
    cwt_config = CWTConfig(
        fmin=cwt_cfg.get('fmin', 1e-4),
        fmax=cwt_cfg.get('fmax', 1e-1),
        n_scales=cwt_cfg.get('n_scales', 64),
        target_height=cwt_cfg.get('target_height', 64),
        target_width=cwt_cfg.get('target_width', 3600),
        sampling_rate=1.0,
        wavelet=cwt_cfg.get('wavelet', 'morl'),
        use_global_norm=True,
        global_mean=trainer.best_val_loss,  # Placeholder - would need to load properly
        global_std=1.0,
    )
    
    # Actually, trainer already has this capability
    # We need to load the CWT data that was already created during training
    # For now, let's extract latents from the training loader
    
    logger.info("Extracting latents from training data...")
    train_latents = []
    trainer.model.eval()
    
    with torch.no_grad():
        for batch in trainer.train_loader:
            x = batch[0].to(trainer.device)
            latent = trainer.model.encode(x)
            train_latents.append(latent.cpu().numpy())
    
    train_latents = np.vstack(train_latents)
    logger.info(f"Training latents shape: {train_latents.shape}")
    
    # Build manifold
    manifold_config = LatentManifoldConfig(
        k_neighbors=config['manifold']['k_neighbors'],
        tangent_dim=config['manifold'].get('tangent_dim'),
        metric=config['manifold'].get('metric', 'euclidean')
    )
    
    logger.info(f"Building k-NN manifold (k={manifold_config.k_neighbors})...")
    manifold = LatentManifold(train_latents, manifold_config)
    
    # Save manifold
    manifold_path = Path(config['training']['save_dir']) / 'manifold.npz'
    manifold.save(str(manifold_path))
    logger.info(f"Manifold saved to: {manifold_path}")
    
    return manifold, train_latents


def step4_evaluate(config: dict, trainer, manifold):
    """Step 4: Evaluate beta coefficient."""
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluating beta coefficient")
    logger.info("=" * 60)
    
    # Load test data - DatasetGenerator creates output_dir/name subdirectory
    output_dir = Path(config['data']['output_dir'])
    dataset_name = config['data'].get('name', 'lisa_dataset')
    data_dir = output_dir / dataset_name
    
    test_path = data_dir / "test.h5"
    with h5py.File(test_path, 'r') as f:
        test_data = f['data'][:]
        test_labels = f['labels'][:]
    
    logger.info(f"Test data: {len(test_data)} segments")
    logger.info(f"  Background: {np.sum(test_labels == 0)}")
    logger.info(f"  Resolvable sources: {np.sum(test_labels == 1)}")
    
    # Process test data through CWT and model
    logger.info("Processing test data through CWT and model...")
    from src.preprocessing import LISACWTTransform, CWTConfig
    
    # CWT config
    cwt_cfg = config.get('preprocessing', {}).get('cwt', {})
    cwt_config = CWTConfig(
        fmin=cwt_cfg.get('fmin', 1e-4),
        fmax=cwt_cfg.get('fmax', 1e-1),
        n_scales=cwt_cfg.get('n_scales', 64),
        target_height=cwt_cfg.get('target_height', 64),
        target_width=cwt_cfg.get('target_width', 3600),
        sampling_rate=1.0,
        wavelet=cwt_cfg.get('wavelet', 'morl'),
        use_global_norm=True,
        # Load global normalization from preprocessing
        global_mean=None,  # Will be computed if needed
        global_std=None,
    )
    
    # Check if preprocessed test data exists
    processed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/processed_cwt'))
    test_cwt_file = processed_dir / 'test_cwt.npy'
    
    if test_cwt_file.exists():
        logger.info(f"Loading preprocessed test CWT from {test_cwt_file}")
        test_cwt = np.load(test_cwt_file)
    else:
        logger.info("Preprocessing test data (not found in cache)...")
        cwt_transform = LISACWTTransform(cwt_config)
        test_cwt = []
        for signal in tqdm(test_data, desc="Test CWT"):
            cwt = cwt_transform.transform(signal)
            if cwt is not None:
                test_cwt.append(cwt)
        test_cwt = np.array(test_cwt)
    
    # Add channel dimension
    test_cwt = test_cwt[:, np.newaxis, :, :]
    
    # Process through model
    logger.info("Extracting latents and reconstruction errors...")
    test_latents = []
    recon_errors = []
    
    trainer.model.eval()
    batch_size = config['training']['batch_size']
    
    with torch.no_grad():
        for i in range(0, len(test_cwt), batch_size):
            batch = test_cwt[i:i + batch_size]
            x = torch.FloatTensor(batch).to(trainer.device)
            recon, latent = trainer.model(x)
            
            # Reconstruction error (MSE per sample)
            error = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
            
            test_latents.append(latent.cpu().numpy())
            recon_errors.append(error.cpu().numpy())
    
    test_latents = np.vstack(test_latents)
    recon_errors = np.concatenate(recon_errors)
    
    logger.info(f"Test latents shape: {test_latents.shape}")
    logger.info(f"Reconstruction errors shape: {recon_errors.shape}")
    
    # Diagnostics
    logger.info("")
    logger.info("DIAGNOSTICS:")
    logger.info(f"  Test labels unique: {np.unique(test_labels, return_counts=True)}")
    logger.info(f"  Recon error range: [{np.min(recon_errors):.6f}, {np.max(recon_errors):.6f}]")
    logger.info(f"  Recon error std: {np.std(recon_errors):.6f}")
    logger.info(f"  Latent std (mean across dims): {np.mean(np.std(test_latents, axis=0)):.6f}")
    logger.info(f"  Background recon error (mean): {np.mean(recon_errors[test_labels == 0]):.6f}")
    logger.info(f"  Signal recon error (mean): {np.mean(recon_errors[test_labels == 1]):.6f}")
    logger.info("")
    
    # Grid search over alpha, beta
    eval_config = config['evaluation']
    alpha_range = eval_config['alpha_ae_range']
    beta_range = eval_config['beta_manifold_range']
    
    logger.info(f"Grid search: alpha in {alpha_range}, beta in {beta_range}")
    
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    
    best_auc = 0
    best_alpha = 0
    best_beta = 0
    results = []
    
    for alpha in alpha_range:
        for beta in beta_range:
            # Create scorer
            scorer_config = ManifoldScorerConfig(
                mode='ae_plus_manifold',
                alpha_ae=alpha,
                beta_manifold=beta,
                use_density=eval_config.get('use_density', False),
                gamma_density=eval_config.get('gamma_density', 0.0)
            )
            
            scorer = ManifoldScorer(manifold, scorer_config)
            
            # Compute scores
            scores = scorer.score_batch(recon_errors, test_latents)
            combined_scores = scores['combined']
            
            # Compute AUC
            if len(np.unique(test_labels)) > 1:
                auc = roc_auc_score(test_labels, combined_scores)
                
                # Compute precision and recall at median threshold
                threshold = np.median(combined_scores)
                predictions = (combined_scores > threshold).astype(int)
                precision = precision_score(test_labels, predictions, zero_division=0)
                recall = recall_score(test_labels, predictions, zero_division=0)
            else:
                auc = 0.5
                precision = 0.0
                recall = 0.0
            
            results.append({
                'alpha': alpha,
                'beta': beta,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'mean_ae_error': float(np.mean(scores['ae_error'])),
                'mean_manifold_norm': float(np.mean(scores['manifold_norm'])),
            })
            
            if auc > best_auc:
                best_auc = auc
                best_alpha = alpha
                best_beta = beta
            
            logger.info(f"  alpha={alpha:.2f}, beta={beta:.2f}: AUC={auc:.4f}")
    
    # Get best result details
    best_result = [r for r in results if r['auc'] == best_auc][0]
    
    logger.info("=" * 60)
    logger.info("BEST RESULT:")
    logger.info(f"  alpha = {best_alpha}")
    logger.info(f"  beta = {best_beta}")
    logger.info(f"  AUC = {best_auc:.4f}")
    logger.info(f"  Precision = {best_result['precision']:.4f}")
    logger.info(f"  Recall = {best_result['recall']:.4f}")
    logger.info("=" * 60)
    
    # Save results
    results_dir = Path(eval_config.get('output_dir', 'results/test_run'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / 'grid_search_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_alpha': best_alpha,
            'best_beta': best_beta,
            'best_auc': best_auc,
            'all_results': results,
            'config': config
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return results, best_alpha, best_beta, best_auc


def main():
    parser = argparse.ArgumentParser(description='Run end-to-end LISA test')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to test configuration YAML'
    )
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data generation (data already exists)'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip CWT preprocessing (preprocessed data exists)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training (model already trained)'
    )
    parser.add_argument(
        '--skip-manifold',
        action='store_true',
        help='Skip manifold building (manifold already built)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'end_to_end_test_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    logger.info("=" * 60)
    logger.info("LISA MANIFOLD LEARNING: END-TO-END TEST")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Start time: {timestamp}")
    
    # Load config
    config = load_config(args.config)
    
    start_time = time.time()
    
    try:
        # Step 1: Generate data
        if not args.skip_data:
            train_data, test_data, test_labels = step1_generate_data(config)
        else:
            logger.info("Skipping data generation (--skip-data)")
        
        # Step 2: Train autoencoder
        if not args.skip_training:
            trainer, train_results = step2_train_autoencoder(config, skip_preprocessing=args.skip_preprocessing)
        else:
            logger.info("Skipping training (--skip-training), loading trained model...")
            from src.training.trainer import LISAAutoencoderTrainer
            from src.models import load_model, CWT_LSTM_Autoencoder
            
            trainer = LISAAutoencoderTrainer(config)
            
            # Load trained model
            model_path = Path(config['training']['save_dir']) / 'best_model.pth'
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {model_path}")
            
            trainer.model, metadata = load_model(
                model_path,
                CWT_LSTM_Autoencoder,
                latent_dim=config['model']['latent_dim']
            )
            trainer.model.to(trainer.device)
            logger.info(f"Loaded model from {model_path}")
            logger.info(f"  Epoch: {metadata.get('epoch', 'unknown')}")
            logger.info(f"  Val loss: {metadata.get('val_loss', 'unknown'):.6f}")
        
        # Step 3: Build manifold
        if not args.skip_manifold:
            manifold, train_latents = step3_build_manifold(config, trainer)
        else:
            logger.info("Skipping manifold building (--skip-manifold), loading saved manifold...")
            from src.geometry.latent_manifold import LatentManifold
            
            manifold_path = Path(config['training']['save_dir']) / 'manifold.npz'
            if not manifold_path.exists():
                raise FileNotFoundError(f"Manifold not found: {manifold_path}")
            
            manifold = LatentManifold.load(str(manifold_path))
            logger.info(f"Loaded manifold from {manifold_path}")
            logger.info(f"  k_neighbors: {manifold.config.k_neighbors}")
            logger.info(f"  Training points: {manifold.train_latents.shape[0]}")
        
        # Step 4: Evaluate
        results, best_alpha, best_beta, best_auc = step4_evaluate(config, trainer, manifold)
        
        # Final summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("END-TO-END TEST COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info("")
        logger.info("KEY RESULT:")
        logger.info(f"  Best beta = {best_beta}")
        logger.info(f"  Best AUC = {best_auc:.4f}")
        logger.info("")
        if best_beta > 0:
            logger.info("SUCCESS: Manifold geometry HELPS (beta > 0)!")
            logger.info("  Geometric structure distinguishes resolvable sources from confusion.")
        else:
            logger.info("NEGATIVE: Manifold geometry doesn't help (beta = 0)")
            logger.info("  Like LIGO, geometry provides no additional information.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during pipeline: {e}", exc_info=True)
        raise
    
    logger.info(f"Log saved to: {log_file}")


if __name__ == '__main__':
    main()

