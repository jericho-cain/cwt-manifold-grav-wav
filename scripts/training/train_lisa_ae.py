"""
Train LISA Autoencoder

Script to train autoencoder on LISA confusion background for manifold-based
anomaly detection.

Usage:
    python scripts/training/train_lisa_ae.py --config config/test_run.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.trainer import train_lisa_autoencoder

def main():
    parser = argparse.ArgumentParser(description='Train LISA Autoencoder')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration YAML'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting LISA autoencoder training with config: {args.config}")
    
    # Train
    trainer, results = train_lisa_autoencoder(args.config)
    
    logger.info("=" * 60)
    logger.info("Training Results:")
    logger.info(f"  Best validation loss: {results['best_val_loss']:.6f}")
    logger.info(f"  Final train loss: {results['final_train_loss']:.6f}")
    logger.info(f"  Final val loss: {results['final_val_loss']:.6f}")
    logger.info(f"  Epochs trained: {results['epochs_trained']}")
    logger.info("=" * 60)
    
    logger.info("Training complete!")

if __name__ == '__main__':
    main()

