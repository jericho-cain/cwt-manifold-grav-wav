"""
LISA Autoencoder Trainer

Trains autoencoder on LISA confusion background data for manifold-based
anomaly detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import h5py
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from tqdm import tqdm
import yaml

from src.models import create_model, save_model, load_model
from src.preprocessing import LISACWTTransform, CWTConfig

logger = logging.getLogger(__name__)


class LISAAutoencoderTrainer:
    """
    Trainer for LISA autoencoder on confusion background.
    
    Handles complete training pipeline:
    - Load HDF5 data
    - Apply CWT preprocessing
    - Train autoencoder
    - Validation and early stopping
    - Save checkpoints and latents
    
    Parameters
    ----------
    config : Dict or Path
        Training configuration dictionary or path to YAML file
    """
    
    def __init__(self, config: Any):
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.criterion: Optional[nn.Module] = None
        
        # Data loaders
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.epochs_without_improvement = 0
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load HDF5 data and apply CWT preprocessing.
        
        Returns
        -------
        train_cwt : np.ndarray
            Training data in CWT format (N, H, W)
        test_cwt : np.ndarray
            Test data in CWT format (M, H, W)
        """
        logger.info("Loading and preprocessing data...")
        
        # Paths - DatasetGenerator creates output_dir/name subdirectory
        output_dir = Path(self.config['data']['output_dir'])
        dataset_name = self.config['data'].get('name', 'lisa_dataset')
        data_dir = output_dir / dataset_name
        
        train_path = data_dir / "train.h5"
        test_path = data_dir / "test.h5"
        
        # CWT config
        cwt_cfg = self.config.get('preprocessing', {}).get('cwt', {})
        cwt_config = CWTConfig(
            fmin=cwt_cfg.get('fmin', 1e-4),
            fmax=cwt_cfg.get('fmax', 1e-1),
            n_scales=cwt_cfg.get('n_scales', 64),
            target_height=cwt_cfg.get('target_height', 64),
            target_width=cwt_cfg.get('target_width', 3600),
            sampling_rate=1.0,
            wavelet=cwt_cfg.get('wavelet', 'morl'),
            use_global_norm=cwt_cfg.get('use_global_norm', True),
        )
        
        # Step 1: Compute global normalization from RAW training background
        # NOTE: Must compute from RAW time-domain data, NOT from CWT outputs!
        # Computing from CWT outputs gives useless stats (mean≈0, std=1) since
        # CWT already does per-segment normalization.
        logger.info("Computing global normalization statistics from raw time-domain data...")
        with h5py.File(train_path, 'r') as f:
            train_data = f['data'][:]
        
        # Sample for normalization (use first 100 segments or all if fewer)
        n_norm_samples = min(100, len(train_data))
        norm_samples = train_data[:n_norm_samples]
        
        # Compute global mean/std directly from raw time-domain data
        # This is the CORRECT approach - same as legacy LIGO code
        all_values = norm_samples.flatten()
        global_mean = np.mean(all_values)
        global_std = np.std(all_values)
        
        logger.info(f"Global normalization (from raw data): mean={global_mean:.6e}, std={global_std:.6e}")
        
        # Update CWT config with global stats
        cwt_config.global_mean = float(global_mean)
        cwt_config.global_std = float(global_std)
        
        # Step 2: Create CWT transformer
        cwt_transform = LISACWTTransform(cwt_config)
        
        # Step 3: Process training data
        logger.info(f"Processing {len(train_data)} training segments...")
        train_cwt = []
        for signal in tqdm(train_data, desc="Training CWT"):
            cwt = cwt_transform.transform(signal)
            if cwt is not None:
                train_cwt.append(cwt)
        train_cwt = np.array(train_cwt)
        
        # Step 4: Process test data
        logger.info(f"Processing test data...")
        with h5py.File(test_path, 'r') as f:
            test_data = f['data'][:]
        
        test_cwt = []
        for signal in tqdm(test_data, desc="Test CWT"):
            cwt = cwt_transform.transform(signal)
            if cwt is not None:
                test_cwt.append(cwt)
        test_cwt = np.array(test_cwt)
        
        logger.info(f"Training CWT shape: {train_cwt.shape}")
        logger.info(f"Test CWT shape: {test_cwt.shape}")
        
        # Save preprocessed CWT data for future runs
        processed_dir = Path(self.config.get('preprocessing', {}).get('output_dir', 'data/processed_cwt'))
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(processed_dir / 'train_cwt.npy', train_cwt)
        np.save(processed_dir / 'test_cwt.npy', test_cwt)
        logger.info(f"Saved preprocessed CWT data to {processed_dir}")
        
        return train_cwt, test_cwt
    
    def setup_model(self):
        """Initialize model, optimizer, criterion, and scheduler."""
        logger.info("Setting up model...")
        
        # Create model
        model_config = self.config['model']
        self.model = create_model(
            model_config['type'],
            input_height=model_config['input_height'],
            input_width=model_config['input_width'],
            latent_dim=model_config.get('latent_dim', 32),
            lstm_hidden=model_config.get('lstm_hidden', 64),
            dropout=model_config.get('dropout', 0.1),
        )
        self.model.to(self.device)
        
        # Print model info
        info = self.model.get_model_info()
        logger.info(f"Model: {info['architecture']}")
        logger.info(f"Parameters: {info['total_parameters']:,}")
        logger.info(f"Model size: {info['model_size_mb']:.2f} MB")
        
        # Optimizer
        train_config = self.config['training']
        if train_config.get('optimizer', 'adam').lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config.get('weight_decay', 1e-5)
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                momentum=train_config.get('momentum', 0.9),
                weight_decay=train_config.get('weight_decay', 1e-5)
            )
        
        # Loss function
        loss_fn = train_config.get('loss_function', 'mse').lower()
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_fn == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        # Scheduler
        scheduler_type = train_config.get('scheduler', 'reduce_on_plateau').lower()
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs']
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer: {train_config.get('optimizer', 'adam')}")
        logger.info(f"Loss: {loss_fn}")
        logger.info(f"LR scheduler: {scheduler_type}")
    
    def setup_data_loaders(self, train_cwt: np.ndarray):
        """
        Create train/val data loaders.
        
        Parameters
        ----------
        train_cwt : np.ndarray
            Training data in CWT format
        """
        logger.info("Setting up data loaders...")
        
        # Add channel dimension
        train_cwt = train_cwt[:, np.newaxis, :, :]  # (N, 1, H, W)
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_cwt)
        
        # Train/val split
        val_split = self.config['training']['validation_split']
        n_train = int(len(train_tensor) * (1 - val_split))
        n_val = len(train_tensor) - n_train
        
        train_dataset, val_dataset = random_split(
            TensorDataset(train_tensor),
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Data loaders
        batch_size = self.config['training']['batch_size']
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=torch.cuda.is_available()
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Training samples: {n_train}")
        logger.info(f"Validation samples: {n_val}")
        logger.info(f"Batch size: {batch_size}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns
        -------
        avg_loss : float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            x = batch[0].to(self.device)
            
            # Forward pass
            recon, latent = self.model(x)
            loss = self.criterion(recon, x)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate model.
        
        Returns
        -------
        avg_loss : float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch[0].to(self.device)
                
                # Forward pass
                recon, latent = self.model(x)
                loss = self.criterion(recon, x)
                
                total_loss += loss.item() * x.size(0)
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns
        -------
        results : Dict
            Training results and metrics
        """
        train_config = self.config['training']
        num_epochs = train_config['num_epochs']
        patience = train_config.get('early_stopping_patience', 5)
        min_delta = train_config.get('early_stopping_min_delta', 1e-4)
        save_every = train_config.get('save_every_n_epochs', 5)
        
        model_dir = Path(train_config.get('save_dir', 'models'))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch}/{num_epochs}: "
                       f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss - min_delta:
                logger.info(f"Validation loss improved: {self.best_val_loss:.6f} → {val_loss:.6f}")
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                best_path = model_dir / 'best_model.pth'
                save_model(self.model, best_path, {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': self.config
                })
                logger.info(f"Saved best model to {best_path}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Periodic checkpoint
            if epoch % save_every == 0:
                checkpoint_path = model_dir / f"checkpoint_epoch_{epoch}.pth"
                save_model(self.model, checkpoint_path, {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_path = model_dir / 'final_model.pth'
        save_model(self.model, final_path, {
            'epoch': epoch,
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'config': self.config
        })
        logger.info(f"Saved final model to {final_path}")
        
        # Results
        results = {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'epochs_trained': len(self.train_losses),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        return results
    
    def extract_latents(self, data: np.ndarray) -> np.ndarray:
        """
        Extract latent representations from data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (N, H, W)
        
        Returns
        -------
        latents : np.ndarray
            Latent representations (N, latent_dim)
        """
        self.model.eval()
        
        # Add channel dimension
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]
        
        latents = []
        
        with torch.no_grad():
            for i in range(0, len(data), self.config['training']['batch_size']):
                batch = data[i:i + self.config['training']['batch_size']]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                latent = self.model.encode(batch_tensor)
                latents.append(latent.cpu().numpy())
        
        return np.vstack(latents)


def train_lisa_autoencoder(config_path: str) -> Tuple[LISAAutoencoderTrainer, Dict[str, Any]]:
    """
    Convenience function to train LISA autoencoder.
    
    Parameters
    ----------
    config_path : str
        Path to training configuration YAML
    
    Returns
    -------
    trainer : LISAAutoencoderTrainer
        Trained trainer instance
    results : Dict
        Training results
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer
    trainer = LISAAutoencoderTrainer(config_path)
    
    # Load and preprocess data
    train_cwt, test_cwt = trainer.load_and_preprocess_data()
    
    # Setup model and data
    trainer.setup_model()
    trainer.setup_data_loaders(train_cwt)
    
    # Train
    results = trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
    
    return trainer, results

