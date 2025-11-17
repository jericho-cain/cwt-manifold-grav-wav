"""
Tests for LISA autoencoder models.

Tests the CWT autoencoder architecture adapted for LISA data.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from src.models.cwtlstm import (
    CWTAutoencoder,
    SimpleCWTAutoencoder,
    create_model,
    save_model,
    load_model,
)


class TestCWTAutoencoder:
    """Test CWT autoencoder."""
    
    def test_model_creation_lisa_dimensions(self):
        """Test creating model with LISA dimensions."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        assert model.input_height == 64
        assert model.input_width == 3600
        assert model.latent_dim == 32
    
    def test_forward_pass_lisa(self):
        """Test forward pass with LISA-sized input."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        # LISA CWT input
        x = torch.randn(2, 1, 64, 3600)
        
        reconstructed, latent = model(x)
        
        # Check shapes
        assert reconstructed.shape == (2, 1, 64, 3600)
        assert latent.shape == (2, 32)
    
    def test_encode_decode_lisa(self):
        """Test encode and decode separately."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        x = torch.randn(2, 1, 64, 3600)
        
        # Encode
        latent = model.encode(x)
        assert latent.shape == (2, 32)
        
        # Decode
        reconstructed = model.decode(latent)
        assert reconstructed.shape == (2, 1, 64, 3600)
    
    def test_different_dimensions(self):
        """Test model works with different dimensions."""
        # Test with various LISA-like dimensions
        dimensions = [
            (64, 3600),  # Standard LISA
            (32, 1800),  # Smaller
            (128, 7200), # Larger
        ]
        
        for height, width in dimensions:
            model = CWTAutoencoder(
                input_height=height,
                input_width=width,
                latent_dim=32
            )
            
            x = torch.randn(1, 1, height, width)
            reconstructed, latent = model(x)
            
            assert reconstructed.shape == (1, 1, height, width)
            assert latent.shape == (1, 32)
    
    def test_model_info(self):
        """Test get_model_info returns correct information."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        info = model.get_model_info()
        
        assert info['input_height'] == 64
        assert info['input_width'] == 3600
        assert info['latent_dim'] == 32
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert 'LISA' in info['architecture']
    
    def test_batch_processing(self):
        """Test model handles different batch sizes."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 1, 64, 3600)
            reconstructed, latent = model(x)
            
            assert reconstructed.shape[0] == batch_size
            assert latent.shape[0] == batch_size


class TestSimpleCWTAutoencoder:
    """Test simplified CWT autoencoder."""
    
    def test_simple_model_creation(self):
        """Test creating simple model."""
        model = SimpleCWTAutoencoder(
            height=64,
            width=3600,
            latent_dim=64
        )
        
        assert model.height == 64
        assert model.width == 3600
    
    def test_simple_forward_pass(self):
        """Test forward pass with simple model."""
        model = SimpleCWTAutoencoder(
            height=64,
            width=3600,
            latent_dim=64
        )
        
        x = torch.randn(2, 1, 64, 3600)
        reconstructed, latent = model(x)
        
        assert reconstructed.shape == (2, 1, 64, 3600)
        assert latent.shape == (2, 64)


class TestModelFactory:
    """Test model factory function."""
    
    def test_create_cwt_ae_model(self):
        """Test creating CWT autoencoder model via factory."""
        model = create_model(
            'cwt_ae',
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        assert isinstance(model, CWTAutoencoder)
        assert model.input_height == 64
        assert model.input_width == 3600
    
    def test_create_simple_model(self):
        """Test creating simple model via factory."""
        model = create_model(
            'simple_cwt',
            input_height=64,
            input_width=3600,
            latent_dim=64
        )
        
        assert isinstance(model, SimpleCWTAutoencoder)
        assert model.height == 64
        assert model.width == 3600
    
    def test_invalid_model_type(self):
        """Test factory raises error for invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model(
                'invalid_model',
                input_height=64,
                input_width=3600
            )


class TestModelSaveLoad:
    """Test model saving and loading."""
    
    def test_save_load_model(self):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            model = CWTAutoencoder(
                input_height=64,
                input_width=3600,
                latent_dim=32
            )
            
            save_path = Path(tmpdir) / 'test_model.pth'
            metadata = {'test': 'metadata'}
            
            save_model(model, save_path, metadata)
            
            # Check file exists
            assert save_path.exists()
            
            # Load model
            loaded_model, loaded_metadata = load_model(
                save_path,
                CWTAutoencoder,
                latent_dim=32
            )
            
            # Check loaded correctly
            assert loaded_model.input_height == 64
            assert loaded_model.input_width == 3600
            assert loaded_model.latent_dim == 32
            assert loaded_metadata['test'] == 'metadata'
    
    def test_save_load_preserves_weights(self):
        """Test that saving and loading preserves model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model
            model = CWTAutoencoder(
                input_height=64,
                input_width=3600,
                latent_dim=32
            )
            
            # Get initial output
            x = torch.randn(1, 1, 64, 3600)
            with torch.no_grad():
                initial_output, _ = model(x)
            
            # Save and load
            save_path = Path(tmpdir) / 'test_model.pth'
            save_model(model, save_path)
            loaded_model, _ = load_model(save_path, CWTAutoencoder, latent_dim=32)
            
            # Get output from loaded model
            with torch.no_grad():
                loaded_output, _ = loaded_model(x)
            
            # Outputs should be identical
            assert torch.allclose(initial_output, loaded_output, rtol=1e-5)
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            load_model(
                Path('/nonexistent/path/model.pth'),
                CWTAutoencoder
            )


class TestModelGradients:
    """Test model gradients and training."""
    
    def test_model_has_gradients(self):
        """Test model parameters have gradients."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        x = torch.randn(2, 1, 64, 3600, requires_grad=True)
        reconstructed, latent = model(x)
        
        # Compute loss
        loss = torch.mean((reconstructed - x) ** 2)
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        x = torch.randn(2, 1, 64, 3600)
        
        # Forward pass
        reconstructed, _ = model(x)
        loss = criterion(reconstructed, x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss should be a scalar
        assert loss.item() > 0


class TestModelMemory:
    """Test model memory efficiency."""
    
    def test_model_size_reasonable(self):
        """Test model size is reasonable for LISA data."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        info = model.get_model_info()
        
        # Model should be < 100 MB (reasonable for training)
        assert info['model_size_mb'] < 100
    
    def test_forward_pass_memory(self):
        """Test forward pass doesn't explode memory."""
        model = CWTAutoencoder(
            input_height=64,
            input_width=3600,
            latent_dim=32
        )
        
        # Should handle batch without memory error
        x = torch.randn(4, 1, 64, 3600)
        
        # This should not raise MemoryError
        reconstructed, latent = model(x)
        
        assert reconstructed.shape == x.shape

