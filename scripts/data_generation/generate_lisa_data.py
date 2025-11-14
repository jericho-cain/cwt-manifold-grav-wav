#!/usr/bin/env python
"""
Generate LISA dataset for training and testing.

Usage:
    python scripts/data_generation/generate_lisa_data.py --config config/data_generation.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_generator import LISADatasetGenerator, DatasetConfig


def load_config(config_path: str) -> DatasetConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle nested config (extract 'data' section if present)
    if 'data' in config_dict:
        data_config = config_dict['data']
    else:
        data_config = config_dict
    
    return DatasetConfig(**data_config)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LISA dataset for anomaly detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_generation.yaml",
        help="Path to configuration YAML file",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Generate dataset
    generator = LISADatasetGenerator(config)
    generator.generate_and_save()


if __name__ == "__main__":
    main()

