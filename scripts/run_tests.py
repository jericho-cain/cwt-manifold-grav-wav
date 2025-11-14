#!/usr/bin/env python
"""
Run test suite with various options.

Usage:
    python scripts/run_tests.py              # Run all tests
    python scripts/run_tests.py --fast       # Skip slow tests
    python scripts/run_tests.py --module noise  # Test specific module
    python scripts/run_tests.py --coverage   # Run with coverage
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(args):
    """Run pytest with specified options."""
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
        ])
    
    # Skip slow tests if requested
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Run specific test module if requested
    if args.module:
        test_file = f"tests/test_{args.module}.py"
        if not Path(test_file).exists():
            print(f"Error: Test file {test_file} not found")
            print(f"Available modules: lisa_noise, lisa_waveforms, dataset_generator")
            return 1
        cmd.append(test_file)
    
    # Run specific test if requested
    if args.test:
        cmd.extend(["-k", args.test])
    
    # Add any extra args
    if args.extra:
        cmd.extend(args.extra)
    
    print("Running:", " ".join(cmd))
    print("-" * 60)
    
    # Run pytest
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run LISA manifold test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-m", "--module",
        type=str,
        help="Test specific module (e.g., 'lisa_noise', 'lisa_waveforms', 'dataset_generator')",
    )
    
    parser.add_argument(
        "-k", "--test",
        type=str,
        help="Run tests matching pattern (pytest -k)",
    )
    
    parser.add_argument(
        "-f", "--fast",
        action="store_true",
        help="Skip slow tests",
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run with coverage analysis",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    parser.add_argument(
        "extra",
        nargs="*",
        help="Additional pytest arguments",
    )
    
    args = parser.parse_args()
    
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())

