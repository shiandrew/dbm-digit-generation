# DBM Digit Generation

A Deep Boltzmann Machine (DBM) implementation for digit generation, focusing on MNIST and Fashion-MNIST datasets.

## Overview

This project implements various DBM architectures for generating and manipulating digit images. It includes:
- Core DBM implementation
- Conditional DBM variant
- Convolutional DBM variant
- Training utilities with contrastive divergence
- Sampling and generation capabilities
- Comprehensive evaluation metrics

## Features

- Multiple DBM architectures (Standard, Conditional, Convolutional)
- Efficient training with contrastive divergence
- Gibbs sampling for sample generation
- Latent space interpolation
- Comprehensive evaluation metrics
- Visualization tools
- Experiment tracking and logging

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
```

## Project Structure

```
dbm_digit_generation/
├── src/                    # Source code
├── configs/               # Configuration files
├── scripts/              # Executable scripts
├── notebooks/           # Jupyter notebooks
├── tests/              # Unit tests
├── data/              # Data directory
├── models/           # Saved models
├── results/         # Results and outputs
└── docs/           # Documentation
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train a model:
```bash
python scripts/train.py --config configs/mnist_config.yaml
```

3. Generate samples:
```bash
python scripts/generate.py --model_path models/checkpoints/latest.pt
```

## Documentation

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Theoretical Background](docs/theory_background.md)
- [Tutorials](docs/tutorials/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 