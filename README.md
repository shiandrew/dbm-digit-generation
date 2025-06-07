# Deep Boltzmann Machine for Digit Generation

A PyTorch implementation of Deep Boltzmann Machines (DBM) for digit generation, focusing on MNIST and Fashion-MNIST datasets. This project provides a streamlined framework for training and evaluating DBMs with essential components.

## Features

- Core DBM implementation with RBM building blocks
- Efficient Gibbs sampling for sample generation
- Basic evaluation metrics
- Simple configuration system
- Essential visualization tools
- Focused test coverage

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shiandrew/dbm-digit-generation.git
cd dbm-digit-generation
```

2. Create and activate a virtual environment:

For Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
dbm_digit_generation/
│
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   │   ├── rbm.py              # Restricted Boltzmann Machine
│   │   ├── dbm.py              # Deep Boltzmann Machine
│   │   └── conditional_dbm.py   # Optional: Conditional DBM
│   │
│   ├── data/                    # Data handling
│   │   └── loaders.py          # Dataset loaders
│   │
│   ├── training/                # Training utilities
│   │   └── trainer.py          # Training loop
│   │
│   ├── sampling/                # Sampling
│   │   └── gibbs_sampler.py    # Gibbs sampling
│   │
│   ├── evaluation/              # Evaluation
│   │   └── metrics.py          # Basic metrics
│   │
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration
│       └── visualization.py    # Basic plotting
│
├── configs/                     # Configuration files
│   ├── default.yaml            # Default settings
│   └── fashion_mnist_config.yaml
│
├── scripts/                     # Executable scripts
│   ├── train.py                # Training script
│   ├── generate.py             # Generation script
│   └── evaluate.py             # Evaluation script
│
├── data/                        # Data directory
│   ├── raw/                    # Raw datasets
│   └── processed/              # Processed data
│
├── models/                      # Saved models
│   └── checkpoints/            # Training checkpoints
│
└── results/                     # Results
    ├── figures/                # Generated plots
    └── samples/                # Generated samples
```

## Usage

### Training

Train a DBM on MNIST:
```bash
python scripts/train.py --config configs/default.yaml
```

Train on Fashion-MNIST:
```bash
python scripts/train.py --config configs/fashion_mnist_config.yaml
```

### Generation

Generate samples from a trained model:
```bash
python scripts/generate.py --model_path models/checkpoints/best_model.pkl
```

### Evaluation

Evaluate model performance:
```bash
python scripts/evaluate.py --model_path models/checkpoints/best_model.pkl
```

## Development

Run tests:
```bash
python -m pytest tests/
```

## Configuration

The project uses YAML configuration files for managing hyperparameters. Key files:

- `configs/default.yaml`: Default settings for MNIST
- `configs/fashion_mnist_config.yaml`: Settings for Fashion-MNIST

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- MNIST and Fashion-MNIST dataset creators 