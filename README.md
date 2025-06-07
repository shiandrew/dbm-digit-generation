# Deep Boltzmann Machine for Digit Generation

A PyTorch implementation of Deep Boltzmann Machines (DBM) for digit generation, focusing on MNIST and other digit datasets. This project provides a flexible framework for training, sampling, and evaluating DBMs with various architectures and configurations.

## Features

- Implementation of Deep Boltzmann Machines (DBM)
- Support for various architectures (standard, conditional, convolutional)
- Efficient Gibbs sampling for sample generation
- Comprehensive evaluation metrics
- Flexible configuration system
- Training visualization and monitoring
- Extensive test coverage
- Jupyter notebooks for exploration and analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dbm-digit-generation.git
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

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training

Train a basic DBM on MNIST:
```bash
python scripts/train.py --config configs/mnist_config.yaml
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

## Project Structure

```
dbm_digit_generation/
│
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   ├── data/                    # Data handling
│   ├── training/                # Training utilities
│   ├── sampling/                # Sampling and generation
│   ├── evaluation/              # Model evaluation
│   └── utils/                   # Utility functions
│
├── configs/                     # Configuration files
├── scripts/                     # Executable scripts
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── data/                        # Data directory
├── models/                      # Saved models
├── results/                     # Results and outputs
└── docs/                        # Documentation
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- isort for import sorting
- mypy for type checking

Run all checks:
```bash
black src tests
flake8 src tests
isort src tests
mypy src
```

### Jupyter Notebooks

Start Jupyter:
```bash
jupyter notebook notebooks/
```

## Configuration

The project uses YAML configuration files for managing hyperparameters and experiment settings. Key configuration files:

- `configs/default.yaml`: Default settings
- `configs/mnist_config.yaml`: MNIST-specific configuration
- `configs/experiments/`: Experiment-specific configurations

## Documentation

- [API Reference](docs/api_reference.md)
- [User Guide](docs/user_guide.md)
- [Theoretical Background](docs/theory_background.md)
- [Tutorials](docs/tutorials/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dbm_digit_generation,
  author = {Your Name},
  title = {Deep Boltzmann Machine for Digit Generation},
  year = {2024},
  url = {https://github.com/yourusername/dbm-digit-generation}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- MNIST dataset creators
- The deep learning community for valuable insights and research 