# Deep Boltzmann Machine for Digit Generation

A PyTorch implementation of Deep Boltzmann Machines (DBM) for digit generation, focusing on MNIST and Fashion-MNIST datasets. This project provides a complete framework for training and evaluating DBMs with proper mean-field inference and advanced sampling techniques.

## Features

- Complete DBM implementation with proper mean-field inference
- Restricted Boltzmann Machine (RBM) base class
- Advanced Gibbs sampling with temperature control and simulated annealing
- Multiple training configurations (baseline, improved, memory-optimized)
- Comprehensive evaluation metrics and visualization tools

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/shiandrew/dbm-digit-generation.git
cd dbm-digit-generation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train with different configurations:

```bash
# Baseline model (500x500 hidden layers)
python scripts/train.py --config configs/01_baseline_config.yaml

# Improved architecture (300x200 with pretraining)
python scripts/train.py --config configs/02_improved_architecture.yaml

# Memory-optimized (256x128 for limited GPU RAM)
python scripts/train.py --config configs/03_memory_optimized_final.yaml
```

### Generation

Generate samples from trained models:

```bash
# Basic generation
python scripts/generate.py --model_path models/02_improved_attempt_300x200/best_model.pt --num_samples 16

# High-quality generation with annealing
python scripts/generate.py --model_path models/02_improved_attempt_300x200/best_model.pt --num_samples 16 --use_annealing --num_steps 5000

# Temperature-controlled sampling
python scripts/generate.py --model_path models/02_improved_attempt_300x200/best_model.pt --temperature 0.8
```

### Evaluation

```bash
python scripts/evaluate.py --model_path models/02_improved_attempt_300x200/best_model.pt --dataset mnist
```

### Testing

```bash
python tests/test_dbm_complete.py
```

### Monitoring

```bash
tensorboard --logdir results/
```

## Configuration

The project uses YAML configuration files:

- `configs/01_baseline_config.yaml`: Standard DBM with 500x500 hidden layers
- `configs/02_improved_architecture.yaml`: Optimized architecture with pretraining
- `configs/03_memory_optimized_final.yaml`: Memory-efficient for limited GPU RAM

## License

This project is licensed under the MIT License - see the LICENSE file for details.