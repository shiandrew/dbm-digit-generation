from setuptools import setup, find_packages
import os

# Create necessary directories
directories = [
    'src',
    'src/models',
    'src/data',
    'src/training',
    'src/sampling',
    'src/evaluation',
    'src/utils',
    'configs',
    'configs/experiments',
    'scripts',
    'notebooks',
    'tests',
    'tests/test_models',
    'tests/test_data',
    'tests/test_utils',
    'data/raw',
    'data/processed',
    'data/external',
    'models/checkpoints',
    'models/pretrained',
    'models/experiments',
    'results/figures',
    'results/samples',
    'results/logs',
    'results/reports',
    'docs',
    'docs/tutorials'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Create __init__.py files
init_files = [
    'src/__init__.py',
    'src/models/__init__.py',
    'src/data/__init__.py',
    'src/training/__init__.py',
    'src/sampling/__init__.py',
    'src/evaluation/__init__.py',
    'src/utils/__init__.py',
    'tests/__init__.py'
]

for init_file in init_files:
    with open(init_file, 'a') as f:
        pass

setup(
    name="dbm_digit_generation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'scikit-learn>=0.24.2',
        'matplotlib>=3.4.3',
        'seaborn>=0.11.2',
        'pandas>=1.3.3',
        'tqdm>=4.62.3',
        'PyYAML>=5.4.1',
        'tensorboard>=2.7.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Deep Boltzmann Machine implementation for digit generation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dbm-digit-generation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
) 