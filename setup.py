from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

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
    with open(init_file, 'w') as f:
        f.write('# This file makes Python treat the directory as a package\n')

setup(
    name="dbm-digit-generation",
    version="0.1.0",
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires=read_requirements(),
    python_requires='>=3.8',
    
    # Project metadata
    author="Andrew Shi",
    author_email="shian@uci.edu",
    description="A Deep Boltzmann Machine implementation for digit generation",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    url="https://github.com/shiandrew/dbm-digit-generation",
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    
    # Entry points for command line scripts
    entry_points={
        'console_scripts': [
            'dbm-train=scripts.train:main',
            'dbm-generate=scripts.generate:main',
            'dbm-evaluate=scripts.evaluate:main',
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        '': ['*.yml', '*.yaml', '*.txt', '*.md'],
    },
)