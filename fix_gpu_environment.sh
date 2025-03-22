#!/bin/bash

# This script fixes the CuPy installation to match the CUDA version

# Load required modules
echo "Loading Python and CUDA modules..."
module load python
module load cuda

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | cut -d ',' -f 1)
echo "Detected CUDA version: $CUDA_VERSION"

# Activate the environment
echo "Activating gpu_ctree environment..."
source /blue/hknopf/Phillip.sherlock/gpu_ctree/gpu_ctree_env/bin/activate

# Remove existing CuPy installations
echo "Removing existing CuPy installations..."
pip uninstall -y cupy-cuda11x cupy-cuda116 cupy

# Install appropriate CuPy version
if [[ $CUDA_VERSION == *"11."* ]]; then
    echo "Installing CuPy for CUDA 11.x..."
    pip install cupy-cuda11x
elif [[ $CUDA_VERSION == *"12."* ]]; then
    echo "Installing CuPy for CUDA 12.x..."
    pip install cupy-cuda12x
else
    echo "Unknown CUDA version. Installing latest CuPy..."
    pip install cupy
fi

# Install other required GPU packages
echo "Installing other required GPU packages..."
pip install numba

echo "Environment setup complete!"