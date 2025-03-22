#!/bin/bash
#SBATCH --job-name=gpu_ctree_boston
#SBATCH --output=gpu_ctree_boston_%j.out
#SBATCH --error=gpu_ctree_boston_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $SLURM_JOB_GPUS"

# Load necessary modules based on what's available
module load python
module load cuda

# Check CUDA availability
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set CUDA visible devices to use all 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate the environment
source /blue/hknopf/Phillip.sherlock/gpu_ctree/gpu_ctree_env/bin/activate

# Check if GPU is available through Python
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'GPU devices: {cp.cuda.runtime.getDeviceCount()}'); print(f'Current device: {cp.cuda.runtime.getDevice()}')" || echo "Could not initialize CuPy"

# Run the Boston Housing example
python boston_housing_example.py

echo "Job completed at: $(date)"
