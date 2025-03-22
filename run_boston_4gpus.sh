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

# Load necessary modules - check which Python and CUDA are available
module spider python
module spider cuda
# Load appropriate modules based on what's available
module load python
module load cuda

# Check CUDA availability
nvidia-smi

# Make sure the GPU environment is set up
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Checking for CUDA with Python:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "Torch not available"
python -c "import cupy; print(f'CuPy version: {cupy.__version__}')" 2>/dev/null || echo "CuPy not available"

# Activate the environment
source /blue/hknopf/Phillip.sherlock/gpu_ctree/gpu_ctree_env/bin/activate

# Install/update required GPU packages if needed
pip install cupy-cuda11x  # Adjust version as needed
pip install numba

# Run the Boston Housing example
python boston_housing_example.py

echo "Job completed at: $(date)"
