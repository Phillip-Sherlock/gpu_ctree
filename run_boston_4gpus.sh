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

# Set the PYTHONPATH to include the project root
export PYTHONPATH="/blue/hknopf/Phillip.sherlock/gpu_ctree:$PYTHONPATH"

# Check if GPU is available through Python
echo "Testing GPU availability with CuPy:"
python -c "
try:
    import cupy as cp
    print(f'CuPy version: {cp.__version__}')
    print(f'GPU devices: {cp.cuda.runtime.getDeviceCount()}')
    print(f'Current device: {cp.cuda.runtime.getDevice()}')
except ImportError as e:
    print(f'Could not import CuPy: {e}')
except Exception as e:
    print(f'CuPy error: {e}')
"

# Test imports directly
echo "Testing imports:"
python -c "
try:
    from gpu_ctree.utils import check_cuda_availability
    print('✓ Successfully imported check_cuda_availability')
    
    from gpu_ctree import gpu_ctree_control
    print('✓ Successfully imported gpu_ctree_control')
    
    from gpu_ctree import GPUCTree
    print('✓ Successfully imported GPUCTree')
    
    # Now test the top-level import
    import gpu_ctree
    print(f'✓ Imported gpu_ctree package: {gpu_ctree.__version__}')
except ImportError as e:
    print(f'✗ Import error: {e}')
except Exception as e:
    print(f'✗ Other error: {e}')
"

# Run the Boston Housing example
echo "Running the Boston Housing example..."
python boston_housing_example.py

echo "Job completed at: $(date)"