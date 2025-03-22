#!/bin/bash

# Load necessary modules
module load python/3.9
module load cuda/11.6

# Set the path to the gpu_ctree installation
GPU_CTREE_PATH="/blue/hknopf/Phillip.sherlock/gpu_ctree"

# Activate the virtual environment
source ${GPU_CTREE_PATH}/gpu_ctree_env/bin/activate

# Run the provided script or command
if [ $# -eq 0 ]; then
    echo "Usage: ./use_gpu_ctree.sh your_script.py"
    echo "Or start Python: ./use_gpu_ctree.sh python"
else
    "$@"
fi
