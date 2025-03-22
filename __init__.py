"""
GPU-CTree: GPU-Accelerated Conditional Inference Trees with Look-Ahead

This package implements a GPU-accelerated version of conditional inference trees
enhanced with look-ahead capabilities for more globally optimal tree structures,
multivariate outcome support, and comprehensive statistical significance testing.
"""

__version__ = '0.1.0'

# First, ensure gpu_ctree.utils is importable without circular imports
from gpu_ctree.utils import check_cuda_availability, to_gpu, to_cpu, clear_gpu_memory, get_gpu_memory_info

# Then import the core components
from gpu_ctree.core import GPUCTree
from gpu_ctree.controls import gpu_ctree_control, GPUCTreeControls
from gpu_ctree.r_interface import gpu_ctree, predict_gpu_ctree

# Make these available at the package level
__all__ = [
    'GPUCTree',
    'gpu_ctree',
    'predict_gpu_ctree',
    'GPUCTreeControls',
    'gpu_ctree_control',
    'check_cuda_availability',
    'to_gpu',
    'to_cpu',
    'clear_gpu_memory',
    'get_gpu_memory_info'
]