"""
GPU-CTree: GPU-Accelerated Conditional Inference Trees with Look-Ahead

This package implements a GPU-accelerated version of conditional inference trees
enhanced with look-ahead capabilities for more globally optimal tree structures,
multivariate outcome support, and comprehensive statistical significance testing.
"""

__version__ = '0.1.0'

# Import main components for easier access
from .core import GPUCTree
from .r_interface import gpu_ctree, predict_gpu_ctree
from .controls import GPUCTreeControls, gpu_ctree_control
from .utils import check_cuda_availability

# Make these available at the package level
__all__ = [
    'GPUCTree',
    'gpu_ctree',
    'predict_gpu_ctree',
    'GPUCTreeControls',
    'gpu_ctree_control',
    'check_cuda_availability'
]
