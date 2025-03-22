"""
CUDA utilities for GPU acceleration in the gpu_ctree package.
"""

# Check if required dependencies are available
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import functions if dependencies are available
if CUPY_AVAILABLE:
    from .kernels import (
        gpu_permutation_test,
        gpu_compute_split_criterion,
        gpu_compute_node_statistics
    )
    from .utils import (
        to_gpu,
        to_cpu,
        clear_gpu_memory,
        get_gpu_memory_info
    )
    
    __all__ = [
        'gpu_permutation_test',
        'gpu_compute_split_criterion',
        'gpu_compute_node_statistics',
        'to_gpu',
        'to_cpu',
        'clear_gpu_memory',
        'get_gpu_memory_info',
        'CUPY_AVAILABLE'
    ]
else:
    # Define the flag so other modules can check
    __all__ = ['CUPY_AVAILABLE']