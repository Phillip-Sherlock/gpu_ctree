"""
CUDA utilities for GPU acceleration in the gpu_ctree package.

This module provides GPU-accelerated implementations of key statistical
operations used in conditional inference trees.
"""

# Check if required dependencies are available
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import functions if dependencies are available
if CUPY_AVAILABLE:
    try:
        from .kernels import (
            gpu_permutation_test,
            gpu_compute_split_criterion, 
            gpu_compute_node_statistics
        )
        from .utils import (
            to_gpu,
            to_cpu,
            clear_gpu_memory,
            get_gpu_memory_info,
            check_cuda_availability
        )
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import GPU functions: {str(e)}")
        # Define dummy functions that return None when GPU acceleration fails
        def gpu_permutation_test(*args, **kwargs):
            import warnings
            warnings.warn("GPU acceleration unavailable. Falling back to CPU.")
            return None, None
            
        def gpu_compute_split_criterion(*args, **kwargs):
            import warnings
            warnings.warn("GPU acceleration unavailable. Falling back to CPU.")
            return None
            
        def gpu_compute_node_statistics(*args, **kwargs):
            import warnings
            warnings.warn("GPU acceleration unavailable. Falling back to CPU.")
            return None
    
    __all__ = [
        'gpu_permutation_test',
        'gpu_compute_split_criterion',
        'gpu_compute_node_statistics',
        'to_gpu',
        'to_cpu',
        'clear_gpu_memory',
        'get_gpu_memory_info',
        'check_cuda_availability',
        'CUPY_AVAILABLE'
    ]
else:
    # Export the check function regardless of CuPy availability
    from .utils import check_cuda_availability
    
    __all__ = ['check_cuda_availability', 'CUPY_AVAILABLE']
    
    import warnings
    warnings.warn("CuPy not available. GPU acceleration will be disabled.")
