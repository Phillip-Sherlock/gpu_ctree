"""
GPU-CTree module for GPU-accelerated conditional inference trees.

This module provides GPU-accelerated implementations of key statistical
operations used in conditional inference trees.
"""

# Check if required dependencies are available
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import GPU functions if dependencies are available
if CUPY_AVAILABLE:
    try:
        from .kernels import (
            gpu_permutation_test,
            gpu_compute_split_criterion, 
            gpu_compute_node_statistics
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

# DO NOT import other package modules here to avoid circular imports