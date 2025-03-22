"""
CUDA-specific utilities for GPU-CTree.

This module provides helper functions for GPU operations,
including memory management and data transfer utilities.
"""

import numpy as np
from typing import Union, Tuple, Optional, Any

# Try to import CUDA-related libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import cuda as nb_cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def get_gpu_memory_info():
    """
    Get information about GPU memory usage.
    
    Returns
    -------
    dict
        Dictionary with memory information.
    """
    if not CUPY_AVAILABLE:
        return {"error": "CuPy not available"}
    
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        device = cp.cuda.Device()
        
        info = {
            "total": device.mem_info[0],
            "free": device.mem_info[1],
            "used": device.mem_info[0] - device.mem_info[1],
            "mempool_used_bytes": mempool.used_bytes(),
            "mempool_total_bytes": mempool.total_bytes(),
            "pinned_mempool_used_bytes": pinned_mempool.used_bytes(),
            "pinned_mempool_total_bytes": pinned_mempool.total_bytes(),
        }
        
        # Calculate percentages
        info["used_percent"] = (info["used"] / info["total"]) * 100
        info["free_percent"] = (info["free"] / info["total"]) * 100
        
        return info
    except Exception as e:
        return {"error": str(e)}


def clear_gpu_memory():
    """
    Clear GPU memory caches.
    
    Returns
    -------
    bool
        True if memory was cleared successfully, False otherwise.
    """
    if not CUPY_AVAILABLE:
        return False
    
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        return True
    except Exception:
        return False


def to_gpu(arr: np.ndarray) -> 'cp.ndarray':
    """
    Transfer a numpy array to the GPU.
    
    Parameters
    ----------
    arr : ndarray
        Numpy array to transfer.
        
    Returns
    -------
    cp.ndarray
        CuPy array on the GPU.
        
    Raises
    ------
    ImportError
        If CuPy is not available.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU operations but not installed")
    
    if isinstance(arr, cp.ndarray):
        return arr
    
    return cp.asarray(arr)


def to_cpu(arr: 'cp.ndarray') -> np.ndarray:
    """
    Transfer a GPU array to the CPU.
    
    Parameters
    ----------
    arr : cp.ndarray
        CuPy array on GPU.
        
    Returns
    -------
    ndarray
        Numpy array on CPU.
    """
    if not CUPY_AVAILABLE:
        return arr  # Already a numpy array
    
    if isinstance(arr, np.ndarray):
        return arr
    
    return arr.get()


def is_gpu_array(arr: Any) -> bool:
    """
    Check if an array is a GPU array.
    
    Parameters
    ----------
    arr : Any
        Array to check.
        
    Returns
    -------
    bool
        True if the array is on GPU, False otherwise.
    """
    if not CUPY_AVAILABLE:
        return False
    
    return isinstance(arr, cp.ndarray)


def check_cuda_capacity(data_size: int, element_size: int = 4) -> bool:
    """
    Check if the GPU has enough memory for a given data size.
    
    Parameters
    ----------
    data_size : int
        Number of elements in the data.
    element_size : int, default=4
        Size of each element in bytes (default: 4 bytes for float32).
        
    Returns
    -------
    bool
        True if there's enough GPU memory, False otherwise.
    """
    if not CUPY_AVAILABLE:
        return False
    
    try:
        device = cp.cuda.Device()
        free_memory = device.mem_info[1]
        required_memory = data_size * element_size
        
        # Add a safety margin (80% of free memory)
        return required_memory <= free_memory * 0.8
    except Exception:
        return False
