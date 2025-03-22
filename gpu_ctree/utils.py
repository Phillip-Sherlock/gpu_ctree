"""
Utility functions for the GPU-CTree package.

This module provides utility functions for GPU acceleration,
data preprocessing, and other helper functions.
"""

import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder

def check_cuda_availability():
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns
    -------
    bool
        True if CUDA is available, False otherwise
    """
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except ImportError:
        return False

def to_gpu(array):
    """
    Transfer a NumPy array to GPU memory.
    
    Parameters
    ----------
    array : array-like
        NumPy array to transfer to GPU
        
    Returns
    -------
    array
        CuPy array in GPU memory
    """
    try:
        import cupy as cp
        return cp.asarray(array)
    except ImportError:
        warnings.warn("CuPy not available. Cannot transfer to GPU.")
        return array

def to_cpu(array):
    """
    Transfer a CuPy array to CPU memory.
    
    Parameters
    ----------
    array : array-like
        CuPy array to transfer to CPU
        
    Returns
    -------
    array
        NumPy array in CPU memory
    """
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    except ImportError:
        return array

def clear_gpu_memory():
    """
    Clear GPU memory to prevent memory leaks.
    """
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        warnings.warn("CuPy not available. Cannot clear GPU memory.")

def get_gpu_memory_info():
    """
    Get GPU memory information.
    
    Returns
    -------
    dict
        Dictionary with GPU memory information:
        - free: free memory in bytes
        - total: total memory in bytes
        - used: used memory in bytes
        - percent_used: percentage of memory used
    """
    try:
        import cupy as cp
        mem_info = cp.cuda.runtime.memGetInfo()
        free = mem_info[0]
        total = mem_info[1]
        used = total - free
        percent_used = (used / total) * 100
        
        return {
            'free': free,
            'total': total,
            'used': used,
            'percent_used': percent_used
        }
    except ImportError:
        warnings.warn("CuPy not available. Cannot get GPU memory information.")
        return {
            'free': 0,
            'total': 0,
            'used': 0,
            'percent_used': 0
        }

def encode_categorical(X, categorical_features=None):
    """
    Encode categorical features with LabelEncoder.
    
    Parameters
    ----------
    X : array-like or pandas DataFrame
        The input data
    categorical_features : list or None
        List of categorical feature indices or names. If None,
        automatically detect object and category dtypes.
        
    Returns
    -------
    X_transformed : array or DataFrame
        Transformed data with encoded categorical features
    encoders : dict
        Dictionary of LabelEncoders for each categorical feature
    """
    import pandas as pd
    
    # Convert to DataFrame if not already
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    
    # Auto-detect categorical features if not specified
    if categorical_features is None:
        categorical_features = []
        for col in X_df.columns:
            if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                categorical_features.append(col)
    
    # Ensure categorical_features has column names
    if len(categorical_features) > 0 and isinstance(categorical_features[0], int):
        categorical_features = [X_df.columns[i] for i in categorical_features]
    
    # Create and fit encoders
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col])
        encoders[col] = le
    
    # Return DataFrame or numpy array depending on input type
    if isinstance(X, pd.DataFrame):
        return X_df, encoders
    else:
        return X_df.values, encoders

# Any additional utility functions go here