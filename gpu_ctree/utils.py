"""
Utility functions for GPU-CTree package.
"""

import os
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns
    -------
    bool
        True if CUDA is available, False otherwise.
    """
    try:
        import cupy
        return cupy.cuda.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def encode_categorical(X, feature_types=None):
    """
    Encode categorical variables as integers.
    
    Parameters
    ----------
    X : array-like
        Input data matrix.
    feature_types : list of str, optional
        List of feature types ('numeric' or 'categorical').
        If None, infers from data.
        
    Returns
    -------
    X_encoded : array
        Encoded input matrix.
    encoders : dict
        Dictionary mapping feature indices to their encoders.
    feature_types : list
        List of feature types used.
    """
    import pandas as pd
    
    # Convert to pandas DataFrame for easier handling
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    n_features = X.shape[1]
    
    # Infer feature types if not provided
    if feature_types is None:
        feature_types = []
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col]) or pd.api.types.is_object_dtype(X[col]):
                feature_types.append('categorical')
            else:
                feature_types.append('numeric')
    
    # Initialize encoders dictionary
    encoders = {}
    X_encoded = X.copy()
    
    # Encode categorical features
    for i, (col, feat_type) in enumerate(zip(X.columns, feature_types)):
        if feat_type == 'categorical':
            # Create a mapping dictionary
            unique_vals = sorted(X[col].unique())
            encoder = {val: idx for idx, val in enumerate(unique_vals)}
            encoders[i] = encoder
            
            # Apply encoding
            X_encoded[col] = X[col].map(encoder)
    
    return X_encoded.values, encoders, feature_types


def parse_formula(formula: str) -> Tuple[str, List[str]]:
    """
    Parse an R-style formula into outcome and predictor variables.
    
    Parameters
    ----------
    formula : str
        R-style formula (e.g., "y ~ x1 + x2").
        
    Returns
    -------
    tuple
        (outcome_var, predictor_vars)
    """
    import re
    
    # Strip whitespace and check format
    formula = formula.strip()
    match = re.match(r"(.+?)\s*~\s*(.+)$", formula)
    if not match:
        raise ValueError(f"Invalid formula format: {formula}")
    
    outcome = match.group(1).strip()
    predictors_str = match.group(2).strip()
    
    # Handle special case '.' for all other variables
    if predictors_str == '.':
        return outcome, ['.']
    
    # Parse predictors
    predictors = [p.strip() for p in re.split(r'\s*\+\s*', predictors_str)]
    
    return outcome, predictors


def check_environment():
    """
    Check and report the computing environment.
    
    Returns
    -------
    dict
        Dictionary with environment information.
    """
    env_info = {
        'gpu_available': check_cuda_availability(),
        'python_version': None,
        'numpy_version': None,
        'pandas_version': None,
        'sklearn_version': None,
        'cuda_version': None,
        'cupy_version': None
    }
    
    import sys
    env_info['python_version'] = sys.version
    
    try:
        import numpy
        env_info['numpy_version'] = numpy.__version__
    except ImportError:
        pass
    
    try:
        import pandas
        env_info['pandas_version'] = pandas.__version__
    except ImportError:
        pass
    
    try:
        import sklearn
        env_info['sklearn_version'] = sklearn.__version__
    except ImportError:
        pass
    
    if env_info['gpu_available']:
        try:
            import cupy
            env_info['cupy_version'] = cupy.__version__
            
            # Get CUDA version
            cuda_version = cupy.cuda.runtime.runtimeGetVersion()
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            env_info['cuda_version'] = f"{major}.{minor}"
        except Exception:
            pass
    
    return env_info
