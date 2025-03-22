"""
GPU kernels for accelerated computation in the gpu_ctree package.

This module provides GPU-accelerated implementations of key statistical
operations used in conditional inference trees, including node statistics,
split criteria, and permutation tests.
"""

import numpy as np
import warnings

# Check if GPU libraries are available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# GPU kernel implementations
def gpu_compute_node_statistics(X, y, node_indices):
    """
    Compute node statistics for each feature and outcome using GPU acceleration.
    
    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)
    y : array-like
        Target values of shape (n_samples,) or (n_samples, n_outcomes)
    node_indices : array-like
        Node indices for each sample
        
    Returns
    -------
    node_statistics : array
        Node statistics for each feature and outcome
    """
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy is required for GPU acceleration. Falling back to CPU.")
        return None
        
    try:
        # Convert inputs to GPU arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        node_indices_gpu = cp.asarray(node_indices, dtype=cp.int32)
        
        # Get dimensions
        n_samples, n_features = X_gpu.shape
        if y_gpu.ndim == 1:
            y_gpu = y_gpu.reshape(-1, 1)  # Convert to 2D array
        n_outcomes = y_gpu.shape[1]
        
        # Get unique node indices
        unique_nodes = cp.unique(node_indices_gpu)
        n_nodes = len(unique_nodes)
        
        # Initialize statistics array
        node_stats = cp.zeros((n_nodes, n_features + n_outcomes), dtype=cp.float32)
        
        # Compute statistics for each node
        for i in range(n_samples):
            node_idx = node_indices_gpu[i]
            # For each node, accumulate feature values
            for j in range(n_features):
                node_stats[node_idx, j] += X_gpu[i, j]
            # And outcome values
            for k in range(n_outcomes):
                node_stats[node_idx, n_features + k] += y_gpu[i, k]
                
        # Convert back to CPU and return
        return cp.asnumpy(node_stats)
        
    except Exception as e:
        warnings.warn(f"GPU computation failed: {str(e)}. Falling back to CPU.")
        return None

def gpu_compute_split_criterion(X, y, node_indices, feature_idx, split_points):
    """
    Compute split criteria for a feature using GPU acceleration.
    
    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)
    y : array-like
        Target values of shape (n_samples,) or (n_samples, n_outcomes)
    node_indices : array-like
        Node indices for each sample (1 for samples in current node, 0 otherwise)
    feature_idx : int
        Index of the feature to compute split criteria for
    split_points : array-like
        Potential split points for the feature
        
    Returns
    -------
    criterion_values : array
        Criterion values for each split point
    """
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy is required for GPU acceleration. Falling back to CPU.")
        return None
        
    try:
        # Convert inputs to GPU arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        node_indices_gpu = cp.asarray(node_indices, dtype=cp.int32)
        split_points_gpu = cp.asarray(split_points, dtype=cp.float32)
        
        # Get dimensions
        n_samples = X_gpu.shape[0]
        n_splits = len(split_points_gpu)
        
        # Initialize criterion values
        criterion_values = cp.zeros(n_splits, dtype=cp.float32)
        
        # Compute criterion for each split point
        for s, split_val in enumerate(split_points_gpu):
            left_sum = cp.float32(0.0)
            right_sum = cp.float32(0.0)
            left_count = 0
            right_count = 0
            
            # Sort samples into left and right based on split
            for i in range(n_samples):
                if node_indices_gpu[i] == 1:  # Only consider samples in this node
                    if X_gpu[i, feature_idx] <= split_val:
                        left_sum += y_gpu[i]
                        left_count += 1
                    else:
                        right_sum += y_gpu[i]
                        right_count += 1
            
            # Compute criterion (variance reduction)
            if left_count > 0 and right_count > 0:
                left_mean = left_sum / left_count
                right_mean = right_sum / right_count
                total_mean = (left_sum + right_sum) / (left_count + right_count)
                
                # Weighted variance reduction
                left_weight = float(left_count) / (left_count + right_count)
                right_weight = float(right_count) / (left_count + right_count)
                
                criterion = total_mean * total_mean - (
                    left_weight * left_mean * left_mean + 
                    right_weight * right_mean * right_mean
                )
                criterion_values[s] = criterion
        
        # Convert back to CPU and return
        return cp.asnumpy(criterion_values)
        
    except Exception as e:
        warnings.warn(f"GPU computation failed: {str(e)}. Falling back to CPU.")
        return None

def gpu_permutation_test(X, y, n_permutations=1000, random_state=None):
    """
    Perform permutation test using GPU acceleration to compute feature importance.
    
    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)
    y : array-like
        Target values of shape (n_samples,) or (n_samples, n_outcomes)
    n_permutations : int, default=1000
        Number of permutations to perform
    random_state : int or None, default=None
        Random state for reproducibility
        
    Returns
    -------
    p_values : array
        P-values for each feature
    test_statistics : array
        Test statistics for each feature
    """
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy is required for GPU acceleration. Falling back to CPU.")
        return None, None
        
    try:
        # Set random seed if provided
        if random_state is not None:
            cp.random.seed(random_state)
        
        # Convert inputs to GPU arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        
        # Get dimensions
        n_samples, n_features = X_gpu.shape
        if y_gpu.ndim == 1:
            y_gpu = y_gpu.reshape(-1, 1)
        n_outcomes = y_gpu.shape[1]
        
        # Compute original test statistics (correlation)
        orig_test_stats = cp.zeros(n_features, dtype=cp.float32)
        for j in range(n_features):
            if n_outcomes == 1:
                # For single outcome, use absolute correlation
                orig_test_stats[j] = cp.abs(cp.corrcoef(X_gpu[:, j], y_gpu[:, 0])[0, 1])
            else:
                # For multiple outcomes, use sum of absolute correlations
                corrs = cp.abs(cp.corrcoef(X_gpu[:, j], y_gpu.T)[0, 1:n_outcomes+1])
                orig_test_stats[j] = cp.sum(corrs)
        
        # Initialize permutation test statistics
        perm_test_stats = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        
        # Run permutation test
        for i in range(n_permutations):
            # Generate permutation indices
            perm_idx = cp.random.permutation(n_samples)
            y_perm = y_gpu[perm_idx]
            
            # Compute test statistics for permuted data
            for j in range(n_features):
                if n_outcomes == 1:
                    perm_test_stats[i, j] = cp.abs(cp.corrcoef(X_gpu[:, j], y_perm[:, 0])[0, 1])
                else:
                    corrs = cp.abs(cp.corrcoef(X_gpu[:, j], y_perm.T)[0, 1:n_outcomes+1])
                    perm_test_stats[i, j] = cp.sum(corrs)
        
        # Compute p-values
        p_values = cp.zeros(n_features, dtype=cp.float32)
        for j in range(n_features):
            p_values[j] = cp.sum(perm_test_stats[:, j] >= orig_test_stats[j]) / n_permutations
        
        # Convert back to CPU and return
        return cp.asnumpy(p_values), cp.asnumpy(orig_test_stats)
        
    except Exception as e:
        warnings.warn(f"GPU computation failed: {str(e)}. Falling back to CPU.")
        return None, None
