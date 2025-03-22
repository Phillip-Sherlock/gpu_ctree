"""
CUDA kernels for GPU-accelerated tree operations.
"""

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple

# Try to import cupy, handle ImportError gracefully
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import numba if available for JIT compilation
try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Define optimized CUDA kernels
OPTIMIZED_PREDICT_KERNEL = """
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_TREE_DEPTH 32  // Adjust based on your tree depth

extern "C" __global__ void optimized_predict_kernel(const float* features, int* predictions, 
                                        const int* tree_structure, const float* thresholds,
                                        const int* feature_indices, const int* class_labels,
                                        int n_samples, int n_features, int max_nodes) {
    
    // Allocate shared memory for frequently accessed tree data
    __shared__ int shared_feature_indices[MAX_TREE_DEPTH];
    __shared__ float shared_thresholds[MAX_TREE_DEPTH];
    
    // Load tree data into shared memory (first threads per block)
    if (threadIdx.x < MAX_TREE_DEPTH) {
        if (threadIdx.x < max_nodes) {
            shared_feature_indices[threadIdx.x] = feature_indices[threadIdx.x];
            shared_thresholds[threadIdx.x] = thresholds[threadIdx.x];
        }
    }
    
    // Ensure shared memory is loaded before proceeding
    __syncthreads();
    
    // Calculate global thread index with grid-stride loop for handling large datasets
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_samples; i += blockDim.x * gridDim.x) {
        int node_idx = 0;  // Start at the root
        
        // Traverse tree - use loop unrolling for common depths to reduce branch overhead
        #pragma unroll 4
        while (node_idx < max_nodes && tree_structure[node_idx] != -1) {
            int feature_idx = shared_feature_indices[node_idx];
            float threshold = shared_thresholds[node_idx];
            float feature_val = features[i * n_features + feature_idx];
            
            // Use branch-free version where possible to reduce warp divergence
            bool go_left = feature_val <= threshold;
            node_idx = tree_structure[node_idx * 2 + (go_left ? 0 : 1)];
        }
        
        // Assign prediction
        predictions[i] = class_labels[node_idx];
    }
}
"""

def get_optimized_predict_kernel():
    """Get optimized predict kernel"""
    mod = SourceModule(OPTIMIZED_PREDICT_KERNEL)
    return mod.get_function("optimized_predict_kernel")


def gpu_permutation_test(X: np.ndarray, 
                         y: np.ndarray, 
                         n_permutations: int = 1000,
                         random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform permutation tests for independence between X and y.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input features.
    y : ndarray of shape (n_samples,) or (n_samples, n_outcomes)
        Target values.
    n_permutations : int, default=1000
        Number of permutations for the test.
    random_state : int, optional
        Random seed for reproducible results.
        
    Returns
    -------
    p_values : ndarray of shape (n_features,)
        P-values for each feature.
    test_stats : ndarray of shape (n_features,)
        Test statistics for each feature.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration but not installed")
    
    # Set random seed
    if random_state is not None:
        cp.random.seed(random_state)
    
    # Transfer data to GPU
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)
    
    n_samples, n_features = X_gpu.shape
    
    # Ensure y has shape (n_samples, n_outcomes)
    if y_gpu.ndim == 1:
        y_gpu = y_gpu.reshape(-1, 1)
    
    n_outcomes = y_gpu.shape[1]
    
    # Compute original test statistics
    orig_stats = compute_test_statistics(X_gpu, y_gpu)
    
    # Initialize arrays for permutation results
    perm_stats = cp.zeros((n_permutations, n_features), dtype=cp.float32)
    
    # Perform permutations
    for i in range(n_permutations):
        # Permute y values
        perm_indices = cp.random.permutation(n_samples)
        y_perm = y_gpu[perm_indices]
        
        # Compute test statistics for permuted data
        perm_stats[i] = compute_test_statistics(X_gpu, y_perm)
    
    # Compute p-values (proportion of permutation statistics >= original)
    p_values = cp.mean(perm_stats >= orig_stats, axis=0)
    
    # Transfer results back to CPU
    return p_values.get(), orig_stats.get()


def compute_test_statistics(X: 'cp.ndarray', y: 'cp.ndarray') -> 'cp.ndarray':
    """
    Compute test statistics between X and y.
    
    Parameters
    ----------
    X : cp.ndarray of shape (n_samples, n_features)
        Input features on GPU.
    y : cp.ndarray of shape (n_samples, n_outcomes)
        Target values on GPU.
        
    Returns
    -------
    test_stats : cp.ndarray of shape (n_features,)
        Test statistics for each feature.
    """
    n_samples, n_features = X.shape
    n_outcomes = y.shape[1]
    
    # Standardize X and y
    X_std = (X - cp.mean(X, axis=0)) / cp.std(X, axis=0, ddof=1)
    y_std = (y - cp.mean(y, axis=0)) / cp.std(y, axis=0, ddof=1)
    
    # Handle potential NaNs from division by zero
    X_std = cp.nan_to_num(X_std)
    y_std = cp.nan_to_num(y_std)
    
    # Compute correlation-based test statistics
    # For multiple outcomes, use sum of squared correlations
    test_stats = cp.zeros(n_features, dtype=cp.float32)
    
    for j in range(n_features):
        corr_sum_sq = 0.0
        for k in range(n_outcomes):
            corr = cp.sum(X_std[:, j] * y_std[:, k]) / n_samples
            corr_sum_sq += corr * corr
        
        test_stats[j] = corr_sum_sq
    
    return test_stats


def gpu_compute_split_criterion(X: np.ndarray, 
                              y: np.ndarray, 
                              feature_idx: int,
                              min_samples_leaf: int = 5) -> Tuple[float, float]:
    """
    Compute the best split point for a feature using GPU acceleration.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input features.
    y : ndarray of shape (n_samples,) or (n_samples, n_outcomes)
        Target values.
    feature_idx : int
        Index of the feature to split on.
    min_samples_leaf : int, default=5
        Minimum samples required in each leaf.
        
    Returns
    -------
    split_value : float
        The value to split on.
    criterion : float
        The criterion value (higher is better).
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration but not installed")
    
    # Transfer data to GPU
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)
    
    # Extract the feature to split on
    feature = X_gpu[:, feature_idx]
    
    # Ensure y has shape (n_samples, n_outcomes)
    if y_gpu.ndim == 1:
        y_gpu = y_gpu.reshape(-1, 1)
    
    # Sort data by feature value
    sort_indices = cp.argsort(feature)
    feature_sorted = feature[sort_indices]
    y_sorted = y_gpu[sort_indices]
    
    n_samples = X_gpu.shape[0]
    
    # Find unique values and potential split points
    unique_values = cp.unique(feature_sorted)
    
    # If there's only one unique value, cannot split
    if len(unique_values) <= 1:
        return float('nan'), 0.0
    
    # Find midpoints between unique values as potential splits
    split_candidates = (unique_values[:-1] + unique_values[1:]) / 2
    
    # Initialize arrays for results
    n_splits = len(split_candidates)
    criteria = cp.zeros(n_splits, dtype=cp.float32)
    
    # Evaluate each candidate split
    for i, split in enumerate(split_candidates):
        # Determine left and right indices
        left_mask = feature_sorted <= split
        right_mask = ~left_mask
        
        n_left = cp.sum(left_mask)
        n_right = n_samples - n_left
        
        # Skip if split doesn't meet min_samples_leaf
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue
        
        # Compute criterion (e.g., reduction in variance)
        left_mean = cp.mean(y_sorted[left_mask], axis=0)
        right_mean = cp.mean(y_sorted[right_mask], axis=0)
        
        # Weighted impurity reduction
        total_var = cp.var(y_gpu, axis=0) * n_samples
        left_var = cp.var(y_sorted[left_mask], axis=0) * n_left if n_left > 1 else 0
        right_var = cp.var(y_sorted[right_mask], axis=0) * n_right if n_right > 1 else 0
        
        impurity_reduction = cp.sum(total_var - (left_var + right_var))
        criteria[i] = impurity_reduction
    
    # Find best split
    best_idx = cp.argmax(criteria)
    
    # If no valid split was found, return NaN
    if criteria[best_idx] == 0:
        return float('nan'), 0.0
    
    return float(split_candidates[best_idx]), float(criteria[best_idx])


def gpu_compute_node_statistics(X: np.ndarray, 
                             y: np.ndarray,
                             sample_indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute node statistics using GPU acceleration.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input features.
    y : ndarray of shape (n_samples,) or (n_samples, n_outcomes)
        Target values.
    sample_indices : ndarray of shape (n_node_samples,), optional
        Indices of the samples in the node.
        
    Returns
    -------
    stats : dict
        Dictionary with node statistics.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration but not installed")
    
    # Transfer data to GPU
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)
    
    # Select samples if indices are provided
    if sample_indices is not None:
        X_node = X_gpu[sample_indices]
        y_node = y_gpu[sample_indices]
    else:
        X_node = X_gpu
        y_node = y_gpu
    
    # Ensure y has shape (n_samples, n_outcomes)
    if y_node.ndim == 1:
        y_node = y_node.reshape(-1, 1)
    
    n_samples = X_node.shape[0]
    
    # Compute basic statistics
    node_stats = {
        'n_samples': int(n_samples),
        'y_mean': cp.mean(y_node, axis=0).get(),
        'y_var': cp.var(y_node, axis=0).get() if n_samples > 1 else np.zeros(y_node.shape[1]),
        'X_mean': cp.mean(X_node, axis=0).get(),
        'X_var': cp.var(X_node, axis=0).get() if n_samples > 1 else np.zeros(X_node.shape[1])
    }
    
    return node_stats
