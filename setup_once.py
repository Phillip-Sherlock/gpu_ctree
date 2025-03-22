"""
One-time setup script for GPU-CTree environment.
Run this once to configure the GPU environment permanently.
"""
import os
import sys
import subprocess
from pathlib import Path
import warnings

def get_cuda_version():
    """Get system CUDA version."""
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        cuda_line = [line for line in nvidia_smi.split('\n') if 'CUDA Version' in line]
        if cuda_line:
            return cuda_line[0].split('CUDA Version: ')[1].strip()
        return None
    except Exception:
        return None

def fix_environment():
    """Fix the GPU environment permanently."""
    print("Starting GPU-CTree environment setup...")
    
    # Get Python executable path
    python_exe = sys.executable
    pip_exe = f"{python_exe} -m pip"
    
    # Detect CUDA version
    cuda_version = get_cuda_version()
    print(f"Detected CUDA version: {cuda_version or 'None'}")
    
    # Remove existing CuPy installations
    print("\nRemoving existing CuPy installations...")
    subprocess.call(f"{pip_exe} uninstall -y cupy-cuda11x cupy-cuda116 cupy", shell=True)
    
    # Install appropriate CuPy
    print("\nInstalling appropriate CuPy version...")
    if cuda_version:
        major = int(cuda_version.split('.')[0])
        if major == 11:
            subprocess.call(f"{pip_exe} install cupy-cuda11x", shell=True)
        elif major == 12:
            subprocess.call(f"{pip_exe} install cupy-cuda12x", shell=True)
        else:
            print(f"Unknown CUDA version {cuda_version}. Installing default CuPy...")
            subprocess.call(f"{pip_exe} install cupy", shell=True)
    else:
        print("CUDA not detected. Installing default CuPy...")
        subprocess.call(f"{pip_exe} install cupy", shell=True)
    
    # Install other required GPU packages
    print("\nInstalling other required packages...")
    subprocess.call(f"{pip_exe} install numba", shell=True)
    
    # Fix for the missing GPU function issues
    print("\nUpdating kernels.py to fix GPU function issues...")
    
    # Define the path to kernels.py
    gpu_ctree_dir = Path(__file__).parent
    kernels_path = gpu_ctree_dir / "gpu_ctree" / "kernels.py"
    
    if kernels_path.exists():
        # Create a backup
        backup_path = kernels_path.with_suffix(".py.bak")
        if not backup_path.exists():  # Don't overwrite existing backups
            print(f"Creating backup of kernels.py at {backup_path}")
            with open(kernels_path, 'r') as f_in, open(backup_path, 'w') as f_out:
                f_out.write(f_in.read())
        
        # Write the updated kernels.py content
        with open(kernels_path, 'w') as f:
            f.write('''"""
GPU kernels for accelerated computation in the gpu_ctree package.
"""

# Import necessary libraries
import numpy as np
import warnings

# Import cupy if available for GPU acceleration
try:
    import cupy as cp
    from cupy.cuda.compiler import CompileException
    from cupy import RawKernel, RawModule
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

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

extern "C" __global__ void optimized_predict_kernel(
    const float* __restrict__ X,
    int* __restrict__ predictions,
    const int* __restrict__ tree_structure,
    const float* __restrict__ thresholds,
    const int* __restrict__ features,
    const int* __restrict__ class_predictions,
    const int n_samples,
    const int n_features,
    const int max_nodes) {
    
    __shared__ float shared_data[BLOCK_SIZE * 2];
    
    __syncthreads();
    
    // Calculate global thread index with grid-stride loop for handling large datasets
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_samples; i += blockDim.x * gridDim.x) {
        int node_idx = 0;  // Start at the root
        
        // Traverse tree - use loop unrolling for common depths to reduce branch overhead
        #pragma unroll 4
        while (node_idx < max_nodes && tree_structure[node_idx] != -1) {
            int feature_idx = features[node_idx];
            float threshold = thresholds[node_idx];
            float feature_val = X[i * n_features + feature_idx];
            
            // Go left or right based on the comparison
            if (feature_val <= threshold) {
                node_idx = 2 * node_idx + 1;  // Left child
            } else {
                node_idx = 2 * node_idx + 2;  // Right child
            }
        }
        
        // Assign the class prediction for this leaf node
        predictions[i] = class_predictions[node_idx];
    }
}
"""

# Define kernels for statistical functions
NODE_STATISTICS_KERNEL = """
extern "C" __global__ void node_statistics_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    int* __restrict__ node_indices,
    float* __restrict__ node_statistics,
    int n_samples,
    int n_features,
    int n_outcomes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;
    
    int node_idx = node_indices[idx];
    
    for (int f = 0; f < n_features; f++) {
        float x_val = X[idx * n_features + f];
        atomicAdd(&node_statistics[node_idx * n_features + f], x_val);
    }
    
    for (int o = 0; o < n_outcomes; o++) {
        float y_val = y[idx * n_outcomes + o];
        atomicAdd(&node_statistics[node_idx * n_features * n_outcomes + o + n_features], y_val);
    }
}
"""

SPLIT_CRITERION_KERNEL = """
extern "C" __global__ void split_criterion_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    const int* __restrict__ node_indices,
    const float* __restrict__ split_points,
    float* __restrict__ criterion_values,
    int n_samples,
    int n_features,
    int n_splits_per_feature,
    int feature_idx) {
    
    int split_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (split_idx >= n_splits_per_feature) return;
    
    float split_value = split_points[split_idx];
    float sum_left = 0.0f, sum_right = 0.0f;
    int count_left = 0, count_right = 0;
    
    for (int i = 0; i < n_samples; i++) {
        if (node_indices[i] == 1) { // Only consider samples in the current node
            float x_val = X[i * n_features + feature_idx];
            float y_val = y[i];
            
            if (x_val <= split_value) {
                sum_left += y_val;
                count_left++;
            } else {
                sum_right += y_val;
                count_right++;
            }
        }
    }
    
    // Calculate criterion (e.g., variance reduction)
    float mean_left = count_left > 0 ? sum_left / count_left : 0.0f;
    float mean_right = count_right > 0 ? sum_right / count_right : 0.0f;
    float total_mean = (sum_left + sum_right) / (count_left + count_right);
    
    // Weighted variance reduction
    float criterion = 0.0f;
    if (count_left > 0 && count_right > 0) {
        float weight_left = (float)count_left / (count_left + count_right);
        float weight_right = (float)count_right / (count_left + count_right);
        criterion = total_mean * total_mean - 
                    (weight_left * mean_left * mean_left + 
                     weight_right * mean_right * mean_right);
    }
    
    criterion_values[split_idx] = criterion;
}
"""

PERMUTATION_TEST_KERNEL = """
extern "C" __global__ void permutation_test_kernel(
    const float* __restrict__ X,
    float* __restrict__ y_permuted,
    const float* __restrict__ original_y,
    const int* __restrict__ permutation_indices,
    int n_samples,
    int n_features,
    int permutation_idx,
    int n_permutations_per_block) {
    
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) return;
    
    // Apply different permutations for different blocks
    int perm_offset = permutation_idx * n_samples;
    int perm_sample_idx = permutation_indices[perm_offset + sample_idx];
    
    // Permute the y values
    y_permuted[sample_idx] = original_y[perm_sample_idx];
}
"""

def get_optimized_predict_kernel():
    """Get optimized predict kernel"""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
    
    try:
        return RawKernel(OPTIMIZED_PREDICT_KERNEL, 'optimized_predict_kernel')
    except CompileException as e:
        warnings.warn(f"Failed to compile predict kernel: {str(e)}")
        return None

def get_node_statistics_kernel():
    """Get kernel for computing node statistics"""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
    
    try:
        return RawKernel(NODE_STATISTICS_KERNEL, 'node_statistics_kernel')
    except CompileException as e:
        warnings.warn(f"Failed to compile node statistics kernel: {str(e)}")
        return None

def get_split_criterion_kernel():
    """Get kernel for computing split criteria"""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
    
    try:
        return RawKernel(SPLIT_CRITERION_KERNEL, 'split_criterion_kernel')
    except CompileException as e:
        warnings.warn(f"Failed to compile split criterion kernel: {str(e)}")
        return None

def get_permutation_test_kernel():
    """Get kernel for permutation tests"""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
    
    try:
        return RawKernel(PERMUTATION_TEST_KERNEL, 'permutation_test_kernel')
    except CompileException as e:
        warnings.warn(f"Failed to compile permutation test kernel: {str(e)}")
        return None

def gpu_compute_node_statistics(X, y, node_indices):
    """
    Compute node statistics for each feature and outcome using GPU acceleration.
    
    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)
    y : array-like
        Target values of shape (n_samples, n_outcomes)
    node_indices : array-like
        Node indices for each sample
    
    Returns
    -------
    node_statistics : array
        Statistics for each node
    """
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy is required for GPU acceleration but not installed. Falling back to CPU.")
        return None
    
    try:
        # Convert inputs to GPU arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        node_indices_gpu = cp.asarray(node_indices, dtype=cp.int32)
        
        n_samples, n_features = X_gpu.shape
        if y_gpu.ndim == 1:
            y_gpu = y_gpu.reshape(-1, 1)
        n_outcomes = y_gpu.shape[1]
        
        # Get unique node indices
        unique_nodes = cp.unique(node_indices_gpu)
        n_nodes = len(unique_nodes)
        
        # Initialize node statistics
        node_stats = cp.zeros((n_nodes, n_features + n_outcomes), dtype=cp.float32)
        
        # Get the kernel
        kernel = get_node_statistics_kernel()
        if kernel is None:
            raise RuntimeError("Failed to initialize node statistics kernel")
        
        # Launch the kernel
        threads_per_block = 256
        blocks_per_grid = (n_samples + threads_per_block - 1) // threads_per_block
        kernel((blocks_per_grid,), (threads_per_block,), 
            (X_gpu, y_gpu, node_indices_gpu, node_stats, n_samples, n_features, n_outcomes))
        
        return cp.asnumpy(node_stats)
    
    except Exception as e:
        warnings.warn(f"GPU computation failed: {str(e)}. Falling back to CPU.")
        return None

def gpu_compute_split_criterion(X, y, node_indices, feature_idx, split_points):
    """
    Compute split criterion for a feature using GPU acceleration.
    
    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)
    y : array-like
        Target values of shape (n_samples,)
    node_indices : array-like
        Node indices for each sample
    feature_idx : int
        Feature index to compute split criterion for
    split_points : array-like
        Potential split points
    
    Returns
    -------
    criterion_values : array
        Criterion value for each split point
    """
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy is required for GPU acceleration but not installed. Falling back to CPU.")
        return None
    
    try:
        # Convert inputs to GPU arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        node_indices_gpu = cp.asarray(node_indices, dtype=cp.int32)
        split_points_gpu = cp.asarray(split_points, dtype=cp.float32)
        
        n_samples, n_features = X_gpu.shape
        n_splits = len(split_points)
        
        # Initialize criterion values
        criterion_values = cp.zeros(n_splits, dtype=cp.float32)
        
        # Get the kernel
        kernel = get_split_criterion_kernel()
        if kernel is None:
            raise RuntimeError("Failed to initialize split criterion kernel")
        
        # Launch the kernel
        threads_per_block = 256
        blocks_per_grid = (n_splits + threads_per_block - 1) // threads_per_block
        kernel((blocks_per_grid,), (threads_per_block,), 
            (X_gpu, y_gpu, node_indices_gpu, split_points_gpu, criterion_values, 
             n_samples, n_features, n_splits, feature_idx))
        
        return cp.asnumpy(criterion_values)
    
    except Exception as e:
        warnings.warn(f"GPU computation failed: {str(e)}. Falling back to CPU.")
        return None

def gpu_permutation_test(X, y, n_permutations=1000, random_state=None):
    """
    Perform permutation test using GPU acceleration.
    
    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)
    y : array-like
        Target values of shape (n_samples,)
    n_permutations : int, default=1000
        Number of permutations
    random_state : int or None, default=None
        Random state for reproducibility
    
    Returns
    -------
    p_values : array
        P-values for each feature
    test_stats : array
        Test statistics for each feature
    """
    if not CUPY_AVAILABLE:
        warnings.warn("CuPy is required for GPU acceleration but not installed. Falling back to CPU.")
        return None, None
    
    try:
        # Set random seed
        if random_state is not None:
            cp.random.seed(random_state)
        
        # Transfer data to GPU
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        
        n_samples, n_features = X_gpu.shape
        
        # Ensure y has shape (n_samples, n_outcomes)
        if y_gpu.ndim == 1:
            y_gpu = y_gpu.reshape(-1, 1)
        
        n_outcomes = y_gpu.shape[1]
        
        # Generate permutation indices
        permutation_indices = cp.zeros((n_permutations, n_samples), dtype=cp.int32)
        for i in range(n_permutations):
            permutation_indices[i] = cp.random.permutation(n_samples)
        
        # Initialize permuted y and test statistics
        y_permuted = cp.zeros_like(y_gpu)
        orig_test_stats = cp.zeros(n_features, dtype=cp.float32)
        perm_test_stats = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        
        # Get the kernel
        kernel = get_permutation_test_kernel()
        if kernel is None:
            raise RuntimeError("Failed to initialize permutation test kernel")
        
        # Compute original test statistics
        for j in range(n_features):
            # Calculate correlation or other test statistics
            if n_outcomes == 1:
                orig_test_stats[j] = cp.abs(cp.corrcoef(X_gpu[:, j], y_gpu[:, 0])[0, 1])
            else:
                # For multivariate outcomes, use a different statistic
                orig_test_stats[j] = cp.sum(cp.abs(cp.corrcoef(X_gpu[:, j], y_gpu.T)[0, 1:]))
        
        # Permutation test
        threads_per_block = 256
        blocks_per_grid = (n_samples + threads_per_block - 1) // threads_per_block
        
        for i in range(n_permutations):
            # Apply permutation
            kernel((blocks_per_grid,), (threads_per_block,), 
                (X_gpu, y_permuted, y_gpu, permutation_indices, n_samples, n_features, i, 1))
            
            # Compute test statistics for permuted data
            for j in range(n_features):
                if n_outcomes == 1:
                    perm_test_stats[i, j] = cp.abs(cp.corrcoef(X_gpu[:, j], y_permuted[:, 0])[0, 1])
                else:
                    perm_test_stats[i, j] = cp.sum(cp.abs(cp.corrcoef(X_gpu[:, j], y_permuted.T)[0, 1:]))
        
        # Compute p-values
        p_values = cp.zeros(n_features, dtype=cp.float32)
        for j in range(n_features):
            p_values[j] = cp.sum(perm_test_stats[:, j] >= orig_test_stats[j]) / n_permutations
        
        return cp.asnumpy(p_values), cp.asnumpy(orig_test_stats)
    
    except Exception as e:
        warnings.warn(f"GPU computation failed: {str(e)}. Falling back to CPU.")
        return None, None
''')
    else:
        print(f"Warning: kernels.py not found at expected location: {kernels_path}")

    # Update module __init__.py
    module_init_path = gpu_ctree_dir / "gpu_ctree" / "__init__.py"
    if module_init_path.exists():
        # Create a backup
        backup_path = module_init_path.with_suffix(".py.bak")
        if not backup_path.exists():
            print(f"Creating backup of __init__.py at {backup_path}")
            with open(module_init_path, 'r') as f_in, open(backup_path, 'w') as f_out:
                f_out.write(f_in.read())
        
        # Write the updated __init__.py content
        with open(module_init_path, 'w') as f:
            f.write('''"""
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
    try:
        from .kernels import (
            gpu_permutation_test,
            gpu_compute_split_criterion,
            gpu_compute_node_statistics
        )
    except (ImportError, AttributeError) as e:
        import warnings
        warnings.warn(f"Could not import GPU functions: {str(e)}")
        # Define dummy functions that return None when GPU acceleration fails
        def gpu_permutation_test(*args, **kwargs):
            warnings.warn("GPU acceleration unavailable. Falling back to CPU.")
            return None, None
            
        def gpu_compute_split_criterion(*args, **kwargs):
            warnings.warn("GPU acceleration unavailable. Falling back to CPU.")
            return None
            
        def gpu_compute_node_statistics(*args, **kwargs):
            warnings.warn("GPU acceleration unavailable. Falling back to CPU.")
            return None

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
    
    import warnings
    warnings.warn("CuPy not available. GPU acceleration will be disabled.")
''')
    else:
        print(f"Warning: __init__.py not found at expected location: {module_init_path}")
    
    # Create .gpuctreerc file in the home directory
    home = Path.home()
    rc_file = home / ".gpuctreerc"
    
    with open(rc_file, "w") as f:
        f.write(f"CUDA_VERSION={cuda_version or 'unknown'}\n")
        f.write(f"SETUP_DATE={subprocess.check_output(['date']).decode('utf-8').strip()}\n")
    
    print(f"\nConfiguration saved to {rc_file}")
    print("\nGPU-CTree environment setup complete!")
    print("You can now run your GPU-CTree examples without additional setup.")

if __name__ == "__main__":
    fix_environment()
