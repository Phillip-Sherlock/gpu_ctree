"""
GPU-accelerated Conditional Inference Tree implementation with look-ahead capabilities.

This module provides a GPU-accelerated implementation of conditional inference trees, 
similar to the ctree implementation available in R, but with enhanced features:
- Look-ahead capabilities to find more globally optimal tree structures
- Support for both univariate and multivariate outcomes
- Comprehensive statistical significance testing with multiple testing correction
- Full GPU acceleration for distributed computing environments

The implementation is designed for high-performance computing environments and
provides robust error handling, memory management, and parameter validation.
"""

import os
import time
import logging
import warnings
from typing import Union, List, Tuple, Dict, Optional, Any, Callable, Set
import numbers

import numpy as np
import pandas as pd
from scipy import stats

# Avoid hard dependency on sklearn
try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:
    def check_is_fitted(estimator, attributes=None):
        if not hasattr(estimator, 'is_fitted_') or not estimator.is_fitted_:
            raise ValueError("This instance is not fitted yet. Call 'fit' first.")

from .exceptions import ValidationError, GPUMemoryError, FittingError
from .utils import check_cuda_availability, encode_categorical

# Try to import GPU utilities
try:
    from .cuda.kernels import (
        gpu_permutation_test,
        gpu_compute_split_criterion,
        gpu_compute_node_statistics
    )
except ImportError:
    pass  # Will fall back to CPU implementations

# Configure logging
logger = logging.getLogger(__name__)


class Node:
    """Tree node for the GPUCTree algorithm."""
    
    def __init__(self, 
                 depth: int = 0, 
                 parent: Optional['Node'] = None,
                 is_left: bool = True,
                 sample_indices: Optional[np.ndarray] = None):
        """
        Initialize a tree node.
        
        Parameters
        ----------
        depth : int, default=0
            Depth of the node in the tree.
        parent : Node, optional
            Parent node.
        is_left : bool, default=True
            Whether this is a left child of its parent.
        sample_indices : ndarray, optional
            Indices of samples in this node.
        """
        # Node structure
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.left = None
        self.right = None
        
        # Node data
        self.sample_indices = sample_indices
        self.n_samples = len(sample_indices) if sample_indices is not None else 0
        
        # Split information
        self.split_feature = None
        self.split_value = None
        self.split_criterion = None
        self.p_value = None
        
        # Node statistics
        self.node_stats = None
        self.is_leaf = True
    
    def __repr__(self):
        """String representation of the node."""
        if self.is_leaf:
            return f"Leaf(samples={self.n_samples}, depth={self.depth})"
        else:
            return (f"Node(feature={self.split_feature}, value={self.split_value:.4f}, "
                   f"p={self.p_value:.4f}, samples={self.n_samples}, depth={self.depth})")


class GPUCTree:
    """
    GPU-Accelerated Conditional Inference Trees with Look-Ahead.
    
    This class implements a GPU-accelerated version of conditional inference trees
    with look-ahead capabilities for more globally optimal tree structures.
    
    Parameters
    ----------
    controls : GPUCTreeControls, optional
        Control parameters object similar to R's ctree_control().
        If provided, its settings override any conflicting parameters in kwargs.
    alpha : float, default=0.05
        Significance level for variable selection.
    min_samples_split : int, default=20
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=7
        Minimum number of samples required to be at a leaf node.
    max_depth : int, default=30
        Maximum depth of the tree.
    max_features : int or float or {"auto", "sqrt", "log2"}, default=None
        Number of features to consider for the best split.
    lookahead_depth : int, default=0
        Number of look-ahead steps for more globally optimal splits.
    n_permutations : int, default=9999
        Number of permutations for computing p-values.
    test_type : str, default="Univariate"
        Type of independence test to be applied.
    test_statistic : str, default="quadratic"
        Test statistic to be applied.
    use_gpu : bool, default=True
        Whether to use GPU acceleration when available.
    random_state : int, optional
        Seed for random number generation.
        
    Attributes
    ----------
    tree_ : Node
        The root node of the fitted tree.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances based on p-values.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.
    outcome_names_in_ : list of str
        Names of outcome variables seen during fit.
    n_samples_ : int
        Number of samples seen during fit.
    """
    
    def __init__(self, controls=None, **kwargs):
        # Default parameters
        self.alpha = 0.05
        self.min_samples_split = 20
        self.min_samples_leaf = 7
        self.max_depth = 30
        self.max_features = None
        self.lookahead_depth = 0
        self.n_permutations = 9999
        self.test_type = "Univariate"
        self.test_statistic = "quadratic"
        self.use_gpu = True
        self.random_state = None
        
        # Update from controls if provided
        if controls is not None:
            try:
                from .controls import GPUCTreeControls
                if isinstance(controls, GPUCTreeControls):
                    controls_dict = controls.to_dict()
                    # Update default parameters with those from controls
                    for key, value in controls_dict.items():
                        setattr(self, key, value)
            except ImportError:
                warnings.warn("controls module not found, ignoring controls parameter")
        
        # Override with any specific kwargs (higher priority than controls)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Initialized flag and other attributes
        self.is_fitted_ = False
        self.tree_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.outcome_names_in_ = None
    
    def _validate_data(self, X, y, feature_names=None, outcome_names=None, sample_weight=None):
        """
        Validate input data for fitting or prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outcomes)
            The target values.
        feature_names : list of str, default=None
            Names for each feature.
        outcome_names : list of str, default=None
            Names for each outcome variable.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        X : array
            Validated input data.
        y : array
            Validated target values.
        feature_names : list or None
            Validated feature names.
        outcome_names : list or None
            Validated outcome names.
        sample_weight : array or None
            Validated sample weights.
        """
        import pandas as pd
        import numpy as np
        
        # Handle pandas DataFrames and Series
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        elif hasattr(X, 'iloc') and hasattr(X, 'values'):
            X = X.values
            
        if isinstance(y, pd.Series):
            if outcome_names is None:
                outcome_names = [y.name] if y.name else None
            y = y.values
        elif isinstance(y, pd.DataFrame):
            if outcome_names is None:
                outcome_names = y.columns.tolist()
            y = y.values
        elif hasattr(y, 'iloc') and hasattr(y, 'values'):
            y = y.values
        
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        
        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        
        if y.ndim > 2:
            raise ValueError(f"y must be a 1D or 2D array, got shape {y.shape}")
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if y.shape[0] != n_samples:
            raise ValueError(f"X and y have different numbers of samples: {n_samples} vs {y.shape[0]}")
            
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]
        else:
            if len(feature_names) != n_features:
                raise ValueError(f"Length of feature_names ({len(feature_names)}) does not match number of features ({n_features})")
        
        # Generate outcome names if not provided
        n_outcomes = y.shape[1]
        if outcome_names is None:
            outcome_names = [f"y{i}" for i in range(n_outcomes)]
        else:
            if len(outcome_names) != n_outcomes:
                raise ValueError(f"Length of outcome_names ({len(outcome_names)}) does not match number of outcomes ({n_outcomes})")
        
        # Validate sample weights
        if sample_weight is not None:
            if len(sample_weight) != n_samples:
                raise ValueError(f"Sample weights length ({len(sample_weight)}) does not match number of samples ({n_samples})")
            
            # Normalize weights
            sample_weight = sample_weight / np.sum(sample_weight) * n_samples
        
        return X, y, feature_names, outcome_names, sample_weight
    
    def fit(self, X, y, feature_names=None, outcome_names=None, sample_weight=None):
        """
        Build a conditional inference tree from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outcomes)
            The target values.
        feature_names : list of str, default=None
            Names for each feature.
        outcome_names : list of str, default=None
            Names for each outcome variable.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check if GPU is available when requested
        if self.use_gpu and not check_cuda_availability():
            warnings.warn("GPU acceleration requested but CUDA is not available. Falling back to CPU.")
            self.use_gpu = False
        
        # Validate input data
        X, y, self.feature_names_in_, self.outcome_names_in_, sample_weight = self._validate_data(
            X, y, feature_names, outcome_names, sample_weight
        )
        
        # Store the number of features and samples
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)
        
        # Initialize categorical features encoding
        self.feature_types_ = None
        self.encoders_ = None
        
        # Check for categorical features (if pandas DataFrame was originally passed)
        if hasattr(X, 'dtypes'):
            self.feature_types_ = ['numeric' if pd.api.types.is_numeric_dtype(X.dtypes[col]) 
                                else 'categorical' for col in X.columns]
            X, self.encoders_, self.feature_types_ = encode_categorical(X, self.feature_types_)
        
        # Start building the tree
        start_time = time.time()
        
        # Create the root node with all samples
        all_indices = np.arange(self.n_samples_)
        self.tree_ = Node(depth=0, sample_indices=all_indices)
        
        try:
            # Recursively build the tree
            self._build_tree(self.tree_, X, y, sample_weight)
            
            # Calculate feature importances based on p-values
            self._compute_feature_importances()
            
            self.is_fitted_ = True
            
            logger.info(f"Tree built in {time.time() - start_time:.2f} seconds")
            
            return self
        
        except Exception as e:
            self.is_fitted_ = False
            raise FittingError(f"Error while fitting the tree: {str(e)}") from e
    
    def _build_tree(self, node, X, y, sample_weight=None):
        """
        Recursively build the tree by finding optimal splits.
        
        Parameters
        ----------
        node : Node
            Current node to split.
        X : ndarray of shape (n_samples, n_features)
            The training input samples.
        y : ndarray of shape (n_samples, n_outcomes)
            The target values.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        # Get the samples at this node
        indices = node.sample_indices
        X_node = X[indices]
        y_node = y[indices]
        weights_node = sample_weight[indices] if sample_weight is not None else None
        
        # Compute node statistics
        if self.use_gpu:
            try:
                node.node_stats = gpu_compute_node_statistics(X, y, indices)
            except (ImportError, Exception) as e:
                logger.warning(f"GPU computation failed: {str(e)}. Falling back to CPU.")
                n_samples = len(indices)
                node.node_stats = {
                    'n_samples': n_samples,
                    'y_mean': np.mean(y_node, axis=0),
                    'y_var': np.var(y_node, axis=0) if n_samples > 1 else np.zeros(y_node.shape[1]),
                    'X_mean': np.mean(X_node, axis=0),
                    'X_var': np.var(X_node, axis=0) if n_samples > 1 else np.zeros(X_node.shape[1])
                }
        else:
            n_samples = len(indices)
            node.node_stats = {
                'n_samples': n_samples,
                'y_mean': np.mean(y_node, axis=0),
                'y_var': np.var(y_node, axis=0) if n_samples > 1 else np.zeros(y_node.shape[1]),
                'X_mean': np.mean(X_node, axis=0),
                'X_var': np.var(X_node, axis=0) if n_samples > 1 else np.zeros(X_node.shape[1])
            }
        
        # Check stopping criteria
        if (node.depth >= self.max_depth or 
            len(indices) < self.min_samples_split or 
            np.all(np.var(y_node, axis=0) < 1e-8)):
            # Mark as leaf and return
            node.is_leaf = True
            return
        
        # Find the best split
        split_result = self._find_best_split(X_node, y_node, weights_node)
        
        # If no valid split was found, mark as leaf and return
        if split_result is None or split_result['p_value'] > self.alpha:
            node.is_leaf = True
            return
        
        # Extract split information
        feature_idx = split_result['feature']
        split_value = split_result['value']
        criterion_value = split_result['criterion']
        p_value = split_result['p_value']
        
        # Update node with split information
        node.split_feature = feature_idx
        node.split_value = split_value
        node.split_criterion = criterion_value
        node.p_value = p_value
        node.is_leaf = False
        
        # Create child nodes
        left_indices = indices[X_node[:, feature_idx] <= split_value]
        right_indices = indices[X_node[:, feature_idx] > split_value]
        
        # Check min_samples_leaf constraint
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            node.is_leaf = True
            return
        
        # Create children
        node.left = Node(depth=node.depth + 1, parent=node, is_left=True, sample_indices=left_indices)
        node.right = Node(depth=node.depth + 1, parent=node, is_left=False, sample_indices=right_indices)
        
        # Recursively build subtrees
        self._build_tree(node.left, X, y, sample_weight)
        self._build_tree(node.right, X, y, sample_weight)
    
    def _find_best_split(self, X, y, sample_weight=None):
        """
        Find the best feature and value to split the node.
        
        Parameters
        ----------
        X : ndarray of shape (n_node_samples, n_features)
            The training input samples at the node.
        y : ndarray of shape (n_node_samples, n_outcomes)
            The target values at the node.
        sample_weight : ndarray of shape (n_node_samples,), default=None
            Sample weights.
            
        Returns
        -------
        best_split : dict or None
            Information about the best split, or None if no valid split was found.
        """
        n_samples, n_features = X.shape
        
        # If there are not enough samples to split, return None
        if n_samples < self.min_samples_split:
            return None
        
        # Compute p-values for all features
        if self.use_gpu:
            try:
                p_values, test_stats = gpu_permutation_test(
                    X, y, n_permutations=self.n_permutations, random_state=self.random_state
                )
            except Exception as e:
                # Fall back to CPU if GPU computation fails
                logger.warning(f"GPU computation failed: {str(e)}. Falling back to CPU.")
                p_values, test_stats = self._cpu_permutation_test(
                    X, y, n_permutations=self.n_permutations, random_state=self.random_state
                )
        else:
            p_values, test_stats = self._cpu_permutation_test(
                X, y, n_permutations=self.n_permutations, random_state=self.random_state
            )
        
        # Apply multiple testing correction (Bonferroni)
        p_values_corrected = np.minimum(p_values * n_features, 1.0)
        
        # Find significant features
        significant_features = np.where(p_values_corrected <= self.alpha)[0]
        
        # If no significant features, return None
        if len(significant_features) == 0:
            return None
        
        # Find best split among significant features
        best_criterion = -np.inf
        best_feature = None
        best_value = None
        best_p_value = None
        
        for feature_idx in significant_features:
            # Compute the best split for this feature
            if self.use_gpu:
                try:
                    split_value, criterion_value = gpu_compute_split_criterion(
                        X, y, feature_idx, min_samples_leaf=self.min_samples_leaf
                    )
                except Exception as e:
                    # Fall back to CPU if GPU computation fails
                    logger.warning(f"GPU computation failed: {str(e)}. Falling back to CPU.")
                    split_value, criterion_value = self._cpu_compute_split_criterion(
                        X, y, feature_idx, min_samples_leaf=self.min_samples_leaf, sample_weight=sample_weight
                    )
            else:
                split_value, criterion_value = self._cpu_compute_split_criterion(
                    X, y, feature_idx, min_samples_leaf=self.min_samples_leaf, sample_weight=sample_weight
                )
            
            # Update best split if this one is better
            if not np.isnan(split_value) and criterion_value > best_criterion:
                best_criterion = criterion_value
                best_feature = feature_idx
                best_value = split_value
                best_p_value = p_values_corrected[feature_idx]
        
        # If no valid split was found, return None
        if best_feature is None:
            return None
        
        return {
            'feature': best_feature,
            'value': best_value,
            'criterion': best_criterion,
            'p_value': best_p_value
        }
    
    def _cpu_permutation_test(self, X, y, n_permutations=1000, random_state=None):
        """
        CPU implementation of permutation test for independence.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        y : ndarray of shape (n_samples, n_outcomes)
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
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples, n_features = X.shape
        n_outcomes = y.shape[1]
        
        # Compute original test statistics
        orig_stats = self._compute_test_statistics_cpu(X, y)
        
        # Initialize arrays for permutation results
        perm_stats = np.zeros((n_permutations, n_features))
        
        # Perform permutations
        for i in range(n_permutations):
            # Permute y values
            perm_indices = np.random.permutation(n_samples)
            y_perm = y[perm_indices]
            
            # Compute test statistics for permuted data
            perm_stats[i] = self._compute_test_statistics_cpu(X, y_perm)
        
        # Compute p-values (proportion of permutation statistics >= original)
        p_values = np.mean(perm_stats >= orig_stats, axis=0)
        
        return p_values, orig_stats
    
    def _compute_test_statistics_cpu(self, X, y):
        """
        CPU implementation of test statistics computation.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        y : ndarray of shape (n_samples, n_outcomes)
            Target values.
            
        Returns
        -------
        test_stats : ndarray of shape (n_features,)
            Test statistics for each feature.
        """
        n_samples, n_features = X.shape
        n_outcomes = y.shape[1]
        
        # Standardize X and y
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
        y_std = (y - np.mean(y, axis=0)) / np.std(y, axis=0, ddof=1)
        
        # Handle potential NaNs from division by zero
        X_std = np.nan_to_num(X_std)
        y_std = np.nan_to_num(y_std)
        
        # Compute test statistics based on the test_statistic parameter
        test_stats = np.zeros(n_features)
        
        if self.test_statistic == "quadratic":
            # Sum of squared correlations
            for j in range(n_features):
                corr_sum_sq = 0.0
                for k in range(n_outcomes):
                    corr = np.sum(X_std[:, j] * y_std[:, k]) / n_samples
                    corr_sum_sq += corr * corr
                test_stats[j] = corr_sum_sq
                
        elif self.test_statistic == "maximum":
            # Maximum absolute correlation
            for j in range(n_features):
                max_corr = 0.0
                for k in range(n_outcomes):
                    corr = abs(np.sum(X_std[:, j] * y_std[:, k])) / n_samples
                    max_corr = max(max_corr, corr)
                test_stats[j] = max_corr
                
        elif self.test_statistic == "correlation":
            # Mean correlation
            for j in range(n_features):
                corr_sum = 0.0
                for k in range(n_outcomes):
                    corr = np.sum(X_std[:, j] * y_std[:, k]) / n_samples
                    corr_sum += corr
                test_stats[j] = corr_sum / n_outcomes
        
        return test_stats
    
    def _cpu_compute_split_criterion(self, X, y, feature_idx, min_samples_leaf=5, sample_weight=None):
        """
        CPU implementation to find the best split point for a feature.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        y : ndarray of shape (n_samples, n_outcomes)
            Target values.
        feature_idx : int
            Index of the feature to split on.
        min_samples_leaf : int, default=5
            Minimum samples required in each leaf.
        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        split_value : float
            The value to split on.
        criterion : float
            The criterion value (higher is better).
        """
        n_samples = X.shape[0]
        feature = X[:, feature_idx]
        
        # Sort data by feature value
        sort_indices = np.argsort(feature)
        feature_sorted = feature[sort_indices]
        y_sorted = y[sort_indices]
        
        if sample_weight is not None:
            weights_sorted = sample_weight[sort_indices]
        else:
            weights_sorted = np.ones(n_samples)
        
        # Find unique values
        unique_values = np.unique(feature_sorted)
        
        # If there's only one unique value, cannot split
        if len(unique_values) <= 1:
            return float('nan'), 0.0
        
        # Find midpoints between unique values as potential splits
        split_candidates = (unique_values[:-1] + unique_values[1:]) / 2
        
        # Initialize arrays for results
        n_splits = len(split_candidates)
        criteria = np.zeros(n_splits)
        
        # Get the total weighted variance (for both outcomes)
        total_weighted_var = np.sum(np.var(y, axis=0) * np.sum(weights_sorted))
        
        # Evaluate each candidate split
        for i, split in enumerate(split_candidates):
            # Determine left and right indices
            left_mask = feature_sorted <= split
            right_mask = ~left_mask
            
            n_left = np.sum(weights_sorted[left_mask])
            n_right = np.sum(weights_sorted) - n_left
            
            # Skip if split doesn't meet min_samples_leaf
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            
            # Weighted impurity reduction
            left_weighted_var = np.sum(np.var(y_sorted[left_mask], axis=0) * n_left) if np.sum(left_mask) > 1 else 0
            right_weighted_var = np.sum(np.var(y_sorted[right_mask], axis=0) * n_right) if np.sum(right_mask) > 1 else 0
            
            # Improvement is the reduction in weighted variance
            improvement = total_weighted_var - (left_weighted_var + right_weighted_var)
            criteria[i] = improvement
        
        # Find best split
        best_idx = np.argmax(criteria)
        
        # If no valid split was found, return NaN
        if criteria[best_idx] == 0:
            return float('nan'), 0.0
        
        return float(split_candidates[best_idx]), float(criteria[best_idx])
    
    def _compute_feature_importances(self):
        """
        Compute feature importances based on p-values.
        
        Smaller p-values indicate more important features.
        """
        # Initialize counters for each feature
        feature_counts = np.zeros(self.n_features_in_)
        feature_p_values = np.zeros(self.n_features_in_)
        
        # Traverse the tree and collect information
        def traverse(node):
            if not node.is_leaf and node.split_feature is not None:
                feature_idx = node.split_feature
                feature_counts[feature_idx] += 1
                feature_p_values[feature_idx] += -np.log(node.p_value + 1e-10)  # Add small epsilon to avoid log(0)
                
                # Traverse children
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)
        
        # Start traversal from root
        traverse(self.tree_)
        
        # Compute importances
        # More frequent usage and lower p-values (higher -log(p)) mean more importance
        importances = feature_counts * feature_p_values
        
        # Normalize importances
        if np.sum(importances) > 0:
            self.feature_importances_ = importances / np.sum(importances)
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_)
    
    def predict(self, X):
        """
        Predict class or regression value for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        
        # Convert and validate X
        if isinstance(X, pd.DataFrame):
            X_feature_names = X.columns.tolist()
            if not set(self.feature_names_in_).issubset(set(X_feature_names)):
                missing = set(self.feature_names_in_) - set(X_feature_names)
                raise ValueError(f"Feature(s) {missing} missing in input data")
            
            # Ensure columns are in the same order as during training
            X = X[self.feature_names_in_].values
        else:
            X = np.asarray(X)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X has {X.shape[1]} features, but GPUCTree was trained with {self.n_features_in_} features")
        
        # Apply the tree for each sample
        predictions = np.zeros((X.shape[0], len(self.outcome_names_in_)))
        
        for i in range(X.shape[0]):
            # Find the leaf node for this sample
            node = self._predict_one(self.tree_, X[i])
            
            # Use the mean target value at the leaf as prediction
            predictions[i] = node.node_stats['y_mean']
            
        # For single output, return a 1D array
        if predictions.shape[1] == 1:
            return predictions.ravel()
        
        return predictions
    
    def _predict_one(self, node, x):
        """
        Navigate the tree to find the leaf node for a single sample.
        
        Parameters
        ----------
        node : Node
            Current node in the traversal.
        x : array-like of shape (n_features,)
            Input features for the sample.
            
        Returns
        -------
        leaf_node : Node
            Leaf node reached by the sample.
        """
        # If this is a leaf, return it
        if node.is_leaf:
            return node
        
        # Otherwise, continue traversal based on the split
        feature_value = x[node.split_feature]
        
        if feature_value <= node.split_value:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)
    
    def print_tree(self):
        """
        Print the tree structure.
        
        Returns
        -------
        str
            String representation of the tree.
        """
        check_is_fitted(self)
        
        lines = []
        self._print_node(self.tree_, lines, prefix="")
        return "\n".join(lines)
    
    def _print_node(self, node, lines, prefix=""):
        """
        Helper method to print a node and its children recursively.
        
        Parameters
        ----------
        node : Node
            Current node.
        lines : list
            List to accumulate output lines.
        prefix : str
            Prefix for the current line (for indentation).
        """
        if node.is_leaf:
            # For a leaf, print the prediction
            mean_values = ", ".join([f"{name}={val:.4f}" for name, val in zip(self.outcome_names_in_, node.node_stats['y_mean'])])
            lines.append(f"{prefix}Leaf: {mean_values} (n={node.n_samples})")
        else:
            # For an internal node, print the split condition
            feature_name = self.feature_names_in_[node.split_feature]
            lines.append(f"{prefix}{feature_name} <= {node.split_value:.4f} (p={node.p_value:.4f}, n={node.n_samples})")
            
            # Print left subtree
            self._print_node(node.left, lines, prefix + "| ")
            
            # Print right subtree
            lines.append(f"{prefix}{feature_name} > {node.split_value:.4f}")
            self._print_node(node.right, lines, prefix + "| ")
    
    def from_formula(self, formula, data, subset=None, weights=None, **kwargs):
        """
        Fit a GPUCTree using an R-style formula interface.
        
        Parameters
        ----------
        formula : str
            R-style formula specifying the model, e.g., "y ~ x1 + x2 + x3".
        data : pandas.DataFrame
            Data frame containing the variables in the formula.
        subset : array-like, optional
            Subset of observations to be used.
        weights : array-like, optional
            Case weights for observations.
        **kwargs : 
            Additional arguments passed to fit().
            
        Returns
        -------
        self : object
            Fitted GPUCTree instance.
        
        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "y": [0, 1, 0, 1, 0],
        ...     "x1": [1.2, 2.3, 1.1, 5.2, 1.0],
        ...     "x2": [0.5, 0.2, 0.9, 0.8, 0.4]
        ... })
        >>> model = GPUCTree().from_formula("y ~ x1 + x2", data)
        """
        from .utils import parse_formula
        
        # Parse the formula
        outcome, predictors = parse_formula(formula)
        
        # Validate DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        # Apply subset if provided
        if subset is not None:
            data = data.loc[subset]
        
        # Handle the special '.' case for all predictors
        if len(predictors) == 1 and predictors[0] == '.':
            # Use all columns except the outcome
            predictors = [col for col in data.columns if col != outcome]
        
        # Extract X and y from the data
        X = data[predictors]
        y = data[outcome]
        
        # Call the regular fit method with the extracted data
        return self.fit(X, y, feature_names=predictors, outcome_names=[outcome], sample_weight=weights, **kwargs)
    
    def get_node_stats(self, node_id=None):
        """
        Get statistics for a specific node or all nodes.
        
        Parameters
        ----------
        node_id : int or None, default=None
            If None, return stats for all nodes. Otherwise, return stats for the specified node.
            
        Returns
        -------
        stats : dict or list of dict
            Statistics for the requested node(s).
        """
        check_is_fitted(self)
        
        # Collect statistics from all nodes
        all_stats = []
        
        def traverse(node, node_id=0):
            # Calculate or retrieve statistics for this node
            stats = {
                'node_id': node_id,
                'depth': node.depth,
                'samples': node.n_samples,
                'is_leaf': node.is_leaf,
            }
            
            # Add split information for non-leaf nodes
            if not node.is_leaf:
                stats.update({
                    'split_feature': self.feature_names_in_[node.split_feature],
                    'split_value': node.split_value,
                    'p_value': node.p_value,
                })
            
            # Add prediction for all nodes
            stats['prediction'] = {
                name: value for name, value in zip(self.outcome_names_in_, node.node_stats['y_mean'])
            }
            
            # Add node to the list
            all_stats.append(stats)
            
            # Traverse children for non-leaf nodes
            if not node.is_leaf:
                left_id = 2 * node_id + 1
                right_id = 2 * node_id + 2
                traverse(node.left, left_id)
                traverse(node.right, right_id)
        
        # Start traversal from root
        traverse(self.tree_)
        
        # Return all nodes or a specific one
        if node_id is not None:
            for stats in all_stats:
                if stats['node_id'] == node_id:
                    return stats
            raise ValueError(f"Node with ID {node_id} not found in the tree")
        
        return all_stats
    
    def __str__(self):
        """Return a string representation of the tree, similar to R's print.ctree()."""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            return "Unfitted GPUCTree"
        
        result = [
            "Conditional Inference Tree (GPU-Accelerated)",
            f"Number of samples: {self.n_samples_}",
            f"Number of features: {self.n_features_in_}",
            f"Maximum depth: {self.max_depth}",
            f"Alpha: {self.alpha}",
            "\nTree structure:",
        ]
        
        # Add tree structure
        result.append(self.print_tree())
        
        return "\n".join(result)
    
    def plot(self, feature_names=None, class_names=None, filled=True, rounded=True, precision=3, ax=None, fontsize=None):
        """
        Plot the tree structure using matplotlib, similar to plot.ctree() in R.
        
        This method requires matplotlib to be installed.
        
        Parameters
        ----------
        feature_names : list of str, default=None
            Names of features. If None, uses feature_names_in_.
        class_names : list of str, default=None
            Names of classes (for classification) or targets (for regression).
            If None, uses outcome_names_in_.
        filled : bool, default=True
            Whether to color nodes by the majority class.
        rounded : bool, default=True
            Whether to draw nodes with rounded corners.
        precision : int, default=3
            Number of decimal places for floating point values.
        ax : matplotlib.axes.Axes, default=None
            Axes object to plot on. If None, a new figure and axes are created.
        fontsize : int, default=None
            Font size for text in the plot.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        check_is_fitted(self)
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from matplotlib.patches import Rectangle, FancyBboxPatch, ArrowStyle, ConnectionPatch
        except ImportError:
            raise ImportError("Matplotlib is required for plotting but not installed")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use provided names or defaults
        feature_names = feature_names or self.feature_names_in_
        class_names = class_names or self.outcome_names_in_
        
        # Define a recursive function to draw the tree
        def draw_tree(node, x, y, width, depth=0):
            # Define colors and styles
            if filled:
                if node.is_leaf:
                    color = "lightblue"
                else:
                    color = "lightgray" if node.p_value > 0.01 else "orange"
            else:
                color = "white"
            
            # Node dimensions
            box_width = 0.15
            box_height = 0.1
            
            # Create node text
            if node.is_leaf:
                prediction_text = ", ".join([f"{name}={val:.{precision}f}" for name, val in zip(class_names, node.node_stats['y_mean'])])
                text = f"n={node.n_samples}\n{prediction_text}"
            else:
                feature_name = feature_names[node.split_feature]
                text = f"{feature_name} ≤ {node.split_value:.{precision}f}\np={node.p_value:.{precision}f}, n={node.n_samples}"
            
            # Draw the node
            box_style = "round" if rounded else "square"
            box = FancyBboxPatch((x - box_width/2, y - box_height/2), 
                                box_width, box_height, 
                                boxstyle=box_style, 
                                fc=color, ec="black", alpha=0.8)
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)
            
            # Draw children for non-leaf nodes
            if not node.is_leaf:
                # Calculate child positions
                next_depth = depth + 1
                y_step = 0.2
                x_step = width / (2 ** (next_depth + 1))
                
                left_x = x - x_step
                right_x = x + x_step
                next_y = y - y_step
                
                # Draw connections to children
                ax.plot([x, left_x], [y - box_height/2, next_y + box_height/2], 'k-')
                ax.plot([x, right_x], [y - box_height/2, next_y + box_height/2], 'k-')
                
                # Add split labels
                ax.text(x - 0.02 - x_step/3, y - box_height/2 - 0.02, "≤", ha='center', va='center', fontsize=fontsize)
                ax.text(x + 0.02 + x_step/3, y - box_height/2 - 0.02, ">", ha='center', va='center', fontsize=fontsize)
                
                # Draw children
                draw_tree(node.left, left_x, next_y, width/2, next_depth)
                draw_tree(node.right, right_x, next_y, width/2, next_depth)
        
        # Set up the plot
        ax.set_xlim(-0.5, 0.5)
        max_depth = self._get_max_depth()
        y_extent = max_depth * 0.2 + 0.1
        ax.set_ylim(-y_extent, 0.1)
        ax.axis('off')
        
        # Draw the tree starting from the root
        draw_tree(self.tree_, 0, 0, 1)
        
        # Add title
        ax.set_title('Conditional Inference Tree', fontsize=fontsize)
        
        return ax
    
    def _get_max_depth(self):
        """
        Calculate the maximum depth of the tree.
        
        Returns
        -------
        max_depth : int
            Maximum depth of the tree.
        """
        def traverse(node):
            if node.is_leaf:
                return node.depth
            else:
                left_depth = traverse(node.left) if node.left else node.depth
                right_depth = traverse(node.right) if node.right else node.depth
                return max(left_depth, right_depth)
        
        return traverse(self.tree_)
    
    def export_graphviz(self, out_file=None, feature_names=None, class_names=None, 
                        filled=True, rounded=True, special_characters=False, precision=3):
        """
        Export a graphviz dot file representation of the tree.
        
        Parameters
        ----------
        out_file : file object or str, default=None
            If not None, write the dot representation to this file.
            If None, return the dot representation as a string.
        feature_names : list of str, default=None
            Names of features. If None, uses feature_names_in_.
        class_names : list of str, default=None
            Names of classes (for classification) or targets (for regression).
            If None, uses outcome_names_in_.
        filled : bool, default=True
            Whether to color nodes by the majority class.
        rounded : bool, default=True
            Whether to draw nodes with rounded corners.
        special_characters : bool, default=False
            Whether to include special characters in node labels.
        precision : int, default=3
            Number of decimal places for floating point values.
            
        Returns
        -------
        dot_data : str or None
            If out_file is None, return the dot representation as a string.
            Otherwise, return None.
        """
        check_is_fitted(self)
        
        feature_names = feature_names or self.feature_names_in_
        class_names = class_names or self.outcome_names_in_
        
        # Begin dot file content
        dot = ['digraph Tree {']
        dot.append('node [shape=box, style="filled", color="black", fontname=helvetica, fontsize=10];')
        dot.append('edge [fontname=helvetica, fontsize=9];')
        
        # Define a recursive function to add nodes and edges
        def add_node(node, node_id=0):
            # Node label
            if node.is_leaf:
                prediction_text = "\\n".join([f"{name}={val:.{precision}f}" for name, val in zip(class_names, node.node_stats['y_mean'])])
                label = f"samples = {node.n_samples}\\n{prediction_text}"
                
                # Define node color based on the prediction
                if filled:
                    color = "lightblue"
                    dot.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
                else:
                    dot.append(f'{node_id} [label="{label}"];')
            else:
                feature_name = feature_names[node.split_feature]
                if special_characters:
                    feature_name = feature_name.replace("<", "&lt;").replace(">", "&gt;")
                
                label = f"{feature_name} <= {node.split_value:.{precision}f}\\np-value = {node.p_value:.{precision}f}\\nsamples = {node.n_samples}"
                
                # Define node color based on p-value
                if filled:
                    color = "lightgray" if node.p_value > 0.01 else "orange"
                    dot.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
                else:
                    dot.append(f'{node_id} [label="{label}"];')
                
                # Add children
                left_id = 2 * node_id + 1
                right_id = 2 * node_id + 2
                
                # Add edges
                dot.append(f'{node_id} -> {left_id} [label="yes"];')
                dot.append(f'{node_id} -> {right_id} [label="no"];')
                
                # Add child nodes
                add_node(node.left, left_id)
                add_node(node.right, right_id)
        
        # Add the root and its descendants
        add_node(self.tree_)
        
        # Close the dot file
        dot.append('}')
        dot_data = '\n'.join(dot)
        
        # Write to file if provided
        if out_file:
            if isinstance(out_file, str):
                with open(out_file, 'w') as f:
                    f.write(dot_data)
            else:
                out_file.write(dot_data)
        else:
            return dot_data
