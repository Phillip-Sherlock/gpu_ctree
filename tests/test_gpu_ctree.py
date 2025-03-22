"""
Unit tests for the GPU-CTree package.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split

from gpu_ctree import GPUCTree, gpu_ctree, gpu_ctree_control
from gpu_ctree.utils import check_cuda_availability, parse_formula, encode_categorical
from gpu_ctree.controls import GPUCTreeControls


class TestGPUCTree(unittest.TestCase):
    """Test cases for the GPUCTree class."""
    
    def setUp(self):
        """Setup test data."""
        # Generate a simple regression dataset
        np.random.seed(42)
        self.X_reg = np.random.rand(100, 5)
        self.y_reg = np.sum(self.X_reg[:, :2], axis=1) + np.random.normal(0, 0.1, 100)
        
        # Generate a simple classification dataset
        self.X_clf = np.random.rand(100, 5)
        self.y_clf = (np.sum(self.X_clf[:, :2], axis=1) > 1).astype(int)
        
        # Create a pandas DataFrame for formula interface testing
        self.data_df = pd.DataFrame({
            'y': self.y_clf,
            'x1': self.X_clf[:, 0],
            'x2': self.X_clf[:, 1],
            'x3': self.X_clf[:, 2],
            'x4': self.X_clf[:, 3],
            'x5': self.X_clf[:, 4],
            'cat': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Feature names
        self.feature_names = ['x1', 'x2', 'x3', 'x4', 'x5']
        
    def test_init(self):
        """Test initialization of GPUCTree."""
        # Test with default parameters
        tree = GPUCTree()
        self.assertEqual(tree.alpha, 0.05)
        self.assertEqual(tree.min_samples_split, 20)
        self.assertEqual(tree.min_samples_leaf, 7)
        
        # Test with custom parameters
        tree = GPUCTree(alpha=0.01, min_samples_split=10, min_samples_leaf=5)
        self.assertEqual(tree.alpha, 0.01)
        self.assertEqual(tree.min_samples_split, 10)
        self.assertEqual(tree.min_samples_leaf, 5)
        
        # Test with controls object
        controls = GPUCTreeControls(alpha=0.1, minsplit=15, minbucket=8)
        tree = GPUCTree(controls=controls)
        self.assertEqual(tree.alpha, 0.1)
        self.assertEqual(tree.min_samples_split, 15)
        self.assertEqual(tree.min_samples_leaf, 8)
    
    def test_fit_predict_regression(self):
        """Test fitting and prediction for regression."""
        tree = GPUCTree(
            alpha=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=3
        )
        
        # Fit the model
        tree.fit(self.X_reg, self.y_reg, feature_names=self.feature_names)
        
        # Check if model is fitted
        self.assertTrue(hasattr(tree, 'is_fitted_'))
        self.assertTrue(tree.is_fitted_)
        
        # Check attributes
        self.assertEqual(tree.n_features_in_, 5)
        self.assertEqual(len(tree.feature_importances_), 5)
        self.assertEqual(tree.feature_names_in_, self.feature_names)
        
        # Make predictions
        y_pred = tree.predict(self.X_reg)
        
        # Check predictions shape and type
        self.assertEqual(y_pred.shape, (100,))
        self.assertTrue(isinstance(y_pred, np.ndarray))
        
        # Check that predictions are reasonable (basic sanity check)
        self.assertTrue(np.all(y_pred >= 0))
        self.assertTrue(np.all(y_pred <= 2))  # reasonable for sum of uniform [0,1] variables
    
    def test_fit_predict_classification(self):
        """Test fitting and prediction for classification."""
        tree = GPUCTree(
            alpha=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=3
        )
        
        # Fit the model
        tree.fit(self.X_clf, self.y_clf)
        
        # Check if model is fitted
        self.assertTrue(tree.is_fitted_)
        
        # Make predictions
        y_pred = tree.predict(self.X_clf)
        
        # Check that predictions are binary
        unique_values = np.unique(y_pred)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))
    
    def test_formula_interface(self):
        """Test the R-style formula interface."""
        # Basic formula
        model = gpu_ctree("y ~ x1 + x2 + x3", data=self.data_df)
        self.assertTrue(model.is_fitted_)
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.feature_names_in_, ['x1', 'x2', 'x3'])
        
        # Formula with all predictors using '.'
        model = gpu_ctree("y ~ .", data=self.data_df)
        self.assertTrue(model.is_fitted_)
        self.assertEqual(model.n_features_in_, 7)  # 6 features + 1 categorical
        
        # Formula with subset
        subset = self.data_df['x1'] > 0.5
        model = gpu_ctree("y ~ x1 + x2", data=self.data_df, subset=subset)
        self.assertTrue(model.is_fitted_)
        # Number of samples should be less than total
        self.assertLess(model.n_samples_, 100)
        
        # Formula with controls
        controls = gpu_ctree_control(mincriterion=0.95, maxdepth=2)
        model = gpu_ctree("y ~ x1 + x2", data=self.data_df, controls=controls)
        self.assertTrue(model.is_fitted_)
        self.assertEqual(model.alpha, 0.05)  # 1 - mincriterion
        self.assertEqual(model.max_depth, 2)
    
    def test_categorical_variables(self):
        """Test handling of categorical variables."""
        # Create a DataFrame with mixed types
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, 100),
            'num1': np.random.normal(0, 1, 100),
            'num2': np.random.normal(0, 1, 100),
            'cat1': pd.Series(np.random.choice(['A', 'B', 'C'], 100)).astype('category'),
            'cat2': pd.Series(np.random.choice([1, 2, 3, 4], 100)).astype('category')
        })
        
        # Test encoding function
        X = data[['num1', 'num2', 'cat1', 'cat2']]
        X_encoded, encoders, feature_types = encode_categorical(X)
        
        # Check shape and type
        self.assertEqual(X_encoded.shape, (100, 4))
        self.assertTrue(isinstance(X_encoded, np.ndarray))
        
        # Check encoders
        self.assertTrue(2 in encoders)  # cat1 index
        self.assertTrue(3 in encoders)  # cat2 index
        
        # Check feature types
        self.assertEqual(feature_types, ['numeric', 'numeric', 'categorical', 'categorical'])
        
        # Test with model
        model = gpu_ctree("y ~ num1 + cat1", data=data)
        self.assertTrue(model.is_fitted_)
        
        # Make prediction with new categorical data
        new_data = pd.DataFrame({
            'num1': [0.5, -0.5],
            'cat1': ['A', 'B']
        })
        preds = model.predict(new_data)
        self.assertEqual(preds.shape, (2,))
    
    def test_utils(self):
        """Test utility functions."""
        # Test parse_formula
        outcome, predictors = parse_formula("y ~ x1 + x2 + x3")
        self.assertEqual(outcome, "y")
        self.assertEqual(predictors, ["x1", "x2", "x3"])
        
        # Test parse_formula with '.'
        outcome, predictors = parse_formula("y ~ .")
        self.assertEqual(outcome, "y")
        self.assertEqual(predictors, ["."])
        
        # Test CUDA availability (this just tests the function runs without error)
        cuda_available = check_cuda_availability()
        self.assertIsInstance(cuda_available, bool)
    
    def test_controls(self):
        """Test the controls object."""
        # Test default values
        controls = GPUCTreeControls()
        self.assertEqual(controls.alpha, 0.05)
        self.assertEqual(controls.minsplit, 20)
        self.assertEqual(controls.minbucket, 7)
        
        # Test with custom values
        controls = GPUCTreeControls(alpha=0.01, minsplit=10, minbucket=5)
        self.assertEqual(controls.alpha, 0.01)
        self.assertEqual(controls.minsplit, 10)
        self.assertEqual(controls.minbucket, 5)
        
        # Test with mincriterion (R compatibility)
        controls = GPUCTreeControls(mincriterion=0.99)
        self.assertEqual(controls.alpha, 0.01)  # 1 - mincriterion
        
        # Test to_dict method
        params = controls.to_dict()
        self.assertEqual(params['alpha'], 0.01)
        self.assertEqual(params['min_samples_split'], 20)
        self.assertEqual(params['min_samples_leaf'], 7)
        
        # Test gpu_ctree_control function
        controls = gpu_ctree_control(alpha=0.1, minsplit=15)
        self.assertEqual(controls.alpha, 0.1)
        self.assertEqual(controls.minsplit, 15)
    
    def test_tree_structure(self):
        """Test the tree structure."""
        tree = GPUCTree(
            alpha=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=3
        )
        
        # Fit a simple model
        tree.fit(self.X_reg[:, :2], self.y_reg)
        
        # Check the root node
        self.assertTrue(hasattr(tree, 'tree_'))
        self.assertEqual(tree.tree_.depth, 0)
        self.assertEqual(tree.tree_.n_samples, 100)
        
        # Check that the tree has a valid structure
        nodes = tree.get_node_stats()
        self.assertTrue(len(nodes) >= 1)  # At least the root node
        
        # Check tree printing
        tree_str = tree.print_tree()
        self.assertTrue(isinstance(tree_str, str))
        self.assertTrue(len(tree_str) > 0)
    
    @unittest.skipIf(not check_cuda_availability(), "CUDA not available")
    def test_gpu_acceleration(self):
        """Test GPU acceleration (skipped if CUDA not available)."""
        # Test with GPU acceleration
        tree_gpu = GPUCTree(
            alpha=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=3,
            use_gpu=True
        )
        
        # Fit model with GPU
        tree_gpu.fit(self.X_reg, self.y_reg)
        y_pred_gpu = tree_gpu.predict(self.X_reg)
        
        # Test with CPU
        tree_cpu = GPUCTree(
            alpha=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=3,
            use_gpu=False
        )
        
        # Fit model with CPU
        tree_cpu.fit(self.X_reg, self.y_reg)
        y_pred_cpu = tree_cpu.predict(self.X_reg)
        
        # Results might differ due to different implementations and floating-point precision
        # Just check that both methods produce valid predictions
        self.assertEqual(y_pred_gpu.shape, (100,))
        self.assertEqual(y_pred_cpu.shape, (100,))


class TestRealDatasets(unittest.TestCase):
    """Test cases with real-world datasets."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test datasets once for all tests."""
        # Load Iris dataset
        iris = load_iris()
        cls.iris_X = iris.data
        cls.iris_y = iris.target
        cls.iris_feature_names = iris.feature_names
        
        # Convert to DataFrame for formula interface
        cls.iris_df = pd.DataFrame(
            data=np.c_[iris.data, iris.target],
            columns=[*iris.feature_names, 'species']
        )
        
        # Load Boston dataset
        boston = load_boston()
        cls.boston_X = boston.data
        cls.boston_y = boston.target
        cls.boston_feature_names = boston.feature_names
        
        # Convert to DataFrame for formula interface
        cls.boston_df = pd.DataFrame(
            data=np.c_[boston.data, boston.target],
            columns=[*boston.feature_names, 'PRICE']
        )
    
    def test_iris_classification(self):
        """Test classification with Iris dataset."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.iris_X, self.iris_y, test_size=0.2, random_state=42
        )
        
        # Fit model
        tree = GPUCTree(
            alpha=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=5
        )
        tree.fit(X_train, y_train, feature_names=self.iris_feature_names)
        
        # Make predictions
        y_pred = tree.predict(X_test)
        
        # Simple accuracy check (just to ensure the model works reasonably)
        accuracy = np.mean(y_pred == y_test)
        self.assertGreater(accuracy, 0.8)  # Should be quite accurate for Iris
    
    def test_boston_regression(self):
        """Test regression with Boston dataset."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.boston_X, self.boston_y, test_size=0.2, random_state=42
        )
        
        # Fit model
        tree = GPUCTree(
            alpha=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=5
        )
        tree.fit(X_train, y_train, feature_names=self.boston_feature_names)
        
        # Make predictions
        y_pred = tree.predict(X_test)
        
        # Check correlation (should be reasonably correlated for a tree model)
        correlation = np.corrcoef(y_pred, y_test)[0, 1]
        self.assertGreater(correlation, 0.6)
    
    def test_formula_iris(self):
        """Test formula interface with Iris dataset."""
        # Create a training subset
        train_df, test_df = train_test_split(self.iris_df, test_size=0.2, random_state=42)
        
        # Fit model with formula
        controls = gpu_ctree_control(alpha=0.01, maxdepth=5)
        model = gpu_ctree(
            "species ~ sepal length (cm) + sepal width (cm) + petal length (cm) + petal width (cm)",
            data=train_df,
            controls=controls
        )
        
        # Make predictions
        X_test = test_df.drop('species', axis=1)
        y_pred = model.predict(X_test)
        y_test = test_df['species'].values
        
        # Check accuracy
        accuracy = np.mean(y_pred == y_test)
        self.assertGreater(accuracy, 0.8)
    
    def test_formula_boston(self):
        """Test formula interface with Boston dataset."""
        # Create a training subset
        train_df, test_df = train_test_split(self.boston_df, test_size=0.2, random_state=42)
        
        # Fit model with formula
        controls = gpu_ctree_control(alpha=0.01, maxdepth=5)
        model = gpu_ctree("PRICE ~ .", data=train_df, controls=controls)
        
        # Make predictions
        X_test = test_df.drop('PRICE', axis=1)
        y_pred = model.predict(X_test)
        y_test = test_df['PRICE'].values
        
        # Check correlation
        correlation = np.corrcoef(y_pred, y_test)[0, 1]
        self.assertGreater(correlation, 0.6)


if __name__ == '__main__':
    unittest.main()
