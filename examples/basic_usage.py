"""
Basic usage examples for the GPU-CTree package.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from gpu_ctree import GPUCTree, gpu_ctree, gpu_ctree_control
from gpu_ctree.utils import check_cuda_availability


def regression_example():
    """
    Example of using GPU-CTree for regression.
    """
    print("Regression Example using scikit-learn API")
    print("----------------------------------------")
    
    # Check if CUDA is available
    print(f"CUDA available: {check_cuda_availability()}")
    
    # Load Boston Housing dataset
    boston = load_boston()
    X, y = boston.data, boston.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit the model
    model = GPUCTree(
        alpha=0.01,
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=5
    )
    
    print("Fitting the model...")
    model.fit(X_train, y_train, feature_names=boston.feature_names)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Print feature importance
    importances = model.feature_importances_
    print("\nFeature Importance:")
    for i, (name, importance) in enumerate(zip(boston.feature_names, importances)):
        print(f"{name}: {importance:.4f}")
    
    print("\nTree Structure:")
    print(model.print_tree())


def classification_example():
    """
    Example of using GPU-CTree for classification.
    """
    print("\nClassification Example using R-style API")
    print("--------------------------------------")
    
    # Load Iris dataset
    iris = load_iris()
    
    # Create a pandas DataFrame
    data = pd.DataFrame(
        data=np.c_[iris.data, iris.target],
        columns=[*iris.feature_names, 'species']
    )
    
    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create controls object (R-style)
    controls = gpu_ctree_control(
        mincriterion=0.99,  # Equivalent to alpha=0.01
        minsplit=10,
        minbucket=5,
        maxdepth=5
    )
    
    print("Fitting the model using R-style formula interface...")
    # Fit the model using R-style formula interface
    model = gpu_ctree(
        formula="species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width",
        data=train_data,
        controls=controls
    )
    
    # Make predictions
    print("Making predictions...")
    test_X = test_data.drop('species', axis=1)
    test_y = test_data['species']
    
    y_pred = model.predict(test_X)
    
    # Evaluate the model
    accuracy = accuracy_score(test_y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print tree structure
    print("\nTree Structure:")
    print(model)


if __name__ == "__main__":
    # Run examples
    regression_example()
    classification_example()
    
    print("\nAdditional examples of the R-style interface:")
    print("---------------------------------------------")
    
    # Create a simple dataset
    data = pd.DataFrame({
        'y': np.random.normal(0, 1, 100),
        'x1': np.random.normal(0, 1, 100),
        'x2': np.random.normal(0, 1, 100),
        'x3': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Example 1: Basic usage
    print("\nExample 1: Basic usage")
    model1 = gpu_ctree("y ~ x1 + x2 + x3", data=data)
    print(model1)
    
    # Example 2: Using all predictors with '.'
    print("\nExample 2: Using all predictors with '.'")
    model2 = gpu_ctree("y ~ .", data=data)
    print(model2)
    
    # Example 3: Using custom controls
    print("\nExample 3: Using custom controls")
    controls = gpu_ctree_control(maxdepth=2, minbucket=10)
    model3 = gpu_ctree("y ~ x1 + x2", data=data, controls=controls)
    print(model3)
    
    # Example 4: Using subset of data
    print("\nExample 4: Using subset of data")
    subset = data['x1'] > 0
    model4 = gpu_ctree("y ~ x1 + x2", data=data, subset=subset)
    print(model4)
