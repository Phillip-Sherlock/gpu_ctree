"""
GPU-CTree example using Boston Housing dataset with multi-GPU and look-ahead.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
import os
import sys

# Import utility functions first
from gpu_ctree.utils import check_cuda_availability

# Then import the model
from gpu_ctree import GPUCTree, gpu_ctree_control

# Print environment information
print("Environment Information:")
print(f"- GPU Available: {check_cuda_availability()}")

# Try to get CUDA and CuPy information
try:
    import cupy
    print(f"- CuPy Version: {cupy.__version__}")
    cuda_version = cupy.cuda.runtime.runtimeGetVersion()
    major = cuda_version // 1000
    minor = (cuda_version % 1000) // 10
    print(f"- CUDA Version: {major}.{minor}")
    
    # Test importing the GPU functions
    try:
        from gpu_ctree.kernels import gpu_permutation_test, gpu_compute_split_criterion, gpu_compute_node_statistics
        print("- GPU functions successfully imported")
    except ImportError as e:
        print(f"- GPU functions import error: {str(e)}")
except ImportError:
    print("- CuPy not available")

# Get Python and NumPy versions
import numpy as np
print(f"- Python Version: {sys.version.split()[0]}")
print(f"- NumPy Version: {np.__version__}")
print(f"- Using {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')} GPUs")

# Load California Housing dataset (replacement for Boston which is deprecated)
print("\nLoading California Housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {', '.join(housing.feature_names)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# Configure the model with look-ahead
controls = gpu_ctree_control(
    alpha=0.05,           # Significance level
    lookahead=2,          # Look-ahead depth of 2
    minsplit=20,          # Minimum split size
    minbucket=7,          # Minimum bucket size
    maxdepth=10           # Maximum tree depth
)

# Create and train the model
print("\nTraining model with look-ahead...")
start_time = time.time()
model = GPUCTree(
    controls=controls,
    use_gpu=True,
    n_permutations=999,   # Number of permutations for p-value
    random_state=42
)

# Fit the model
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate the model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
print("\nFeature Importance:")
feature_importance = {}
for i, feature in enumerate(model.feature_names_in_):
    feature_importance[feature] = model.feature_importances_[i]
    print(f"- {feature}: {model.feature_importances_[i]:.4f}")

# Save model and detailed results as text
import pickle

# Save the trained model
model_filename = "boston_housing_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to {model_filename}")

# Save detailed results as text
with open("boston_housing_results.txt", "w") as f:
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n\n")
    f.write("Feature Importance:\n")
    for i, feature in enumerate(model.feature_names_in_):
        importance = model.feature_importances_[i]
        f.write(f"- {feature}: {importance:.4f}\n")

# Save the model visualization if possible
try:
    from gpu_ctree.visualization import plot_tree, plot_feature_importance
    import matplotlib.pyplot as plt
    
    print("\nGenerating visualizations...")
    
    # Plot the tree
    plt.figure(figsize=(20, 10))
    plot_tree(model)
    plt.tight_layout()
    plt.savefig("boston_housing_tree.png")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plot_feature_importance(model)
    plt.tight_layout()
    plt.savefig("boston_housing_importance.png")
    
    print("Visualizations saved to current directory.")
except Exception as e:
    print(f"Could not generate visualizations: {str(e)}")

print("\nExample completed successfully!")