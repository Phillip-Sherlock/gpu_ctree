import pandas as pd
import numpy as np
from gpu_ctree import GPUCTree, check_cuda_availability

print(f"CUDA available: {check_cuda_availability()}")

# Create sample data
X = np.random.normal(size=(1000, 5))
y = (X[:, 0] > 0).astype(int)

# Train model
model = GPUCTree(use_gpu=True)
model.fit(X, y)

# Make prediction
pred = model.predict(X[:5])
print(f"Predictions: {pred}")
