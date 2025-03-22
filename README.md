# GPU-CTree: GPU-Accelerated Conditional Inference Trees with Look-Ahead

GPU-CTree is a Python package that implements conditional inference trees with GPU acceleration, designed specifically for high-performance computing environments like HiPerGator. It provides both scikit-learn compatible and R-style interfaces.

## Features

- **GPU Acceleration**: Utilize CUDA-enabled GPUs for faster computation
- **R-Style Interface**: Familiar formula interface for R users transitioning to Python
- **Look-Ahead Algorithm**: More globally optimal splits by evaluating multiple steps ahead
- **Statistical Testing**: Thorough independence tests with multiple testing correction
- **Multivariate Outcomes**: Support for both univariate and multivariate outcome variables
- **Visualization**: Comprehensive tree visualization capabilities

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA Toolkit 11.6+ (for GPU acceleration)

### Basic Installation

```bash
pip install gpu-ctree
