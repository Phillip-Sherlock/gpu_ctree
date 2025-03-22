from setuptools import setup, find_packages

setup(
    name="gpu_ctree",
    version="0.1.0",
    packages=find_packages(),
    description="GPU-Accelerated Conditional Inference Trees with Look-Ahead",
    author="Phillip Sherlock",
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "gpu": ["cupy-cuda116", "numba"],
        "viz": ["matplotlib", "graphviz"],
    }
)
