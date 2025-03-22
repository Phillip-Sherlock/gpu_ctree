"""
Custom exceptions for the gpu_ctree package.
"""

class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class GPUMemoryError(MemoryError):
    """Raised when GPU memory is insufficient for the operation."""
    pass

class FormulaError(Exception):
    """Raised when there's an issue with formula parsing."""
    pass

class FittingError(Exception):
    """Raised when model fitting fails."""
    pass
