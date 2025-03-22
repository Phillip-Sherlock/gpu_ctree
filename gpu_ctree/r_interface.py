"""
R-style interface for GPU-accelerated Conditional Inference Trees.

This module provides functions that mirror the syntax of the 'party' package in R,
making it easier for R users to transition to the Python implementation.
"""

def gpu_ctree(formula, data, weights=None, subset=None, controls=None, **kwargs):
    """
    Fit a GPU-accelerated conditional inference tree using R-style formula interface.
    
    This function mimics the syntax of ctree() from the 'party' package in R.
    
    Parameters
    ----------
    formula : str
        Model formula as a string, e.g., "y ~ x1 + x2 + x3".
    data : pandas.DataFrame
        Data frame containing the variables in the formula.
    weights : array-like, optional
        Optional case weights.
    subset : array-like, optional
        Optional subset of observations to be used.
    controls : GPUCTreeControls, optional
        Optional tree controls created by gpu_ctree_control().
    **kwargs : 
        Additional arguments passed to GPUCTree.
        
    Returns
    -------
    fitted_model : GPUCTree
        Fitted tree model.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from gpu_ctree import gpu_ctree, gpu_ctree_control
    >>> 
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     "y": [0, 1, 0, 1, 0],
    ...     "x1": [1.2, 2.3, 1.1, 5.2, 1.0],
    ...     "x2": [0.5, 0.2, 0.9, 0.8, 0.4]
    ... })
    >>> 
    >>> # Define controls
    >>> controls = gpu_ctree_control(mincriterion=0.95, minsplit=10)
    >>> 
    >>> # Fit the model
    >>> model = gpu_ctree("y ~ x1 + x2", data=data, controls=controls)
    >>> print(model)
    """
    from .core import GPUCTree
    
    # Create the model
    model = GPUCTree(controls=controls, **kwargs)
    
    # Fit using formula interface
    return model.from_formula(formula=formula, data=data, subset=subset, weights=weights)


def predict_gpu_ctree(object, newdata=None, type="response"):
    """
    Predict responses from a fitted GPUCTree model.
    
    This function mimics the syntax of predict.ctree() from the 'party' package in R.
    
    Parameters
    ----------
    object : GPUCTree
        A fitted GPUCTree model.
    newdata : pandas.DataFrame, optional
        New data for prediction. If None, predictions are made on the training data.
    type : str, default="response"
        Type of prediction. Currently only "response" is supported.
        
    Returns
    -------
    predictions : ndarray
        Predicted responses.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from gpu_ctree import gpu_ctree, predict_gpu_ctree
    >>> 
    >>> # Create and fit model
    >>> data = pd.DataFrame({
    ...     "y": [0, 1, 0, 1, 0],
    ...     "x1": [1.2, 2.3, 1.1, 5.2, 1.0],
    ...     "x2": [0.5, 0.2, 0.9, 0.8, 0.4]
    ... })
    >>> model = gpu_ctree("y ~ x1 + x2", data=data)
    >>> 
    >>> # Make predictions
    >>> new_data = pd.DataFrame({
    ...     "x1": [2.0, 3.0],
    ...     "x2": [0.6, 0.7]
    ... })
    >>> predictions = predict_gpu_ctree(model, new_data)
    >>> print(predictions)
    """
    from .core import GPUCTree
    
    if not isinstance(object, GPUCTree):
        raise TypeError("object must be a fitted GPUCTree model")
    
    if type != "response":
        raise ValueError("Currently only type='response' is supported")
    
    # Make predictions
    return object.predict(newdata)


def plot_gpu_ctree(x, type="auto", **kwargs):
    """
    Plot a fitted GPUCTree model.
    
    This function mimics the syntax of plot.ctree() from the 'party' package in R.
    
    Parameters
    ----------
    x : GPUCTree
        A fitted GPUCTree model.
    type : str, default="auto"
        Type of plot. Options are "simple" or "fancy" or "auto".
    **kwargs : 
        Additional arguments passed to the plot method.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from gpu_ctree import gpu_ctree, plot_gpu_ctree
    >>> 
    >>> # Create and fit model
    >>> data = pd.DataFrame({
    ...     "y": [0, 1, 0, 1, 0],
    ...     "x1": [1.2, 2.3, 1.1, 5.2, 1.0],
    ...     "x2": [0.5, 0.2, 0.9, 0.8, 0.4]
    ... })
    >>> model = gpu_ctree("y ~ x1 + x2", data=data)
    >>> 
    >>> # Plot the tree
    >>> plot_gpu_ctree(model)
    """
    from .core import GPUCTree
    
    if not isinstance(x, GPUCTree):
        raise TypeError("x must be a fitted GPUCTree model")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for plotting but not installed")
    
    # Create plot based on type
    if type == "simple" or (type == "auto" and x._get_max_depth() > 5):
        # Use text-based representation for complex trees
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, x.print_tree(), 
                ha='center', va='center', 
                fontfamily='monospace', fontsize=9,
                transform=ax.transAxes)
        ax.axis('off')
    else:
        # Use graphical representation for simpler trees
        ax = x.plot(**kwargs)
    
    plt.tight_layout()
    plt.show()
    
    return ax
