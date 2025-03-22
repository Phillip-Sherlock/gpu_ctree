"""
Control parameters for GPU-accelerated Conditional Inference Trees.

This module provides configuration objects similar to R's ctree_control()
for fine-tuning the behavior of the GPUCTree algorithm.
"""

class GPUCTreeControls:
    """
    Control parameters for the GPUCTree algorithm.
    
    This class mirrors the behavior of ctree_control() in R's party package,
    allowing fine-grained control over the tree-building process.
    
    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for variable selection.
    mincriterion : float, default=None
        1 - alpha, for compatibility with R's interface. If provided, overrides alpha.
    minsplit : int, default=20
        Minimum number of observations in a node to be considered for splitting.
    minbucket : int, default=7
        Minimum number of observations in a terminal node.
    stump : bool, default=False
        Whether to fit a stump (tree with only one split) or a full tree.
    maxdepth : int, default=30
        Maximum depth of the tree.
    mtry : int or None, default=None
        Number of variables randomly sampled as candidates at each split.
        If None, all variables are used.
    lookahead : int, default=0
        Number of look-ahead steps for more globally optimal splits.
    nresample : int, default=9999
        Number of Monte-Carlo replications for the computation of p-values.
    testtype : str, default="Univariate"
        Type of independence test to be applied. Options are "Univariate" or "Teststatistic".
    teststat : str, default="quadratic"
        Test statistic to be applied. Options are "quadratic", "maximum", or "correlation".
    use_gpu : bool, default=True
        Whether to use GPU acceleration when available.
    random_state : int or None, default=None
        Random seed for reproducible results.
    """
    
    def __init__(
        self,
        alpha=0.05,
        mincriterion=None,
        minsplit=20,
        minbucket=7,
        stump=False,
        maxdepth=30,
        mtry=None,
        lookahead=0,
        nresample=9999,
        testtype="Univariate",
        teststat="quadratic",
        use_gpu=True,
        random_state=None
    ):
        # Store all parameters
        self.alpha = alpha
        
        # For R compatibility: mincriterion = 1 - alpha
        if mincriterion is not None:
            self.alpha = 1 - mincriterion
        
        self.minsplit = minsplit
        self.minbucket = minbucket
        self.stump = stump
        self.maxdepth = 1 if stump else maxdepth
        self.mtry = mtry
        self.lookahead = lookahead
        self.nresample = nresample
        
        # Validate and store test parameters
        if testtype not in ["Univariate", "Teststatistic"]:
            raise ValueError("testtype must be 'Univariate' or 'Teststatistic'")
        self.testtype = testtype
        
        if teststat not in ["quadratic", "maximum", "correlation"]:
            raise ValueError("teststat must be 'quadratic', 'maximum', or 'correlation'")
        self.teststat = teststat
        
        self.use_gpu = use_gpu
        self.random_state = random_state
    
    def to_dict(self):
        """Convert the controls to a dictionary of parameters for GPUCTree."""
        return {
            'alpha': self.alpha,
            'min_samples_split': self.minsplit,
            'min_samples_leaf': self.minbucket,
            'max_depth': self.maxdepth,
            'max_features': self.mtry,
            'lookahead_depth': self.lookahead,
            'n_permutations': self.nresample,
            'test_type': self.testtype,
            'test_statistic': self.teststat,
            'use_gpu': self.use_gpu,
            'random_state': self.random_state
        }
    
    def __str__(self):
        """Return a string representation of the controls object."""
        params = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"GPUCTreeControls({', '.join(params)})"


def gpu_ctree_control(**kwargs):
    """
    Create a control parameter object for GPUCTree.
    
    This function mirrors the syntax of ctree_control() in R.
    
    Returns
    -------
    GPUCTreeControls
        A control parameter object that can be passed to GPUCTree.
    
    Examples
    --------
    >>> controls = gpu_ctree_control(alpha=0.01, maxdepth=5, minbucket=10)
    >>> model = gpu_ctree("y ~ x1 + x2", data=data, controls=controls)
    """
    return GPUCTreeControls(**kwargs)
