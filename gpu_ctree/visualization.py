"""
Visualization utilities for GPU-CTree package.
"""

import numpy as np
from typing import Optional, Any, Dict, List, Union
import warnings


def export_tree_to_dot(tree, feature_names=None, class_names=None, out_file=None):
    """
    Export a tree to DOT format for visualization with Graphviz.
    
    Parameters
    ----------
    tree : GPUCTree
        Fitted tree model.
    feature_names : list of str, optional
        Names of features.
    class_names : list of str, optional
        Names of classes/outcomes.
    out_file : str, optional
        File path to write the DOT content. If None, returns the content as a string.
        
    Returns
    -------
    dot_content : str or None
        DOT content as a string if out_file is None, otherwise None.
    """
    return tree.export_graphviz(
        out_file=out_file,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=False,
        precision=3
    )


def plot_feature_importance(tree, feature_names=None, max_features=10, ax=None, figsize=(10, 6)):
    """
    Plot feature importance from a fitted tree.
    
    Parameters
    ----------
    tree : GPUCTree
        Fitted tree model.
    feature_names : list of str, optional
        Names of features.
    max_features : int, default=10
        Maximum number of features to display.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
    figsize : tuple, default=(10, 6)
        Figure size.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for plotting but not installed")
    
    if not hasattr(tree, 'feature_importances_'):
        raise ValueError("Tree does not have feature_importances_ attribute. Is it fitted?")
    
    feature_names = feature_names or tree.feature_names_in_
    importances = tree.feature_importances_
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    
    # Limit to max_features if needed
    if max_features is not None and max_features < len(indices):
        indices = indices[:max_features]
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance')
    
    return ax


def plot_tree_structure(tree, max_depth=None, figsize=(12, 8)):
    """
    Plot the structure of the tree.
    
    Parameters
    ----------
    tree : GPUCTree
        Fitted tree model.
    max_depth : int, optional
        Maximum depth to display.
    figsize : tuple, default=(12, 8)
        Figure size.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError("Matplotlib and NetworkX are required for plotting but not installed")
    
    if not hasattr(tree, 'tree_'):
        raise ValueError("Tree does not have tree_ attribute. Is it fitted?")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Function to recursively add nodes and edges
    def add_node(node, node_id=0, depth=0):
        if max_depth is not None and depth > max_depth:
            return
        
        # Add node
        if node.is_leaf:
            label = f"Leaf\n(n={node.n_samples})"
            G.add_node(node_id, label=label, type='leaf')
        else:
            feature_name = tree.feature_names_in_[node.split_feature]
            label = f"{feature_name} ≤ {node.split_value:.3f}\n(p={node.p_value:.3f})"
            G.add_node(node_id, label=label, type='internal')
            
            # Add children
            left_id = 2 * node_id + 1
            right_id = 2 * node_id + 2
            
            # Add edges
            G.add_edge(node_id, left_id, label="≤")
            G.add_edge(node_id, right_id, label=">")
            
            # Add child nodes
            add_node(node.left, left_id, depth + 1)
            add_node(node.right, right_id, depth + 1)
    
    # Build the graph
    add_node(tree.tree_)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set positions using a hierarchical layout
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Draw nodes
    node_colors = ['lightblue' if G.nodes[n]['type'] == 'leaf' else 'lightgray' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2000, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    # Draw edge labels
    edge_labels = {(u, v): G.edges[u, v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    ax.set_title('Tree Structure')
    ax.axis('off')
    
    return fig
