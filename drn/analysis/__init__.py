# drn/analysis/__init__.py
"""
Analysis and visualization tools for Dynamic Recruitment Networks.

This module provides tools for analyzing DRN behavior, visualizing connectivity
patterns, and comparing with traditional neural networks.
"""

from .visualization import (
    DRNVisualizer, 
    quick_plot_recruitment, 
    quick_plot_training,
    quick_dashboard
)

__all__ = [
    "DRNVisualizer",
    "quick_plot_recruitment",
    "quick_plot_training", 
    "quick_dashboard"
]
