# drn/layers/__init__.py
"""
Layer implementations for Dynamic Recruitment Networks.

This module contains the layer-level abstractions that combine core components
into usable neural network layers.
"""

from .drn_layer import DRNLayer, AdaptiveDRNLayer

__all__ = ["DRNLayer", "AdaptiveDRNLayer"]

