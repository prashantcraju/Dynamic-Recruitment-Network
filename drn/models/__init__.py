# drn/models/__init__.py
"""
Complete model architectures for Dynamic Recruitment Networks.

This module provides high-level model classes that combine multiple DRN layers
into complete neural network architectures.
"""

from .drn_network import DRNNetwork, create_standard_drn

__all__ = ["DRNNetwork", "create_standard_drn"]
