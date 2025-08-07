# drn/training/__init__.py
"""
Training framework for Dynamic Recruitment Networks.

This module provides specialized training procedures, loss functions, and metrics
designed specifically for DRN architectures.
"""

from .trainer import DRNTrainer, create_drn_trainer
from .losses import RecruitmentLoss, FlexibilityLoss, DRNLoss, AdversarialFlexibilityLoss
from .metrics import FlexibilityMetrics, ConnectivityMetrics, ComparisonMetrics

__all__ = [
    "DRNTrainer",
    "create_drn_trainer", 
    "RecruitmentLoss",
    "FlexibilityLoss",
    "DRNLoss",
    "AdversarialFlexibilityLoss",
    "FlexibilityMetrics",
    "ConnectivityMetrics", 
    "ComparisonMetrics"
]
