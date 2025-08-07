# drn/core/__init__.py
"""
Core components of Dynamic Recruitment Networks.

These modules implement the fundamental building blocks that distinguish DRN
from traditional neural networks:
- BasePopulation: Maintains active neural ensembles
- BudgetManager: Manages neurotransmitter-like resource allocation  
- NeuronRecruitment: Dynamically selects neurons for activation
- FeedbackManager: Handles recurrent population feedback
"""

from .base_population import BasePopulation
from .budget_manager import BudgetManager, AdaptiveBudgetManager
from .recruitment import NeuronRecruitment, ContextualRecruitment
from .feedback import FeedbackManager, HierarchicalFeedbackManager

__all__ = [
    "BasePopulation",
    "BudgetManager", 
    "AdaptiveBudgetManager",
    "NeuronRecruitment",
    "ContextualRecruitment", 
    "FeedbackManager",
    "HierarchicalFeedbackManager"
]
