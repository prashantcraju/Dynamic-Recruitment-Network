# drn/__init__.py
"""
Dynamic Recruitment Networks (DRN) - A Biologically-Inspired Neural Architecture

DRN introduces dynamic neuron recruitment based on population activity and budget constraints,
moving beyond traditional 1:1 layer connectivity to create more flexible, human-like AI systems.

Quick Start:
    >>> from drn import DRNNetwork, create_drn_trainer
    >>> from drn.utils.config import get_standard_config
    >>> 
    >>> # Create model
    >>> config = get_standard_config('medium')
    >>> model = DRNNetwork.from_config(config.model)
    >>> 
    >>> # Train model
    >>> trainer = create_drn_trainer(model)
    >>> history = trainer.train(train_loader, val_loader, num_epochs=50)

Main Components:
    - DRNNetwork: Main model architecture
    - DRNTrainer: Specialized training framework
    - FlexibilityMetrics: Cognitive flexibility evaluation
    - DRNVisualizer: Analysis and visualization tools
    - ConfigManager: Configuration management
"""

__version__ = "0.1.0"
__author__ = "Prashant Raju"
__email__ = "rajuprashant@gmail.com"

# Core model components
from .models.drn_network import DRNNetwork, create_standard_drn
from .layers.drn_layer import DRNLayer

# Core building blocks
from .core.base_population import BasePopulation
from .core.budget_manager import BudgetManager, AdaptiveBudgetManager
from .core.recruitment import NeuronRecruitment, ContextualRecruitment
from .core.feedback import FeedbackManager, HierarchicalFeedbackManager

# Training framework
from .training.trainer import DRNTrainer, create_drn_trainer
from .training.losses import RecruitmentLoss, FlexibilityLoss, DRNLoss
from .training.metrics import FlexibilityMetrics, ConnectivityMetrics, ComparisonMetrics

# Analysis and visualization
from .analysis.visualization import DRNVisualizer, quick_plot_recruitment, quick_dashboard

# Configuration management
from .utils.config import (
    ConfigManager, ExperimentConfig, ModelConfig, LayerConfig, TrainingConfig,
    get_standard_config, create_autism_comparison_configs
)

# Main API exports
__all__ = [
    # Core Models
    "DRNNetwork",
    "DRNLayer", 
    "create_standard_drn",
    
    # Core Components
    "BasePopulation",
    "BudgetManager",
    "NeuronRecruitment", 
    "FeedbackManager",
    
    # Training
    "DRNTrainer",
    "create_drn_trainer",
    "RecruitmentLoss",
    "FlexibilityLoss",
    
    # Metrics and Analysis
    "FlexibilityMetrics",
    "ConnectivityMetrics",
    "ComparisonMetrics",
    "DRNVisualizer",
    
    # Configuration
    "ConfigManager",
    "ExperimentConfig",
    "get_standard_config",
    
    # Utilities
    "quick_plot_recruitment",
    "quick_dashboard",
    "create_autism_comparison_configs",
]

# Package metadata
__doc_url__ = "https://github.com/yourorg/dynamic-recruitment-networks"
__repository__ = "https://github.com/yourorg/dynamic-recruitment-networks"
__license__ = "MIT"

# Convenience functions
def create_concept_learning_experiment(input_size=64, output_size=4, config_name='medium'):
    """
    Quick setup for concept learning experiments.
    
    Args:
        input_size: Input feature dimension
        output_size: Number of concept classes
        config_name: Model configuration ('small', 'medium', 'large')
        
    Returns:
        Tuple of (model, trainer, config)
    """
    config = get_standard_config(config_name)
    config.model.input_size = input_size
    config.model.output_size = output_size
    
    model = DRNNetwork(
        input_size=config.model.input_size,
        layer_configs=[vars(lc) for lc in config.model.layer_configs],
        output_size=config.model.output_size
    )
    
    trainer = create_drn_trainer(model)
    
    return model, trainer, config

def create_autism_modeling_experiment(autism_simulation=True):
    """
    Quick setup for autism vs neurotypical modeling.
    
    Args:
        autism_simulation: If True, creates autism-like connectivity patterns
        
    Returns:
        Tuple of (model, trainer, config)
    """
    configs = create_autism_comparison_configs()
    config = configs['autism_like' if autism_simulation else 'neurotypical']
    
    model = DRNNetwork(
        input_size=config.model.input_size,
        layer_configs=[vars(lc) for lc in config.model.layer_configs],
        output_size=config.model.output_size
    )
    
    trainer = create_drn_trainer(
        model=model,
        connectivity_weight=config.training.connectivity_weight,
        flexibility_weight=config.training.flexibility_weight
    )
    
    return model, trainer, config

# Package-level configuration
def set_default_device(device):
    """Set default device for all DRN operations."""
    import torch
    if isinstance(device, str):
        device = torch.device(device)
    # This would be implemented to set a global default
    pass

def get_version_info():
    """Get detailed version information."""
    import torch
    return {
        'drn_version': __version__,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }

# Auto-configuration on import
def _setup_package():
    """Setup package defaults on import."""
    import warnings
    
    # Check for required dependencies
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available. DRN will run on CPU which may be slow for large models.")
    except ImportError:
        warnings.warn("PyTorch not found. Please install PyTorch to use DRN.")
    
    # Set up logging
    import logging
    logging.getLogger('drn').addHandler(logging.NullHandler())

# Run setup
_setup_package()
