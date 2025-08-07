"""
drn/utils/config.py

Configuration management for Dynamic Recruitment Networks.
This module provides configuration classes and utilities for DRN experiments,
including model configurations, training parameters, and research settings.
"""

import yaml
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class LayerConfig:
    """Configuration for a single DRN layer."""
    base_population_size: int = 32
    neuron_pool_size: int = 64
    output_size: int = 16
    initial_budget: float = 50.0
    budget_decay_rate: float = 0.01
    recruitment_threshold: float = 0.1
    population_activation: str = 'relu'
    recurrent_strength: float = 0.1
    inhibition_strength: float = 0.05
    adaptation_rate: float = 0.01
    noise_level: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for complete DRN model."""
    input_size: int = 64
    output_size: int = 10
    layer_configs: List[LayerConfig] = field(default_factory=lambda: [
        LayerConfig(base_population_size=32, neuron_pool_size=64, output_size=16),
        LayerConfig(base_population_size=16, neuron_pool_size=32, output_size=8)
    ])
    global_budget_decay: float = 0.0
    inter_layer_connections: bool = False
    use_global_feedback: bool = True
    output_activation: str = 'none'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        # Handle layer configs separately
        layer_configs = []
        if 'layer_configs' in config_dict:
            for layer_dict in config_dict['layer_configs']:
                if isinstance(layer_dict, dict):
                    layer_configs.append(LayerConfig(**layer_dict))
                else:
                    layer_configs.append(layer_dict)
            config_dict['layer_configs'] = layer_configs
        
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Configuration for training DRN models."""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = 'adam'
    weight_decay: float = 0.0001
    
    # DRN-specific training parameters
    recruitment_loss_weight: float = 0.1
    flexibility_loss_weight: float = 0.05
    connectivity_loss_weight: float = 0.02
    budget_regularization: float = 0.01
    
    # Training schedule
    budget_decay_schedule: str = 'linear'  # 'linear', 'exponential', 'step'
    recruitment_warmup_epochs: int = 10
    flexibility_ramp_epochs: int = 20
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "drn_experiment"
    description: str = "Dynamic Recruitment Network experiment"
    
    # Model and training configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data configuration
    dataset_type: str = 'concept_learning'  # 'concept_learning', 'task_switching', 'interference'
    data_params: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis configuration
    track_connectivity: bool = True
    track_flexibility: bool = True
    analysis_frequency: int = 10
    
    # Research-specific settings
    autism_modeling: bool = False
    baseline_comparison: bool = True
    baseline_models: List[str] = field(default_factory=lambda: ['mlp', 'lstm'])
    
    # Output settings
    save_results: bool = True
    output_dir: str = "results"
    plot_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create ExperimentConfig from dictionary."""
        # Handle nested configs
        if 'model' in config_dict and isinstance(config_dict['model'], dict):
            config_dict['model'] = ModelConfig.from_dict(config_dict['model'])
        
        if 'training' in config_dict and isinstance(config_dict['training'], dict):
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        
        return cls(**config_dict)


# Legacy class for backward compatibility
class DRNConfig:
    """Legacy configuration class for backward compatibility."""
    
    def __init__(self, **kwargs):
        # Set default values
        self.input_size = kwargs.get('input_size', 64)
        self.output_size = kwargs.get('output_size', 10)
        self.layer_configs = kwargs.get('layer_configs', [
            {'base_population_size': 32, 'neuron_pool_size': 64, 'output_size': 16}
        ])
        
        # Set any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class ConfigManager:
    """Configuration manager for DRN experiments."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: ExperimentConfig, filename: str) -> None:
        """Save configuration to file."""
        filepath = self.config_dir / filename
        
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        elif filename.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """Load configuration from file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filename.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
        
        return ExperimentConfig.from_dict(config_dict)
    
    def create_default_configs(self) -> None:
        """Create default configuration files."""
        configs = {
            'default.yaml': get_standard_config('medium'),
            'concept_learning.yaml': get_concept_learning_config(),
            'flexibility_test.yaml': get_flexibility_config(),
            'autism_comparison.yaml': get_autism_comparison_config()
        }
        
        for filename, config in configs.items():
            self.save_config(config, filename)
        
        print(f"Created default configs in {self.config_dir}")


def get_standard_config(size: str = 'medium') -> ExperimentConfig:
    """Get standard DRN configuration."""
    
    if size == 'small':
        model_config = ModelConfig(
            input_size=32,
            output_size=4,
            layer_configs=[
                LayerConfig(base_population_size=16, neuron_pool_size=32, output_size=8)
            ]
        )
        training_config = TrainingConfig(batch_size=16, num_epochs=50)
        
    elif size == 'medium':
        model_config = ModelConfig(
            input_size=64,
            output_size=10,
            layer_configs=[
                LayerConfig(base_population_size=32, neuron_pool_size=64, output_size=16),
                LayerConfig(base_population_size=16, neuron_pool_size=32, output_size=8)
            ]
        )
        training_config = TrainingConfig(batch_size=32, num_epochs=100)
        
    elif size == 'large':
        model_config = ModelConfig(
            input_size=128,
            output_size=20,
            layer_configs=[
                LayerConfig(base_population_size=64, neuron_pool_size=128, output_size=32),
                LayerConfig(base_population_size=32, neuron_pool_size=64, output_size=16),
                LayerConfig(base_population_size=16, neuron_pool_size=32, output_size=8)
            ]
        )
        training_config = TrainingConfig(batch_size=64, num_epochs=150)
    
    else:
        raise ValueError("Size must be 'small', 'medium', or 'large'")
    
    return ExperimentConfig(
        name=f"standard_{size}",
        description=f"Standard {size} DRN configuration",
        model=model_config,
        training=training_config
    )


def get_concept_learning_config() -> ExperimentConfig:
    """Get configuration optimized for concept learning experiments."""
    model_config = ModelConfig(
        input_size=64,
        output_size=8,  # More concepts
        layer_configs=[
            LayerConfig(
                base_population_size=32, 
                neuron_pool_size=64, 
                output_size=16,
                recruitment_threshold=0.05,  # Lower threshold for concept formation
                adaptation_rate=0.02  # Faster adaptation
            ),
            LayerConfig(
                base_population_size=16, 
                neuron_pool_size=32, 
                output_size=8,
                recruitment_threshold=0.1,
                adaptation_rate=0.01
            )
        ],
        use_global_feedback=True,
        inter_layer_connections=True
    )
    
    training_config = TrainingConfig(
        learning_rate=0.0005,  # Slower learning for stable concepts
        batch_size=32,
        num_epochs=150,
        recruitment_loss_weight=0.15,  # Higher recruitment regularization
        flexibility_loss_weight=0.1
    )
    
    return ExperimentConfig(
        name="concept_learning",
        description="Configuration for concept learning experiments",
        model=model_config,
        training=training_config,
        dataset_type='concept_learning',
        data_params={
            'num_concepts': 8,
            'concept_complexity': 'medium',
            'hierarchical': True,
            'samples_per_concept': 200
        }
    )


def get_flexibility_config() -> ExperimentConfig:
    """Get configuration optimized for cognitive flexibility experiments."""
    model_config = ModelConfig(
        input_size=64,
        output_size=4,
        layer_configs=[
            LayerConfig(
                base_population_size=32,
                neuron_pool_size=64,
                output_size=16,
                budget_decay_rate=0.005,  # Slower budget decay for flexibility
                recruitment_threshold=0.08,
                recurrent_strength=0.15  # Stronger recurrence for memory
            )
        ],
        use_global_feedback=True,
        inter_layer_connections=False  # Simpler for flexibility testing
    )
    
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=16,  # Smaller batches for task switching
        num_epochs=100,
        recruitment_loss_weight=0.05,  # Lower recruitment penalty
        flexibility_loss_weight=0.2,  # Higher flexibility emphasis
        flexibility_ramp_epochs=30
    )
    
    return ExperimentConfig(
        name="flexibility_test",
        description="Configuration for cognitive flexibility experiments",
        model=model_config,
        training=training_config,
        dataset_type='task_switching',
        data_params={
            'task_types': ['classification', 'regression', 'sequence'],
            'switch_probability': 0.3,
            'sequence_length': 100
        },
        track_flexibility=True,
        analysis_frequency=5
    )


def get_autism_comparison_config() -> ExperimentConfig:
    """Get configuration for autism vs neurotypical comparison studies."""
    model_config = ModelConfig(
        input_size=64,
        output_size=4,
        layer_configs=[
            LayerConfig(
                base_population_size=32,
                neuron_pool_size=64,
                output_size=16,
                recruitment_threshold=0.12,  # Higher threshold (more rigid)
                adaptation_rate=0.005,  # Slower adaptation
                recurrent_strength=0.2,  # Stronger local connections
                inhibition_strength=0.08  # Stronger inhibition
            )
        ],
        use_global_feedback=False,  # Reduced global processing
        inter_layer_connections=False
    )
    
    training_config = TrainingConfig(
        learning_rate=0.0005,
        batch_size=32,
        num_epochs=120,
        recruitment_loss_weight=0.2,  # Higher rigidity
        flexibility_loss_weight=0.02,  # Lower flexibility
        connectivity_loss_weight=0.05
    )
    
    return ExperimentConfig(
        name="autism_comparison",
        description="Configuration for autism vs neurotypical modeling",
        model=model_config,
        training=training_config,
        dataset_type='autism_modeling',
        data_params={
            'condition': 'autism',
            'processing_style': 'local',
            'pattern_completion_bias': 0.3,
            'sensory_processing_difference': 0.3
        },
        autism_modeling=True,
        baseline_comparison=True
    )


def create_autism_comparison_configs() -> Dict[str, ExperimentConfig]:
    """Create both neurotypical and autism configurations for comparison."""
    
    # Base configuration
    base_config = get_autism_comparison_config()
    
    # Neurotypical configuration
    neurotypical_config = ExperimentConfig.from_dict(base_config.to_dict())
    neurotypical_config.name = "neurotypical"
    neurotypical_config.model.layer_configs[0].recruitment_threshold = 0.08  # Lower threshold
    neurotypical_config.model.layer_configs[0].adaptation_rate = 0.015  # Faster adaptation
    neurotypical_config.model.layer_configs[0].recurrent_strength = 0.1  # Balanced connections
    neurotypical_config.model.use_global_feedback = True  # Enhanced global processing
    neurotypical_config.training.flexibility_loss_weight = 0.1  # Higher flexibility
    neurotypical_config.data_params = {
        'condition': 'neurotypical',
        'processing_style': 'mixed',
        'pattern_completion_bias': 0.7,
        'sensory_processing_difference': 0.1
    }
    
    # Autism configuration (already created above)
    autism_config = base_config
    
    return {
        'neurotypical': neurotypical_config,
        'autism_like': autism_config
    }


def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Validate model config
    if config.model.input_size <= 0:
        issues.append("Model input_size must be positive")
    
    if config.model.output_size <= 0:
        issues.append("Model output_size must be positive")
    
    if not config.model.layer_configs:
        issues.append("Model must have at least one layer")
    
    # Validate layer configs
    for i, layer_config in enumerate(config.model.layer_configs):
        if layer_config.base_population_size <= 0:
            issues.append(f"Layer {i}: base_population_size must be positive")
        
        if layer_config.neuron_pool_size < layer_config.base_population_size:
            issues.append(f"Layer {i}: neuron_pool_size must be >= base_population_size")
        
        if not 0 <= layer_config.recruitment_threshold <= 1:
            issues.append(f"Layer {i}: recruitment_threshold must be between 0 and 1")
    
    # Validate training config
    if config.training.learning_rate <= 0:
        issues.append("Learning rate must be positive")
    
    if config.training.batch_size <= 0:
        issues.append("Batch size must be positive")
    
    if config.training.num_epochs <= 0:
        issues.append("Number of epochs must be positive")
    
    return issues


# Convenience functions for backward compatibility
def create_layer_config(**kwargs) -> Dict[str, Any]:
    """Create layer configuration dictionary."""
    layer_config = LayerConfig(**kwargs)
    return asdict(layer_config)


def create_model_config(**kwargs) -> Dict[str, Any]:
    """Create model configuration dictionary."""
    model_config = ModelConfig(**kwargs)
    return asdict(model_config)


# Configuration presets
PRESET_CONFIGS = {
    'small': get_standard_config('small'),
    'medium': get_standard_config('medium'),
    'large': get_standard_config('large'),
    'concept_learning': get_concept_learning_config(),
    'flexibility': get_flexibility_config(),
    'autism_comparison': get_autism_comparison_config()
}


def get_preset_config(preset_name: str) -> ExperimentConfig:
    """Get a preset configuration by name."""
    if preset_name not in PRESET_CONFIGS:
        available_presets = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    return PRESET_CONFIGS[preset_name]


def list_available_presets() -> List[str]:
    """List all available configuration presets."""
    return list(PRESET_CONFIGS.keys())