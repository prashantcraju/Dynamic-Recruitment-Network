"""
create_configs.py

Script to create all missing configuration files for the DRN package.
This will create the configs/ directory and populate it with default YAML files.
"""

import os
import yaml
from pathlib import Path

def create_configs_directory():
    """Create the configs directory structure"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    print(f"✓ Created configs directory: {config_dir.absolute()}")
    return config_dir

def create_default_config():
    """Create default.yaml configuration"""
    config = {
        'name': 'default_medium',
        'description': 'Default medium-sized DRN configuration',
        'model': {
            'input_size': 64,
            'output_size': 10,
            'layer_configs': [
                {
                    'base_population_size': 32,
                    'neuron_pool_size': 64,
                    'output_size': 16,
                    'initial_budget': 50.0,
                    'budget_decay_rate': 0.01,
                    'recruitment_threshold': 0.1,
                    'population_activation': 'relu',
                    'recurrent_strength': 0.1,
                    'inhibition_strength': 0.05,
                    'adaptation_rate': 0.01,
                    'noise_level': 0.0
                },
                {
                    'base_population_size': 16,
                    'neuron_pool_size': 32,
                    'output_size': 8,
                    'initial_budget': 50.0,
                    'budget_decay_rate': 0.01,
                    'recruitment_threshold': 0.1,
                    'population_activation': 'relu',
                    'recurrent_strength': 0.1,
                    'inhibition_strength': 0.05,
                    'adaptation_rate': 0.01,
                    'noise_level': 0.0
                }
            ],
            'global_budget_decay': 0.0,
            'inter_layer_connections': False,
            'use_global_feedback': True,
            'output_activation': 'none'
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'recruitment_loss_weight': 0.1,
            'flexibility_loss_weight': 0.05,
            'connectivity_loss_weight': 0.02,
            'budget_regularization': 0.01,
            'budget_decay_schedule': 'linear',
            'recruitment_warmup_epochs': 10,
            'flexibility_ramp_epochs': 20,
            'early_stopping': True,
            'patience': 10,
            'min_delta': 0.001
        },
        'dataset_type': 'concept_learning',
        'data_params': {
            'num_concepts': 4,
            'concept_complexity': 'medium',
            'hierarchical': True,
            'samples_per_concept': 200
        },
        'track_connectivity': True,
        'track_flexibility': True,
        'analysis_frequency': 10,
        'autism_modeling': False,
        'baseline_comparison': True,
        'baseline_models': ['mlp', 'lstm'],
        'save_results': True,
        'output_dir': 'results',
        'plot_results': True
    }
    return config

def create_concept_learning_config():
    """Create concept_learning.yaml configuration"""
    config = {
        'name': 'concept_learning',
        'description': 'Configuration optimized for concept learning experiments',
        'model': {
            'input_size': 64,
            'output_size': 8,
            'layer_configs': [
                {
                    'base_population_size': 32,
                    'neuron_pool_size': 64,
                    'output_size': 16,
                    'initial_budget': 50.0,
                    'budget_decay_rate': 0.01,
                    'recruitment_threshold': 0.05,  # Lower threshold for concept formation
                    'population_activation': 'relu',
                    'recurrent_strength': 0.1,
                    'inhibition_strength': 0.05,
                    'adaptation_rate': 0.02,  # Faster adaptation
                    'noise_level': 0.0
                },
                {
                    'base_population_size': 16,
                    'neuron_pool_size': 32,
                    'output_size': 8,
                    'initial_budget': 50.0,
                    'budget_decay_rate': 0.01,
                    'recruitment_threshold': 0.1,
                    'population_activation': 'relu',
                    'recurrent_strength': 0.1,
                    'inhibition_strength': 0.05,
                    'adaptation_rate': 0.01,
                    'noise_level': 0.0
                }
            ],
            'global_budget_decay': 0.0,
            'inter_layer_connections': True,
            'use_global_feedback': True,
            'output_activation': 'none'
        },
        'training': {
            'learning_rate': 0.0005,  # Slower learning for stable concepts
            'batch_size': 32,
            'num_epochs': 150,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'recruitment_loss_weight': 0.15,  # Higher recruitment regularization
            'flexibility_loss_weight': 0.1,
            'connectivity_loss_weight': 0.02,
            'budget_regularization': 0.01,
            'budget_decay_schedule': 'linear',
            'recruitment_warmup_epochs': 10,
            'flexibility_ramp_epochs': 20,
            'early_stopping': True,
            'patience': 15,
            'min_delta': 0.001
        },
        'dataset_type': 'concept_learning',
        'data_params': {
            'num_concepts': 8,
            'concept_complexity': 'medium',
            'hierarchical': True,
            'samples_per_concept': 200
        },
        'track_connectivity': True,
        'track_flexibility': True,
        'analysis_frequency': 5,
        'autism_modeling': False,
        'baseline_comparison': True,
        'baseline_models': ['mlp', 'transformer'],
        'save_results': True,
        'output_dir': 'results/concept_learning',
        'plot_results': True
    }
    return config

def create_flexibility_test_config():
    """Create flexibility_test.yaml configuration"""
    config = {
        'name': 'flexibility_test',
        'description': 'Configuration optimized for cognitive flexibility experiments',
        'model': {
            'input_size': 64,
            'output_size': 4,
            'layer_configs': [
                {
                    'base_population_size': 32,
                    'neuron_pool_size': 64,
                    'output_size': 16,
                    'initial_budget': 50.0,
                    'budget_decay_rate': 0.005,  # Slower budget decay for flexibility
                    'recruitment_threshold': 0.08,
                    'population_activation': 'relu',
                    'recurrent_strength': 0.15,  # Stronger recurrence for memory
                    'inhibition_strength': 0.05,
                    'adaptation_rate': 0.01,
                    'noise_level': 0.0
                }
            ],
            'global_budget_decay': 0.0,
            'inter_layer_connections': False,  # Simpler for flexibility testing
            'use_global_feedback': True,
            'output_activation': 'none'
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 16,  # Smaller batches for task switching
            'num_epochs': 100,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'recruitment_loss_weight': 0.05,  # Lower recruitment penalty
            'flexibility_loss_weight': 0.2,   # Higher flexibility emphasis
            'connectivity_loss_weight': 0.02,
            'budget_regularization': 0.005,
            'budget_decay_schedule': 'linear',
            'recruitment_warmup_epochs': 5,
            'flexibility_ramp_epochs': 30,
            'early_stopping': True,
            'patience': 15,
            'min_delta': 0.001
        },
        'dataset_type': 'task_switching',
        'data_params': {
            'task_types': ['classification', 'regression', 'sequence'],
            'switch_probability': 0.3,
            'sequence_length': 100,
            'num_sequences': 50,
            'task_difficulty': 'medium'
        },
        'track_connectivity': True,
        'track_flexibility': True,
        'analysis_frequency': 5,
        'autism_modeling': False,
        'baseline_comparison': True,
        'baseline_models': ['mlp', 'lstm', 'transformer'],
        'save_results': True,
        'output_dir': 'results/flexibility',
        'plot_results': True
    }
    return config

def create_autism_comparison_configs():
    """Create autism comparison configurations"""

    # Neurotypical configuration
    neurotypical_config = {
        'name': 'neurotypical',
        'description': 'Neurotypical processing configuration',
        'model': {
            'input_size': 64,
            'output_size': 4,
            'layer_configs': [
                {
                    'base_population_size': 32,
                    'neuron_pool_size': 64,
                    'output_size': 16,
                    'initial_budget': 50.0,
                    'budget_decay_rate': 0.01,
                    'recruitment_threshold': 0.08,  # Lower threshold (more flexible)
                    'population_activation': 'relu',
                    'recurrent_strength': 0.1,      # Balanced connections
                    'inhibition_strength': 0.05,
                    'adaptation_rate': 0.015,       # Faster adaptation
                    'noise_level': 0.0
                }
            ],
            'global_budget_decay': 0.0,
            'inter_layer_connections': False,
            'use_global_feedback': True,  # Enhanced global processing
            'output_activation': 'none'
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 120,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'recruitment_loss_weight': 0.1,
            'flexibility_loss_weight': 0.1,  # Higher flexibility
            'connectivity_loss_weight': 0.02,
            'budget_regularization': 0.01,
            'budget_decay_schedule': 'linear',
            'recruitment_warmup_epochs': 10,
            'flexibility_ramp_epochs': 20,
            'early_stopping': True,
            'patience': 10,
            'min_delta': 0.001
        },
        'dataset_type': 'autism_modeling',
        'data_params': {
            'condition': 'neurotypical',
            'processing_style': 'mixed',
            'pattern_completion_bias': 0.7,
            'sensory_processing_difference': 0.1,
            'num_samples': 1000
        },
        'track_connectivity': True,
        'track_flexibility': True,
        'analysis_frequency': 10,
        'autism_modeling': True,
        'baseline_comparison': True,
        'baseline_models': ['mlp'],
        'save_results': True,
        'output_dir': 'results/neurotypical',
        'plot_results': True
    }

    # Autism-like configuration
    autism_config = {
        'name': 'autism_like',
        'description': 'Autism-like processing configuration',
        'model': {
            'input_size': 64,
            'output_size': 4,
            'layer_configs': [
                {
                    'base_population_size': 32,
                    'neuron_pool_size': 64,
                    'output_size': 16,
                    'initial_budget': 50.0,
                    'budget_decay_rate': 0.01,
                    'recruitment_threshold': 0.12,  # Higher threshold (more rigid)
                    'population_activation': 'relu',
                    'recurrent_strength': 0.2,      # Stronger local connections
                    'inhibition_strength': 0.08,    # Stronger inhibition
                    'adaptation_rate': 0.005,       # Slower adaptation
                    'noise_level': 0.0
                }
            ],
            'global_budget_decay': 0.0,
            'inter_layer_connections': False,
            'use_global_feedback': False,  # Reduced global processing
            'output_activation': 'none'
        },
        'training': {
            'learning_rate': 0.0005,
            'batch_size': 32,
            'num_epochs': 120,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'recruitment_loss_weight': 0.2,   # Higher rigidity
            'flexibility_loss_weight': 0.02,  # Lower flexibility
            'connectivity_loss_weight': 0.05,
            'budget_regularization': 0.01,
            'budget_decay_schedule': 'linear',
            'recruitment_warmup_epochs': 10,
            'flexibility_ramp_epochs': 20,
            'early_stopping': True,
            'patience': 10,
            'min_delta': 0.001
        },
        'dataset_type': 'autism_modeling',
        'data_params': {
            'condition': 'autism',
            'processing_style': 'local',
            'pattern_completion_bias': 0.3,
            'sensory_processing_difference': 0.3,
            'num_samples': 1000
        },
        'track_connectivity': True,
        'track_flexibility': True,
        'analysis_frequency': 10,
        'autism_modeling': True,
        'baseline_comparison': True,
        'baseline_models': ['mlp'],
        'save_results': True,
        'output_dir': 'results/autism',
        'plot_results': True
    }

    return neurotypical_config, autism_config

def save_config_to_yaml(config, filepath):
    """Save configuration dictionary to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f"✓ Created: {filepath}")

def create_all_config_files():
    """Create all configuration files"""
    print("Creating DRN Configuration Files")
    print("=" * 40)

    # Create configs directory
    config_dir = create_configs_directory()

    # Create all config files
    configs = {
        'default.yaml': create_default_config(),
        'concept_learning.yaml': create_concept_learning_config(),
        'flexibility_test.yaml': create_flexibility_test_config(),
    }

    # Add autism comparison configs
    neurotypical_config, autism_config = create_autism_comparison_configs()
    configs['neurotypical.yaml'] = neurotypical_config
    configs['autism_like.yaml'] = autism_config

    # Save all configs
    for filename, config in configs.items():
        filepath = config_dir / filename
        save_config_to_yaml(config, filepath)

    print(f"\n✓ Created {len(configs)} configuration files in {config_dir.absolute()}")

    # Create a README for the configs
    readme_content = """# DRN Configuration Files

This directory contains YAML configuration files for Dynamic Recruitment Networks experiments.

## Available Configurations

- **default.yaml**: Standard medium-sized DRN configuration
- **concept_learning.yaml**: Optimized for concept learning experiments
- **flexibility_test.yaml**: Optimized for cognitive flexibility testing
- **neurotypical.yaml**: Neurotypical processing patterns
- **autism_like.yaml**: Autism-like processing patterns

## Usage

```python
from drn.utils.config import ConfigManager

# Load a configuration
config_manager = ConfigManager()
config = config_manager.load_config('concept_learning.yaml')

# Use the configuration
from drn.models.drn_network import DRNNetwork
model = DRNNetwork(
    input_size=config.model.input_size,
    layer_configs=[vars(lc) for lc in config.model.layer_configs],
    output_size=config.model.output_size
)
```

## Customization

You can modify these files directly or create new ones based on these templates.
All configurations follow the same structure with model, training, and data parameters.
"""

    readme_path = config_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created: {readme_path}")

    return config_dir

def test_config_loading():
    """Test that the created configs can be loaded"""
    print("\nTesting Configuration Loading")
    print("=" * 30)

    try:
        config_dir = Path("configs")
        if not config_dir.exists():
            print("✗ Configs directory not found. Run create_all_config_files() first.")
            return False

        # Test loading each config
        config_files = list(config_dir.glob("*.yaml"))
        successful_loads = 0

        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                # Basic validation
                assert 'name' in config
                assert 'model' in config
                assert 'training' in config

                print(f"✓ {config_file.name}: {config['name']}")
                successful_loads += 1

            except Exception as e:
                print(f"✗ {config_file.name}: Failed to load - {e}")

        print(f"\n✓ Successfully loaded {successful_loads}/{len(config_files)} configuration files")
        return successful_loads == len(config_files)

    except Exception as e:
        print(f"✗ Config testing failed: {e}")
        return False

def main():
    """Main function to create all config files"""
    print("DRN Configuration Setup")
    print("=" * 50)

    try:
        # Create all config files
        config_dir = create_all_config_files()

        # Test loading
        if test_config_loading():
            print("\n🎉 All configuration files created and tested successfully!")
            print(f"\nConfiguration files are in: {config_dir.absolute()}")
            print("\nNext steps:")
            print("1. You can now run: python test_population_standalone.py")
            print("2. Or try: from drn.utils.config import ConfigManager")
            print("3. Load configs: config = config_manager.load_config('default.yaml')")
        else:
            print("\n⚠️  Some configuration files failed to load properly.")

    except Exception as e:
        print(f"\n✗ Configuration setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
