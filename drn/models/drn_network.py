"""
drn/models/drn_network.py

Complete Dynamic Recruitment Network model that combines multiple DRN layers
into a full neural network architecture. This is the main model class that
researchers will use for experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import warnings
import math

from ..layers.drn_layer import DRNLayer
try:
    from ..utils.config import DRNConfig, ModelConfig, LayerConfig
except ImportError:
    # Fallback for testing - create minimal config classes
    class DRNConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ModelConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class LayerConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


class DRNNetwork(nn.Module):
    """
    Complete Dynamic Recruitment Network model.
    
    This model stacks multiple DRN layers to create a deep network that uses
    dynamic neuron recruitment instead of traditional dense connectivity.
    
    Key innovations:
    - Population-based processing with selective neuron recruitment
    - Budget-constrained connectivity formation
    - Recurrent feedback between recruited neurons and populations
    - Adaptive architecture that can modify behavior during training
    
    Args:
        input_size (int): Size of input features
        layer_configs (List[Dict]): Configuration for each DRN layer
        output_size (int): Size of final output
        global_budget_decay (float): Global decay rate for all layer budgets
        inter_layer_connections (bool): Whether to allow connections between non-adjacent layers
        use_global_feedback (bool): Whether to use feedback across multiple layers
        output_activation (str): Final output activation ('none', 'softmax', 'sigmoid')
        
    Example:
        >>> config = [
        ...     {'base_population_size': 64, 'neuron_pool_size': 128, 'output_size': 32},
        ...     {'base_population_size': 32, 'neuron_pool_size': 64, 'output_size': 16}
        ... ]
        >>> model = DRNNetwork(input_size=128, layer_configs=config, output_size=10)
        >>> x = torch.randn(32, 128)
        >>> output, info = model(x)
    """
    
    def __init__(
        self,
        input_size: int,
        layer_configs: List[Dict[str, Any]],
        output_size: int,
        global_budget_decay: float = 0.0,
        inter_layer_connections: bool = False,
        use_global_feedback: bool = True,
        output_activation: str = 'none'
    ):
        super(DRNNetwork, self).__init__()
        
        # Validate inputs
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if not layer_configs:
            raise ValueError("layer_configs cannot be empty")
        if not 0.0 <= global_budget_decay <= 1.0:
            raise ValueError("global_budget_decay must be between 0.0 and 1.0")
        if output_activation not in ['none', 'softmax', 'sigmoid', 'tanh']:
            raise ValueError("output_activation must be 'none', 'softmax', 'sigmoid', or 'tanh'")
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = len(layer_configs)
        self.global_budget_decay = global_budget_decay
        self.inter_layer_connections = inter_layer_connections
        self.use_global_feedback = use_global_feedback
        self.output_activation = output_activation
        
        # Input projection to first layer
        first_layer_input = layer_configs[0].get('base_population_size', 64)
        self.input_projection = nn.Linear(input_size, first_layer_input)
        
        # Create DRN layers
        self.drn_layers = nn.ModuleList()
        current_input_size = first_layer_input
        
        for i, config in enumerate(layer_configs):
            # Set input size for each layer
            layer_config = config.copy()
            layer_config['input_size'] = current_input_size
            
            # Apply global budget decay
            if 'initial_budget' in layer_config:
                layer_config['initial_budget'] *= (1 - global_budget_decay) ** i
            
            # Create layer
            layer = DRNLayer(**layer_config)
            self.drn_layers.append(layer)
            
            # Update input size for next layer
            current_input_size = config.get('output_size', current_input_size)
        
        # Final output layer
        final_layer_output = layer_configs[-1].get('output_size', current_input_size)
        self.final_projection = nn.Linear(final_layer_output, output_size)
        
        # Inter-layer connection matrix (if enabled)
        if inter_layer_connections:
            self.inter_layer_weights = nn.Parameter(
                torch.zeros(self.num_layers, self.num_layers)
            )
            # Initialize with small values, emphasizing local connections
            nn.init.normal_(self.inter_layer_weights, 0, 0.1)
            # Zero out self-connections and upper triangular (no backward connections)
            with torch.no_grad():
                self.inter_layer_weights.fill_diagonal_(0)
                self.inter_layer_weights.triu_(1).zero_()
        
        # Global feedback mechanisms
        if use_global_feedback:
            self.global_feedback_transform = nn.Linear(output_size, first_layer_input)
        
        # Model state tracking
        self._forward_count = 0
        self._network_info_history: List[Dict] = []
        self._last_global_feedback = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        nn.init.xavier_uniform_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        
        if self.use_global_feedback:
            nn.init.xavier_uniform_(self.global_feedback_transform.weight)
            nn.init.zeros_(self.global_feedback_transform.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_layer_info: bool = True,
        reset_budgets: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through the complete DRN network.
        
        Args:
            x: Input tensor [batch_size, input_size]
            return_layer_info: Whether to return detailed layer information
            reset_budgets: Whether to reset all layer budgets
            
        Returns:
            If return_layer_info=False: output tensor [batch_size, output_size]
            If return_layer_info=True: (output, network_info_dict)
        """
        batch_size = x.shape[0]
        self._forward_count += 1
        
        # Reset budgets if requested
        if reset_budgets:
            self.reset_all_budgets()
        
        # Project input
        current_input = torch.tanh(self.input_projection(x))
        
        # Apply global feedback from previous forward pass
        if self.use_global_feedback and self._last_global_feedback is not None:
            # Add global feedback to input
            current_input = current_input + 0.1 * self._last_global_feedback[:batch_size]
        
        # Track layer outputs and information
        layer_outputs = []
        layer_infos = []
        
        # Pass through DRN layers
        for i, layer in enumerate(self.drn_layers):
            # Compute inter-layer feedback if enabled
            external_feedback = None
            if self.inter_layer_connections and i > 0:
                external_feedback = self._compute_inter_layer_feedback(
                    layer_outputs, i
                )
            
            # Forward through layer
            layer_output, layer_info = layer(
                current_input, 
                external_feedback=external_feedback,
                force_budget_reset=reset_budgets
            )
            
            layer_outputs.append(layer_output)
            layer_infos.append(layer_info)
            
            # Update input for next layer
            current_input = layer_output
        
        # Final projection
        network_output = self.final_projection(current_input)
        
        # Apply output activation
        if self.output_activation == 'softmax':
            network_output = F.softmax(network_output, dim=-1)
        elif self.output_activation == 'sigmoid':
            network_output = torch.sigmoid(network_output)
        elif self.output_activation == 'tanh':
            network_output = torch.tanh(network_output)
        
        # Global feedback (for next forward pass)
        if self.use_global_feedback and len(layer_infos) > 0:
            global_feedback = self.global_feedback_transform(network_output.detach())
            # Store for potential use in next forward pass
            self._last_global_feedback = global_feedback
        
        if not return_layer_info:
            return network_output
        
        # Compile network information
        network_info = self._compile_network_info(layer_infos, layer_outputs, network_output)
        
        # Store history
        self._network_info_history.append(network_info.copy())
        if len(self._network_info_history) > 500:  # Limit history size
            self._network_info_history.pop(0)
        
        return network_output, network_info
    
    def _compute_inter_layer_feedback(self, layer_outputs: List[torch.Tensor], current_layer: int) -> torch.Tensor:
        """Compute feedback from previous layers to current layer."""
        if not self.inter_layer_connections or current_layer == 0:
            return None
        
        feedback = torch.zeros_like(layer_outputs[0])
        
        for prev_layer in range(current_layer):
            weight = torch.sigmoid(self.inter_layer_weights[current_layer, prev_layer])
            if weight.item() > 0.01:  # Only use significant connections
                # Simple linear combination - could be made more sophisticated
                feedback += weight * layer_outputs[prev_layer]
        
        return feedback
    
    def _compile_network_info(
        self, 
        layer_infos: List[Dict], 
        layer_outputs: List[torch.Tensor],
        network_output: torch.Tensor
    ) -> Dict[str, Any]:
        """Compile comprehensive network information."""
        
        # Aggregate layer metrics
        total_recruited = sum(info['num_recruited'] for info in layer_infos)
        total_budget_spent = sum(info['budget_spent'] for info in layer_infos)
        avg_budget_remaining = sum(info['budget_remaining'] for info in layer_infos) / len(layer_infos)
        
        # Network-level metrics
        network_info = {
            # Aggregated metrics
            'total_neurons_recruited': total_recruited,
            'total_budget_spent': total_budget_spent,
            'avg_budget_remaining': avg_budget_remaining,
            'network_sparsity': self._compute_network_sparsity(layer_infos),
            
            # Layer-specific info
            'layer_infos': layer_infos,
            'num_layers': self.num_layers,
            'forward_count': self._forward_count,
            
            # Connectivity analysis
            'connectivity_patterns': self._analyze_connectivity_patterns(layer_infos),
            'recruitment_efficiency': total_recruited / max(0.001, total_budget_spent),
            
            # Output characteristics
            'output_entropy': self._compute_output_entropy(network_output),
            'output_magnitude': torch.norm(network_output).item(),
            
            # Inter-layer connections (if enabled)
            'inter_layer_strength': self._compute_inter_layer_strength() if self.inter_layer_connections else 0.0,
        }
        
        return network_info
    
    def _compute_network_sparsity(self, layer_infos: List[Dict]) -> float:
        """Compute overall network sparsity."""
        total_possible_connections = sum(
            layer.neuron_pool_size for layer in self.drn_layers
        )
        total_actual_connections = sum(info['num_recruited'] for info in layer_infos)
        
        return 1.0 - (total_actual_connections / max(1, total_possible_connections))
    
    def _analyze_connectivity_patterns(self, layer_infos: List[Dict]) -> Dict[str, Any]:
        """Analyze connectivity patterns across layers."""
        patterns = {
            'recruitment_distribution': [info['num_recruited'] for info in layer_infos],
            'budget_utilization': [info['budget_utilization'] for info in layer_infos],
            'feedback_magnitudes': [info['feedback_magnitude'] for info in layer_infos],
        }
        
        # Compute pattern stability
        pattern_stability = 0.0
        if len(self._network_info_history) >= 2:
            prev_recruitment = [
                info['num_recruited'] for info in 
                self._network_info_history[-2]['layer_infos']
            ]
            current_recruitment = patterns['recruitment_distribution']
            
            # Compute pattern correlation
            if len(prev_recruitment) == len(current_recruitment):
                prev_tensor = torch.tensor(prev_recruitment, dtype=torch.float32)
                current_tensor = torch.tensor(current_recruitment, dtype=torch.float32)
                
                # Normalize vectors
                prev_norm = torch.norm(prev_tensor)
                current_norm = torch.norm(current_tensor)
                
                if prev_norm > 0 and current_norm > 0:
                    pattern_stability = torch.dot(prev_tensor / prev_norm, 
                                                current_tensor / current_norm).item()
        
        patterns['pattern_stability'] = pattern_stability
        
        # Compute recruitment variance across layers
        recruitment_vals = patterns['recruitment_distribution']
        if len(recruitment_vals) > 1:
            mean_recruitment = sum(recruitment_vals) / len(recruitment_vals)
            variance = sum((x - mean_recruitment) ** 2 for x in recruitment_vals) / len(recruitment_vals)
            patterns['recruitment_variance'] = variance
        else:
            patterns['recruitment_variance'] = 0.0
        
        return patterns
    
    def _compute_output_entropy(self, output: torch.Tensor) -> float:
        """Compute entropy of network output."""
        # Apply softmax to get probability distribution
        probs = F.softmax(output, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)  # Add small epsilon for numerical stability
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Return mean entropy across batch
        return entropy.mean().item()
    
    def _compute_inter_layer_strength(self) -> float:
        """Compute strength of inter-layer connections."""
        if not self.inter_layer_connections:
            return 0.0
        
        # Compute mean absolute weight for active connections
        weights = torch.sigmoid(self.inter_layer_weights)
        active_weights = weights[weights > 0.01]
        
        if len(active_weights) > 0:
            return active_weights.mean().item()
        else:
            return 0.0
    
    def reset_all_budgets(self):
        """Reset budgets for all DRN layers."""
        for layer in self.drn_layers:
            if hasattr(layer, 'reset_budget'):
                layer.reset_budget()
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        if not self._network_info_history:
            return {'error': 'No forward passes recorded yet'}
        
        recent_info = self._network_info_history[-1]
        
        # Compute historical trends
        history_length = min(50, len(self._network_info_history))
        recent_history = self._network_info_history[-history_length:]
        
        # Recruitment trends
        recruitment_trend = [info['total_neurons_recruited'] for info in recent_history]
        sparsity_trend = [info['network_sparsity'] for info in recent_history]
        
        stats = {
            'current_state': recent_info,
            'trends': {
                'recruitment_mean': sum(recruitment_trend) / len(recruitment_trend),
                'recruitment_std': self._compute_std(recruitment_trend),
                'sparsity_mean': sum(sparsity_trend) / len(sparsity_trend),
                'sparsity_std': self._compute_std(sparsity_trend),
            },
            'layer_analysis': self._analyze_layer_performance(),
            'forward_count': self._forward_count,
            'history_length': len(self._network_info_history)
        }
        
        return stats
    
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation of a list of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _analyze_layer_performance(self) -> Dict[str, Any]:
        """Analyze performance of individual layers."""
        if not self._network_info_history:
            return {}
        
        recent_info = self._network_info_history[-1]
        layer_infos = recent_info['layer_infos']
        
        analysis = {}
        for i, layer_info in enumerate(layer_infos):
            analysis[f'layer_{i}'] = {
                'recruitment_efficiency': layer_info['num_recruited'] / max(0.001, layer_info['budget_spent']),
                'budget_utilization': layer_info['budget_utilization'],
                'feedback_strength': layer_info['feedback_magnitude'],
                'relative_activity': layer_info['num_recruited'] / max(1, sum(info['num_recruited'] for info in layer_infos))
            }
        
        return analysis
    
    def save_config(self, filepath: str):
        """Save network configuration to file."""
        config = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'global_budget_decay': self.global_budget_decay,
            'inter_layer_connections': self.inter_layer_connections,
            'use_global_feedback': self.use_global_feedback,
            'output_activation': self.output_activation,
            'forward_count': self._forward_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load network configuration from file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Validate loaded config matches current network
        for key, value in config.items():
            if hasattr(self, key) and getattr(self, key) != value:
                warnings.warn(f"Config mismatch for {key}: current={getattr(self, key)}, loaded={value}")
    
    def __repr__(self) -> str:
        """String representation of the network."""
        return (f"DRNNetwork(input_size={self.input_size}, "
                f"output_size={self.output_size}, "
                f"num_layers={self.num_layers}, "
                f"inter_layer_connections={self.inter_layer_connections}, "
                f"use_global_feedback={self.use_global_feedback})")


def create_standard_drn(
    input_size: int = 64,
    output_size: int = 10,
    size: str = 'medium',
    **kwargs
) -> DRNNetwork:
    """
    Factory function to create standard DRN configurations.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension  
        size: Model size ('small', 'medium', 'large')
        **kwargs: Additional model parameters
        
    Returns:
        Configured DRNNetwork
    """
    if size == 'small':
        layer_configs = [
            {'base_population_size': 16, 'neuron_pool_size': 32, 'output_size': 8}
        ]
    elif size == 'medium':
        layer_configs = [
            {'base_population_size': 32, 'neuron_pool_size': 64, 'output_size': 16},
            {'base_population_size': 16, 'neuron_pool_size': 32, 'output_size': 8}
        ]
    elif size == 'large':
        layer_configs = [
            {'base_population_size': 64, 'neuron_pool_size': 128, 'output_size': 32},
            {'base_population_size': 32, 'neuron_pool_size': 64, 'output_size': 16},
            {'base_population_size': 16, 'neuron_pool_size': 32, 'output_size': 8}
        ]
    else:
        raise ValueError("Size must be 'small', 'medium', or 'large'")
    
    return DRNNetwork(
        input_size=input_size,
        layer_configs=layer_configs,
        output_size=output_size,
        **kwargs
    )