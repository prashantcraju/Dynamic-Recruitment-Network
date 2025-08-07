"""
drn/core/base_population.py

Base Population component that maintains active neural ensembles throughout processing.
Unlike traditional layers that process and pass on, base populations stay active and 
coordinate neuron recruitment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import warnings


class BasePopulation(nn.Module):
    """
    Base population that maintains active state throughout processing.
    
    The base population represents a group of neurons that:
    1. Stay active throughout the forward pass
    2. Integrate input and recurrent feedback
    3. Provide the activation pattern for neuron recruitment decisions
    4. Maintain state across time steps in sequential processing
    
    Args:
        input_size (int): Size of input to the population
        population_size (int): Number of neurons in the base population
        activation (str): Activation function ('tanh', 'relu', 'sigmoid')
        feedback_strength (float): Strength of recurrent feedback (0.0 to 1.0)
        use_bias (bool): Whether to use bias terms
        dropout_rate (float): Dropout rate for regularization
        
    Example:
        >>> pop = BasePopulation(input_size=64, population_size=128)
        >>> x = torch.randn(32, 64)  # batch_size=32
        >>> output, state = pop(x)
        >>> print(f"Output shape: {output.shape}")  # [32, 128]
    """
    
    def __init__(
        self,
        input_size: int,
        population_size: int,
        activation: str = 'tanh',
        feedback_strength: float = 0.1,
        use_bias: bool = True,
        dropout_rate: float = 0.0
    ):
        super(BasePopulation, self).__init__()
        
        # Validate inputs
        if input_size <= 0 or population_size <= 0:
            raise ValueError("input_size and population_size must be positive")
        if not 0.0 <= feedback_strength <= 1.0:
            raise ValueError("feedback_strength must be between 0.0 and 1.0")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
            
        self.input_size = input_size
        self.population_size = population_size
        self.feedback_strength = feedback_strength
        
        # Input transformation
        self.input_projection = nn.Linear(input_size, population_size, bias=use_bias)
        
        # Recurrent feedback transformation
        self.feedback_projection = nn.Linear(population_size, population_size, bias=use_bias)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # State management
        self.register_buffer('_current_state', torch.zeros(1, population_size))
        self._batch_size = 1
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. "
                           f"Available: {list(activations.keys())}")
        
        return activations[activation.lower()]
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.feedback_projection.weight)
        
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        if self.feedback_projection.bias is not None:
            nn.init.zeros_(self.feedback_projection.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        feedback: Optional[torch.Tensor] = None,
        reset_state: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the base population.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            feedback: Optional feedback tensor of shape [batch_size, population_size]
            reset_state: Whether to reset the internal state
            
        Returns:
            Tuple of (population_output, info_dict)
            - population_output: Activated population state [batch_size, population_size]
            - info_dict: Dictionary with internal state information
        """
        batch_size = x.shape[0]
        
        # Handle batch size changes
        if batch_size != self._batch_size or reset_state:
            self._reset_state(batch_size)
        
        # Input transformation
        input_contribution = self.input_projection(x)
        
        # Feedback contribution (from previous state and external feedback)
        feedback_contribution = torch.zeros_like(input_contribution)
        
        # Internal recurrent feedback
        if self._current_state.shape[0] == batch_size:
            internal_feedback = self.feedback_projection(self._current_state)
            feedback_contribution += self.feedback_strength * internal_feedback
        
        # External feedback (from recruited neurons)
        if feedback is not None:
            if feedback.shape != (batch_size, self.population_size):
                raise ValueError(f"Feedback shape {feedback.shape} doesn't match "
                               f"expected {(batch_size, self.population_size)}")
            feedback_contribution += feedback
        
        # Combine inputs
        combined_input = input_contribution + feedback_contribution
        
        # Apply activation
        population_output = self.activation(combined_input)
        
        # Apply dropout during training
        if self.dropout is not None and self.training:
            population_output = self.dropout(population_output)
        
        # Update internal state
        self._current_state = population_output.detach()
        self._batch_size = batch_size
        
        # Prepare info dictionary
        info = {
            'input_contribution': input_contribution.detach(),
            'feedback_contribution': feedback_contribution.detach(),
            'activation_mean': population_output.mean().item(),
            'activation_std': population_output.std().item(),
            'state_norm': torch.norm(self._current_state).item()
        }
        
        return population_output, info
    
    def _reset_state(self, batch_size: int):
        """Reset internal state for new batch or sequence."""
        device = self.input_projection.weight.device
        self._current_state = torch.zeros(batch_size, self.population_size, device=device)
        self._batch_size = batch_size
    
    def reset_state(self):
        """Public method to reset state."""
        self._reset_state(self._batch_size)
    
    def get_state(self) -> torch.Tensor:
        """Get current population state."""
        return self._current_state.clone()
    
    def set_state(self, state: torch.Tensor):
        """Set population state."""
        if state.shape[1] != self.population_size:
            raise ValueError(f"State shape {state.shape} doesn't match "
                           f"population_size {self.population_size}")
        self._current_state = state.detach()
        self._batch_size = state.shape[0]
    
    def get_activation_pattern(self) -> torch.Tensor:
        """Get current activation pattern for recruitment decisions."""
        return self._current_state.clone()
    
    def compute_population_statistics(self) -> Dict[str, float]:
        """Compute statistics about current population state."""
        if self._current_state.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        
        state = self._current_state
        return {
            'mean': state.mean().item(),
            'std': state.std().item(),
            'max': state.max().item(),
            'min': state.min().item(),
            'active_neurons': (state.abs() > 0.1).sum().item(),
            'sparsity': (state.abs() < 0.01).float().mean().item()
        }
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'input_size={self.input_size}, '
                f'population_size={self.population_size}, '
                f'feedback_strength={self.feedback_strength}')


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    print("Testing BasePopulation...")
    
    # Create population
    pop = BasePopulation(input_size=10, population_size=20, activation='tanh')
    
    # Test forward pass
    x = torch.randn(5, 10)  # batch_size=5
    output, info = pop(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Info keys: {list(info.keys())}")
    
    # Test with feedback
    feedback = torch.randn(5, 20)
    output2, info2 = pop(x, feedback=feedback)
    print(f"Output with feedback shape: {output2.shape}")
    
    # Test state management
    state = pop.get_state()
    print(f"Current state shape: {state.shape}")
    
    stats = pop.compute_population_statistics()
    print(f"Population statistics: {stats}")
    
    print("BasePopulation tests passed!")