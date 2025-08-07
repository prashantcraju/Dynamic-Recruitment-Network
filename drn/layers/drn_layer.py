"""
drn/layers/drn_layer.py

Minimal DRNLayer implementation for compatibility.
This provides the interface expected by DRNNetwork while using PopulationLayer internally.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from drn.layers.population_layer import PopulationLayer


class DRNLayer(nn.Module):
    """
    Minimal DRN Layer that wraps PopulationLayer for compatibility.

    This is a simplified implementation that provides the interface
    expected by DRNNetwork while the full recruitment system is developed.
    """

    def __init__(
        self,
        input_size: int,
        base_population_size: int = 32,
        neuron_pool_size: int = 64,
        output_size: int = 16,
        initial_budget: float = 50.0,
        budget_decay_rate: float = 0.01,
        recruitment_threshold: float = 0.1,
        population_activation: str = 'relu',
        recurrent_strength: float = 0.1,
        inhibition_strength: float = 0.05,
        adaptation_rate: float = 0.01,
        noise_level: float = 0.0,
        **kwargs
    ):
        super(DRNLayer, self).__init__()

        self.input_size = input_size
        self.base_population_size = base_population_size
        self.neuron_pool_size = neuron_pool_size
        self.output_size = output_size
        self.initial_budget = initial_budget
        self.budget_decay_rate = budget_decay_rate
        self.recruitment_threshold = recruitment_threshold

        # Use PopulationLayer as the core implementation
        num_populations = max(2, neuron_pool_size // base_population_size)
        population_size = base_population_size

        self.population_layer = PopulationLayer(
            input_size=input_size,
            num_populations=num_populations,
            population_size=population_size,
            output_size=output_size,
            competition_strength=inhibition_strength,
            collaboration_strength=recurrent_strength,
            population_specialization=True
        )

        # Budget tracking (simplified)
        self.current_budget = initial_budget
        self._forward_count = 0

    def forward(
        self,
        x: torch.Tensor,
        external_feedback: Optional[torch.Tensor] = None,
        force_budget_reset: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through DRN layer."""
        self._forward_count += 1

        # Reset budget if requested
        if force_budget_reset:
            self.current_budget = self.initial_budget

        # Forward through population layer
        output, pop_info = self.population_layer.forward(
            x,
            reset_populations=force_budget_reset
        )

        # Update budget (simplified)
        budget_used = pop_info['active_populations'] * 2.0  # Simplified cost
        self.current_budget = max(0, self.current_budget - budget_used * self.budget_decay_rate)

        # Create DRNLayer-compatible info
        layer_info = {
            'num_recruited': int(pop_info['active_populations'] * pop_info['num_populations']),
            'budget_spent': budget_used,
            'budget_remaining': self.current_budget,
            'budget_utilization': min(1.0, budget_used / self.initial_budget),
            'feedback_magnitude': 0.1,  # Placeholder
            'population_infos': pop_info.get('population_infos', []),
            'sparsity': pop_info['layer_sparsity'],
            'coherence': pop_info['layer_coherence']
        }

        return output, layer_info

    def reset_budget(self):
        """Reset the layer budget."""
        self.current_budget = self.initial_budget
        self.population_layer.reset_all_populations()


# For backward compatibility
AdaptiveDRNLayer = DRNLayer
