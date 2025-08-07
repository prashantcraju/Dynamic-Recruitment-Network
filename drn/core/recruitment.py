"""
drn/core/recruitment.py

Neuron Recruitment component that dynamically selects individual neurons
for activation based on population state and budget constraints.

This is the core innovation: instead of activating ALL neurons in the next layer
(1:1 connectivity), we selectively recruit only relevant neurons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import math
import warnings


class NeuronRecruitment(nn.Module):
    """
    Handles dynamic recruitment of individual neurons from a pool.
    
    This component implements the key departure from traditional architectures:
    - Traditional: All layer N neurons connect to ALL layer N+1 neurons
    - DRN: Population recruits SELECTED neurons based on relevance and budget
    
    Args:
        population_size (int): Size of the base population providing signals
        neuron_pool_size (int): Size of the pool of recruitable neurons
        recruitment_temperature (float): Temperature for recruitment probability (higher = more random)
        diversity_bonus (float): Bonus for recruiting less-frequently used neurons
        min_connections (int): Minimum number of neurons to recruit
        max_connections (int): Maximum number of neurons to recruit
        use_attention (bool): Whether to use attention mechanism for recruitment
        
    Example:
        >>> recruitment = NeuronRecruitment(population_size=128, neuron_pool_size=256)
        >>> pop_state = torch.randn(32, 128)  # batch_size=32
        >>> budget = 0.5
        >>> indices, outputs = recruitment(pop_state, budget)
    """
    
    def __init__(
        self,
        population_size: int,
        neuron_pool_size: int,
        recruitment_temperature: float = 1.0,
        diversity_bonus: float = 0.1,
        min_connections: int = 1,
        max_connections: Optional[int] = None,
        use_attention: bool = True
    ):
        super(NeuronRecruitment, self).__init__()
        
        # Validate inputs
        if population_size <= 0 or neuron_pool_size <= 0:
            raise ValueError("population_size and neuron_pool_size must be positive")
        if recruitment_temperature <= 0:
            raise ValueError("recruitment_temperature must be positive")
        if min_connections < 0:
            raise ValueError("min_connections must be non-negative")
        if max_connections is not None and max_connections < min_connections:
            raise ValueError("max_connections must be >= min_connections")
        
        self.population_size = population_size
        self.neuron_pool_size = neuron_pool_size
        self.recruitment_temperature = recruitment_temperature
        self.diversity_bonus = diversity_bonus
        self.min_connections = min_connections
        self.max_connections = max_connections or neuron_pool_size // 4
        self.use_attention = use_attention
        
        # Learnable recruitment preferences
        # Each neuron has a vector that determines its affinity to population patterns
        self.recruitment_weights = nn.Parameter(
            torch.randn(neuron_pool_size, population_size) * 0.1
        )
        
        # Optional attention mechanism for recruitment
        if self.use_attention:
            self.attention_query = nn.Linear(population_size, population_size // 2)
            self.attention_key = nn.Linear(population_size, population_size // 2)
            self.attention_value = nn.Linear(population_size, population_size)
        
        # Bias terms for each recruitable neuron
        self.recruitment_bias = nn.Parameter(torch.zeros(neuron_pool_size))
        
        # Diversity tracking (non-learnable)
        self.register_buffer('recruitment_counts', torch.zeros(neuron_pool_size))
        self.register_buffer('total_recruitment_steps', torch.tensor(0))
        
        # History for analysis
        self._recruitment_history: List[List[int]] = []
        self._probability_history: List[torch.Tensor] = []
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize recruitment weights."""
        # Xavier initialization for recruitment weights
        nn.init.xavier_uniform_(self.recruitment_weights)
        
        # Initialize attention layers if used
        if self.use_attention:
            nn.init.xavier_uniform_(self.attention_query.weight)
            nn.init.xavier_uniform_(self.attention_key.weight)
            nn.init.xavier_uniform_(self.attention_value.weight)
            nn.init.zeros_(self.attention_query.bias)
            nn.init.zeros_(self.attention_key.bias)
            nn.init.zeros_(self.attention_value.bias)
    
    def compute_recruitment_probabilities(
        self, 
        population_state: torch.Tensor,
        apply_diversity_bonus: bool = True
    ) -> torch.Tensor:
        """
        Compute recruitment probability for each neuron in the pool.
        
        Args:
            population_state: Current population state [batch_size, population_size]
            apply_diversity_bonus: Whether to apply diversity bonus
            
        Returns:
            Recruitment probabilities [batch_size, neuron_pool_size]
        """
        batch_size = population_state.shape[0]
        
        # Optional attention mechanism
        if self.use_attention:
            population_state = self._apply_attention(population_state)
        
        # Compute affinities: how well each neuron matches current population state
        # [batch_size, neuron_pool_size] = [batch_size, population_size] @ [population_size, neuron_pool_size]^T
        affinities = torch.matmul(population_state, self.recruitment_weights.t())
        
        # Add learnable bias
        affinities = affinities + self.recruitment_bias.unsqueeze(0)
        
        # Apply diversity bonus
        if apply_diversity_bonus and self.diversity_bonus > 0:
            diversity_bonus = self._compute_diversity_bonus()
            affinities = affinities + diversity_bonus.unsqueeze(0)
        
        # Convert to probabilities with temperature
        probabilities = F.softmax(affinities / self.recruitment_temperature, dim=-1)
        
        return probabilities
    
    def _apply_attention(self, population_state: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to population state."""
        # Self-attention over population dimensions
        query = self.attention_query(population_state)
        key = self.attention_key(population_state)
        value = self.attention_value(population_state)
        
        # Compute attention weights
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_state = torch.matmul(attention_weights, value.unsqueeze(1)).squeeze(1)
        
        return attended_state
    
    def _compute_diversity_bonus(self) -> torch.Tensor:
        """
        Compute diversity bonus to encourage recruiting less-used neurons.
        
        Returns:
            Bonus values for each neuron [neuron_pool_size]
        """
        if self.total_recruitment_steps.item() == 0:
            return torch.zeros_like(self.recruitment_counts)
        
        # Compute recruitment frequencies
        frequencies = self.recruitment_counts / self.total_recruitment_steps.float()
        
        # Invert frequencies - less used neurons get higher bonus
        # Add small constant to avoid division by zero
        diversity_bonus = self.diversity_bonus / (frequencies + 1e-6)
        
        return diversity_bonus
    
    def recruit_neurons(
        self,
        population_state: torch.Tensor,
        available_budget: float,
        neuron_pool: nn.ModuleList,
        training: bool = True
    ) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
        """
        Recruit neurons based on population state and budget.
        
        Args:
            population_state: Current population state [batch_size, population_size]
            available_budget: Available budget for recruitment
            neuron_pool: Pool of recruitable neurons
            training: Whether in training mode (affects stochastic sampling)
            
        Returns:
            Tuple of (recruited_indices, outputs, info_dict)
        """
        batch_size = population_state.shape[0]
        
        # Compute recruitment probabilities
        probabilities = self.compute_recruitment_probabilities(population_state)
        
        # Determine number of neurons to recruit based on budget
        max_affordable = max(self.min_connections, 
                           int(available_budget / 0.1))  # Assume cost of 0.1 per neuron
        num_to_recruit = min(self.max_connections, max_affordable)
        
        if num_to_recruit <= 0:
            # No budget - return empty results
            return [], torch.zeros(batch_size, 0), {'recruitment_failed': True}
        
        # Sample neurons for each batch item
        recruited_indices_batch = []
        outputs_batch = []
        
        for b in range(batch_size):
            # Get probabilities for this batch item
            batch_probs = probabilities[b]
            
            if training:
                # Stochastic sampling during training
                recruited_idx = torch.multinomial(
                    batch_probs, 
                    num_samples=num_to_recruit, 
                    replacement=False
                )
            else:
                # Deterministic top-k during inference
                recruited_idx = torch.topk(batch_probs, k=num_to_recruit)[1]
            
            recruited_indices_batch.append(recruited_idx.tolist())
            
            # Get outputs from recruited neurons
            batch_outputs = []
            for idx in recruited_idx:
                neuron_output = neuron_pool[idx](population_state[b:b+1])
                batch_outputs.append(neuron_output)
            
            if batch_outputs:
                batch_output = torch.cat(batch_outputs, dim=-1)
            else:
                batch_output = torch.zeros(1, 0)
            
            outputs_batch.append(batch_output)
        
        # Combine outputs
        if outputs_batch and outputs_batch[0].shape[-1] > 0:
            combined_outputs = torch.cat(outputs_batch, dim=0)
        else:
            combined_outputs = torch.zeros(batch_size, 0)
        
        # Use first batch item's indices as representative (for single-batch processing)
        recruited_indices = recruited_indices_batch[0] if recruited_indices_batch else []
        
        # Update recruitment statistics
        self._update_recruitment_stats(recruited_indices_batch, probabilities)
        
        # Prepare info dictionary
        info = {
            'recruitment_probabilities': probabilities.detach(),
            'recruited_indices': recruited_indices,
            'num_recruited': len(recruited_indices),
            'budget_used': len(recruited_indices) * 0.1,  # Assume cost of 0.1
            'recruitment_entropy': self._compute_entropy(probabilities),
            'diversity_score': self._compute_diversity_score(recruited_indices)
        }
        
        return recruited_indices, combined_outputs, info
    
    def _update_recruitment_stats(self, recruited_indices_batch: List[List[int]], probabilities: torch.Tensor):
        """Update recruitment statistics for analysis."""
        # Update recruitment counts
        for indices in recruited_indices_batch:
            for idx in indices:
                self.recruitment_counts[idx] += 1
        
        self.total_recruitment_steps += 1
        
        # Store history for analysis
        self._recruitment_history.append(recruited_indices_batch[0] if recruited_indices_batch else [])
        self._probability_history.append(probabilities[0].detach().clone())
    
    def _compute_entropy(self, probabilities: torch.Tensor) -> float:
        """Compute entropy of recruitment probabilities."""
        # Average entropy across batch
        log_probs = torch.log(probabilities + 1e-10)
        entropy = -(probabilities * log_probs).sum(dim=-1).mean()
        return entropy.item()
    
    def _compute_diversity_score(self, recruited_indices: List[int]) -> float:
        """Compute diversity score of recruited neurons."""
        if not recruited_indices:
            return 0.0
        
        # Simple diversity: ratio of unique indices to total
        unique_indices = len(set(recruited_indices))
        return unique_indices / len(recruited_indices)
    
    def get_recruitment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recruitment statistics."""
        if self.total_recruitment_steps.item() == 0:
            return {'total_steps': 0, 'message': 'No recruitment steps yet'}
        
        frequencies = self.recruitment_counts / self.total_recruitment_steps.float()
        
        return {
            'total_recruitment_steps': self.total_recruitment_steps.item(),
            'recruitment_frequencies': frequencies.tolist(),
            'most_recruited_neurons': torch.topk(frequencies, k=min(10, self.neuron_pool_size))[1].tolist(),
            'least_recruited_neurons': torch.topk(frequencies, k=min(10, self.neuron_pool_size), largest=False)[1].tolist(),
            'recruitment_variance': frequencies.var().item(),
            'recruitment_entropy': -(frequencies * torch.log(frequencies + 1e-10)).sum().item(),
            'unused_neurons': (frequencies == 0).sum().item()
        }
    
    def reset_recruitment_stats(self):
        """Reset recruitment statistics."""
        self.recruitment_counts.zero_()
        self.total_recruitment_steps.zero_()
        self._recruitment_history.clear()
        self._probability_history.clear()
    
    def get_recruitment_history(self) -> Dict[str, List]:
        """Get recruitment history for visualization."""
        return {
            'recruitment_history': self._recruitment_history.copy(),
            'probability_history': [p.cpu().numpy() for p in self._probability_history]
        }
    
    def set_recruitment_temperature(self, temperature: float):
        """Adjust recruitment temperature during training."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.recruitment_temperature = temperature
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'population_size={self.population_size}, '
                f'neuron_pool_size={self.neuron_pool_size}, '
                f'temperature={self.recruitment_temperature}')


class ContextualRecruitment(NeuronRecruitment):
    """
    Extended recruitment that considers context from previous recruitments.
    
    This version maintains memory of recent recruitment patterns and uses
    this context to make more informed recruitment decisions.
    """
    
    def __init__(self, context_window: int = 5, context_weight: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        
        self.context_window = context_window
        self.context_weight = context_weight
        
        # Context memory
        self.register_buffer('context_memory', 
                           torch.zeros(context_window, self.neuron_pool_size))
        self.register_buffer('context_pointer', torch.tensor(0))
    
    def compute_recruitment_probabilities(self, population_state: torch.Tensor, apply_diversity_bonus: bool = True) -> torch.Tensor:
        """Compute probabilities with contextual information."""
        # Get base probabilities
        base_probs = super().compute_recruitment_probabilities(population_state, apply_diversity_bonus)
        
        # Add context influence
        if self.context_weight > 0:
            # Average context from memory
            context_influence = self.context_memory.mean(dim=0)
            
            # Apply context weight
            context_bias = self.context_weight * context_influence
            
            # Modify probabilities
            logits = torch.log(base_probs + 1e-10) + context_bias.unsqueeze(0)
            contextual_probs = F.softmax(logits, dim=-1)
            
            return contextual_probs
        
        return base_probs
    
    def _update_recruitment_stats(self, recruited_indices_batch: List[List[int]], probabilities: torch.Tensor):
        """Update stats and context memory."""
        super()._update_recruitment_stats(recruited_indices_batch, probabilities)
        
        # Update context memory
        if recruited_indices_batch:
            # Create recruitment pattern
            pattern = torch.zeros(self.neuron_pool_size)
            for idx in recruited_indices_batch[0]:  # Use first batch item
                pattern[idx] = 1.0
            
            # Store in circular buffer
            self.context_memory[self.context_pointer] = pattern
            self.context_pointer = (self.context_pointer + 1) % self.context_window


# Example usage and testing
if __name__ == "__main__":
    print("Testing NeuronRecruitment...")
    
    # Create a simple neuron pool
    class SimpleNeuron(nn.Module):
        def __init__(self, input_size, output_size=1):
            super().__init__()
            self.linear = nn.Linear(input_size, output_size)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create neuron pool
    neuron_pool = nn.ModuleList([
        SimpleNeuron(128) for _ in range(64)
    ])
    
    # Create recruitment mechanism
    recruitment = NeuronRecruitment(
        population_size=128,
        neuron_pool_size=64,
        recruitment_temperature=1.0,
        min_connections=2,
        max_connections=10
    )
    
    # Test recruitment
    population_state = torch.randn(4, 128)  # batch_size=4
    budget = 1.0
    
    recruited_indices, outputs, info = recruitment.recruit_neurons(
        population_state, budget, neuron_pool, training=True
    )
    
    print(f"Recruited indices: {recruited_indices}")
    print(f"Output shape: {outputs.shape}")
    print(f"Info keys: {list(info.keys())}")
    
    # Test multiple recruitment steps
    for step in range(10):
        population_state = torch.randn(2, 128)
        recruited_indices, outputs, info = recruitment.recruit_neurons(
            population_state, budget, neuron_pool, training=True
        )
    
    # Get statistics
    stats = recruitment.get_recruitment_statistics()
    print(f"Recruitment statistics: {list(stats.keys())}")
    print(f"Most recruited neurons: {stats['most_recruited_neurons'][:5]}")
    
    # Test contextual recruitment
    print("\nTesting ContextualRecruitment...")
    contextual_recruitment = ContextualRecruitment(
        population_size=128,
        neuron_pool_size=64,
        context_window=3,
        context_weight=0.3
    )
    
    # Test with context
    for step in range(5):
        population_state = torch.randn(2, 128)
        recruited_indices, outputs, info = contextual_recruitment.recruit_neurons(
            population_state, budget, neuron_pool, training=True
        )
        print(f"Step {step}: recruited {len(recruited_indices)} neurons")
    
    print("NeuronRecruitment tests passed!")