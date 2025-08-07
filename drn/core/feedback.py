"""
drn/core/feedback.py

Feedback Manager component that handles recurrent connections from recruited
neurons back to the base population. This creates dynamic population loops
unlike traditional feedforward architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import math


class FeedbackManager(nn.Module):
    """
    Manages recurrent feedback from recruited neurons to base population.
    
    In traditional networks, information flows strictly forward through layers.
    In DRNs, recruited neurons provide feedback to influence ongoing population
    activity, creating dynamic loops that enhance processing flexibility.
    
    Args:
        population_size (int): Size of the base population
        max_feedback_sources (int): Maximum number of neurons that can provide feedback
        feedback_strength (float): Overall strength of feedback connections
        adaptive_feedback (bool): Whether to learn feedback strengths
        feedback_delay (int): Number of steps to delay feedback (0 = immediate)
        normalization (str): Feedback normalization method ('none', 'l2', 'batch')
        
    Example:
        >>> feedback_mgr = FeedbackManager(population_size=128, max_feedback_sources=64)
        >>> recruited_outputs = torch.randn(32, 64)  # batch_size=32, 64 recruited
        >>> recruited_indices = list(range(64))
        >>> feedback = feedback_mgr(recruited_outputs, recruited_indices)
    """
    
    def __init__(
        self,
        population_size: int,
        max_feedback_sources: int,
        feedback_strength: float = 0.1,
        adaptive_feedback: bool = True,
        feedback_delay: int = 0,
        normalization: str = 'l2'
    ):
        super(FeedbackManager, self).__init__()
        
        # Validate inputs
        if population_size <= 0 or max_feedback_sources <= 0:
            raise ValueError("population_size and max_feedback_sources must be positive")
        if not 0.0 <= feedback_strength <= 1.0:
            raise ValueError("feedback_strength must be between 0.0 and 1.0")
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be non-negative")
        if normalization not in ['none', 'l2', 'batch']:
            raise ValueError("normalization must be 'none', 'l2', or 'batch'")
        
        self.population_size = population_size
        self.max_feedback_sources = max_feedback_sources
        self.feedback_strength = feedback_strength
        self.adaptive_feedback = adaptive_feedback
        self.feedback_delay = feedback_delay
        self.normalization = normalization
        
        # Feedback transformation weights
        # Maps from recruited neuron outputs back to population space
        self.feedback_transform = nn.Linear(1, population_size, bias=False)  # Each neuron gives 1 output
        
        # Adaptive feedback weights (if enabled)
        if self.adaptive_feedback:
            # Learnable weight for each potential feedback source
            self.feedback_weights = nn.Parameter(
                torch.ones(max_feedback_sources) * feedback_strength
            )
        else:
            # Fixed feedback weights
            self.register_buffer('feedback_weights', 
                               torch.ones(max_feedback_sources) * feedback_strength)
        
        # Feedback delay buffer
        if self.feedback_delay > 0:
            self.register_buffer('feedback_buffer', 
                               torch.zeros(feedback_delay, 1, population_size))
            self.register_buffer('buffer_pointer', torch.tensor(0))
        
        # Feedback statistics tracking
        self.register_buffer('total_feedback_steps', torch.tensor(0))
        self.register_buffer('feedback_magnitude_sum', torch.tensor(0.0))
        
        # History for analysis
        self._feedback_history: List[torch.Tensor] = []
        self._source_history: List[List[int]] = []
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize feedback transformation weights."""
        nn.init.xavier_uniform_(self.feedback_transform.weight)
        
        # Scale down initial weights for stability
        self.feedback_transform.weight.data *= 0.1
    
    def forward(
        self,
        recruited_outputs: torch.Tensor,
        recruited_indices: List[int],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute feedback from recruited neurons to population.
        
        Args:
            recruited_outputs: Outputs from recruited neurons [batch_size, num_recruited]
            recruited_indices: Indices of recruited neurons
            batch_size: Batch size (inferred if not provided)
            
        Returns:
            Feedback tensor to add to population [batch_size, population_size]
        """
        if recruited_outputs.numel() == 0 or not recruited_indices:
            # No recruited neurons - return zero feedback
            if batch_size is None:
                batch_size = 1
            return torch.zeros(batch_size, self.population_size, 
                             device=recruited_outputs.device)
        
        actual_batch_size = recruited_outputs.shape[0]
        num_recruited = len(recruited_indices)
        
        # Process each recruited neuron's contribution
        feedback_contributions = []
        
        for i, neuron_idx in enumerate(recruited_indices):
            if i >= self.max_feedback_sources:
                break  # Respect maximum feedback sources
            
            # Get this neuron's output for all batch items
            if i < recruited_outputs.shape[1]:
                neuron_output = recruited_outputs[:, i:i+1]  # [batch_size, 1]
            else:
                # Handle case where recruited_outputs has fewer columns than indices
                neuron_output = torch.zeros(actual_batch_size, 1, 
                                          device=recruited_outputs.device)
            
            # Transform to population space
            feedback_contrib = self.feedback_transform(neuron_output)  # [batch_size, population_size]
            
            # Apply adaptive weighting
            if self.adaptive_feedback and i < len(self.feedback_weights):
                feedback_contrib = feedback_contrib * self.feedback_weights[i]
            
            feedback_contributions.append(feedback_contrib)
        
        if not feedback_contributions:
            return torch.zeros(actual_batch_size, self.population_size,
                             device=recruited_outputs.device)
        
        # Combine all feedback contributions
        total_feedback = torch.stack(feedback_contributions, dim=0).sum(dim=0)
        
        # Apply normalization
        total_feedback = self._apply_normalization(total_feedback)
        
        # Apply delay if specified
        if self.feedback_delay > 0:
            total_feedback = self._apply_delay(total_feedback)
        
        # Update statistics
        self._update_feedback_stats(total_feedback, recruited_indices)
        
        return total_feedback
    
    def _apply_normalization(self, feedback: torch.Tensor) -> torch.Tensor:
        """Apply specified normalization to feedback."""
        if self.normalization == 'none':
            return feedback
        elif self.normalization == 'l2':
            # L2 normalize across population dimension
            norm = torch.norm(feedback, p=2, dim=-1, keepdim=True)
            return feedback / (norm + 1e-8)
        elif self.normalization == 'batch':
            # Batch normalize across batch dimension
            mean = feedback.mean(dim=0, keepdim=True)
            std = feedback.std(dim=0, keepdim=True)
            return (feedback - mean) / (std + 1e-8)
        
        return feedback
    
    def _apply_delay(self, feedback: torch.Tensor) -> torch.Tensor:
        """Apply feedback delay using circular buffer."""
        # Store current feedback in buffer
        self.feedback_buffer[self.buffer_pointer] = feedback.mean(dim=0, keepdim=True)
        
        # Get delayed feedback
        delayed_pointer = (self.buffer_pointer - self.feedback_delay) % self.feedback_delay
        delayed_feedback = self.feedback_buffer[delayed_pointer]
        
        # Update pointer
        self.buffer_pointer = (self.buffer_pointer + 1) % self.feedback_delay
        
        # Expand to match batch size
        batch_size = feedback.shape[0]
        return delayed_feedback.expand(batch_size, -1)
    
    def _update_feedback_stats(self, feedback: torch.Tensor, recruited_indices: List[int]):
        """Update feedback statistics for analysis."""
        self.total_feedback_steps += 1
        
        # Track feedback magnitude
        feedback_magnitude = torch.norm(feedback).item()
        self.feedback_magnitude_sum += feedback_magnitude
        
        # Store history
        self._feedback_history.append(feedback.detach().clone())
        self._source_history.append(recruited_indices.copy())
        
        # Limit history size
        if len(self._feedback_history) > 1000:
            self._feedback_history.pop(0)
            self._source_history.pop(0)
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics."""
        if self.total_feedback_steps.item() == 0:
            return {'total_steps': 0, 'message': 'No feedback steps yet'}
        
        avg_magnitude = self.feedback_magnitude_sum.item() / self.total_feedback_steps.item()
        
        stats = {
            'total_feedback_steps': self.total_feedback_steps.item(),
            'average_feedback_magnitude': avg_magnitude,
            'feedback_strength': self.feedback_strength,
            'adaptive_feedback': self.adaptive_feedback
        }
        
        if self.adaptive_feedback:
            stats['learned_feedback_weights'] = self.feedback_weights.detach().cpu().numpy().tolist()
            stats['feedback_weight_variance'] = self.feedback_weights.var().item()
        
        return stats
    
    def get_feedback_history(self) -> Dict[str, List]:
        """Get feedback history for visualization."""
        return {
            'feedback_history': [f.cpu().numpy() for f in self._feedback_history],
            'source_history': self._source_history.copy()
        }
    
    def reset_feedback_stats(self):
        """Reset feedback statistics."""
        self.total_feedback_steps.zero_()
        self.feedback_magnitude_sum.zero_()
        self._feedback_history.clear()
        self._source_history.clear()
        
        if self.feedback_delay > 0:
            self.feedback_buffer.zero_()
            self.buffer_pointer.zero_()
    
    def set_feedback_strength(self, strength: float):
        """Adjust overall feedback strength."""
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Feedback strength must be between 0.0 and 1.0")
        
        self.feedback_strength = strength
        
        if not self.adaptive_feedback:
            self.feedback_weights.fill_(strength)
    
    def get_effective_feedback_weights(self) -> torch.Tensor:
        """Get current effective feedback weights."""
        if self.adaptive_feedback:
            return torch.sigmoid(self.feedback_weights) * self.feedback_strength
        else:
            return self.feedback_weights
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'population_size={self.population_size}, '
                f'max_feedback_sources={self.max_feedback_sources}, '
                f'feedback_strength={self.feedback_strength}, '
                f'adaptive={self.adaptive_feedback}')


class HierarchicalFeedbackManager(FeedbackManager):
    """
    Extended feedback manager with hierarchical feedback patterns.
    
    This version can handle feedback at multiple timescales and levels,
    more closely mimicking biological neural feedback systems.
    """
    
    def __init__(
        self,
        feedback_levels: List[Dict[str, Any]],
        **kwargs
    ):
        """
        Args:
            feedback_levels: List of feedback level configurations
                Each dict should contain: {'timescale': int, 'strength': float}
        """
        super().__init__(**kwargs)
        
        self.feedback_levels = feedback_levels
        
        # Create separate feedback transforms for each level
        self.level_transforms = nn.ModuleList([
            nn.Linear(1, self.population_size, bias=False)
            for _ in feedback_levels
        ])
        
        # Initialize level-specific weights
        for transform in self.level_transforms:
            nn.init.xavier_uniform_(transform.weight)
            transform.weight.data *= 0.05  # Even smaller for stability
    
    def forward(
        self,
        recruited_outputs: torch.Tensor,
        recruited_indices: List[int],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute hierarchical feedback with multiple timescales."""
        # Get base feedback
        base_feedback = super().forward(recruited_outputs, recruited_indices, batch_size)
        
        if recruited_outputs.numel() == 0:
            return base_feedback
        
        # Add level-specific feedback
        hierarchical_feedback = base_feedback
        
        for level_idx, level_config in enumerate(self.feedback_levels):
            timescale = level_config.get('timescale', 1)
            strength = level_config.get('strength', 0.05)
            
            # Only apply this level's feedback at its timescale
            if self.total_feedback_steps.item() % timescale == 0:
                level_feedback = self._compute_level_feedback(
                    recruited_outputs, recruited_indices, level_idx, strength
                )
                hierarchical_feedback = hierarchical_feedback + level_feedback
        
        return hierarchical_feedback
    
    def _compute_level_feedback(
        self,
        recruited_outputs: torch.Tensor,
        recruited_indices: List[int],
        level_idx: int,
        strength: float
    ) -> torch.Tensor:
        """Compute feedback for a specific hierarchical level."""
        if level_idx >= len(self.level_transforms):
            return torch.zeros_like(recruited_outputs[:, :1] @ 
                                  torch.zeros(1, self.population_size))
        
        # Use different neurons for different levels
        level_indices = recruited_indices[level_idx::len(self.feedback_levels)]
        
        if not level_indices:
            return torch.zeros(recruited_outputs.shape[0], self.population_size,
                             device=recruited_outputs.device)
        
        # Average outputs for this level
        level_outputs = recruited_outputs[:, :len(level_indices)]
        if level_outputs.shape[1] == 0:
            return torch.zeros(recruited_outputs.shape[0], self.population_size,
                             device=recruited_outputs.device)
        
        avg_output = level_outputs.mean(dim=1, keepdim=True)
        
        # Transform to population space
        level_feedback = self.level_transforms[level_idx](avg_output)
        
        return level_feedback * strength


# Example usage and testing
if __name__ == "__main__":
    print("Testing FeedbackManager...")
    
    # Create feedback manager
    feedback_mgr = FeedbackManager(
        population_size=64,
        max_feedback_sources=32,
        feedback_strength=0.2,
        adaptive_feedback=True
    )
    
    # Test basic feedback
    recruited_outputs = torch.randn(8, 16)  # batch_size=8, 16 recruited neurons
    recruited_indices = list(range(16))
    
    feedback = feedback_mgr(recruited_outputs, recruited_indices)
    print(f"Feedback shape: {feedback.shape}")
    print(f"Feedback magnitude: {torch.norm(feedback).item():.4f}")
    
    # Test with no recruited neurons
    empty_outputs = torch.empty(8, 0)
    empty_indices = []
    empty_feedback = feedback_mgr(empty_outputs, empty_indices)
    print(f"Empty feedback shape: {empty_feedback.shape}")
    print(f"Empty feedback magnitude: {torch.norm(empty_feedback).item():.4f}")
    
    # Test multiple steps
    for step in range(10):
        recruited_outputs = torch.randn(4, 8)
        recruited_indices = list(range(8))
        feedback = feedback_mgr(recruited_outputs, recruited_indices)
    
    # Get statistics
    stats = feedback_mgr.get_feedback_statistics()
    print(f"Feedback statistics: {list(stats.keys())}")
    print(f"Average magnitude: {stats['average_feedback_magnitude']:.4f}")
    
    # Test hierarchical feedback
    print("\nTesting HierarchicalFeedbackManager...")
    
    hierarchical_feedback = HierarchicalFeedbackManager(
        population_size=64,
        max_feedback_sources=32,
        feedback_levels=[
            {'timescale': 1, 'strength': 0.1},  # Fast feedback
            {'timescale': 3, 'strength': 0.05}, # Medium feedback
            {'timescale': 5, 'strength': 0.02}  # Slow feedback
        ]
    )
    
    # Test hierarchical feedback
    for step in range(10):
        recruited_outputs = torch.randn(4, 12)
        recruited_indices = list(range(12))
        h_feedback = hierarchical_feedback(recruited_outputs, recruited_indices)
        if step % 3 == 0:
            print(f"Step {step}: hierarchical feedback magnitude: {torch.norm(h_feedback).item():.4f}")
    
    print("FeedbackManager tests passed!")