"""
drn/core/budget_manager.py

Budget Manager component that handles neurotransmitter-like resource allocation.
This is a key innovation that constrains connectivity formation, unlike traditional
networks where all connections are always available.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
import warnings
import math


class BudgetManager(nn.Module):
    """
    Manages neurotransmitter-like budget for dynamic connection formation.
    
    The budget represents a limited resource that gets depleted as neurons are recruited.
    This creates natural sparsity and forces the network to be selective about which
    connections to form, mimicking biological neurotransmitter constraints.
    
    Args:
        initial_budget (float): Starting budget amount (typically 1.0)
        connection_cost (float): Cost per recruited neuron (0.0 to 1.0)
        budget_decay_rate (float): Exponential decay rate per time step
        budget_recovery_rate (float): Recovery rate when no connections are made
        min_budget (float): Minimum budget level (prevents complete depletion)
        adaptive_cost (bool): Whether to adapt cost based on network size
        
    Example:
        >>> budget = BudgetManager(initial_budget=1.0, connection_cost=0.1)
        >>> can_afford = budget.can_afford_connections(5)
        >>> if can_afford:
        ...     actual_cost = budget.spend_budget(5)
        >>> remaining = budget.get_current_budget()
    """
    
    def __init__(
        self,
        initial_budget: float = 1.0,
        connection_cost: float = 0.1,
        budget_decay_rate: float = 0.0,
        budget_recovery_rate: float = 0.01,
        min_budget: float = 0.01,
        adaptive_cost: bool = False
    ):
        super(BudgetManager, self).__init__()
        
        # Validate inputs
        if initial_budget <= 0:
            raise ValueError("initial_budget must be positive")
        if not 0 <= connection_cost <= 1:
            raise ValueError("connection_cost must be between 0 and 1")
        if budget_decay_rate < 0:
            raise ValueError("budget_decay_rate must be non-negative")
        if budget_recovery_rate < 0:
            raise ValueError("budget_recovery_rate must be non-negative")
        if not 0 <= min_budget <= initial_budget:
            raise ValueError("min_budget must be between 0 and initial_budget")
        
        self.initial_budget = initial_budget
        self.base_connection_cost = connection_cost
        self.budget_decay_rate = budget_decay_rate
        self.budget_recovery_rate = budget_recovery_rate
        self.min_budget = min_budget
        self.adaptive_cost = adaptive_cost
        
        # State tracking
        self.register_buffer('_current_budget', torch.tensor(initial_budget))
        self.register_buffer('_time_step', torch.tensor(0))
        self.register_buffer('_total_spent', torch.tensor(0.0))
        self.register_buffer('_total_connections', torch.tensor(0))
        
        # History tracking for analysis
        self._budget_history: List[float] = []
        self._connection_history: List[int] = []
        self._cost_history: List[float] = []
        
    def can_afford_connections(self, num_connections: int, neuron_pool_size: Optional[int] = None) -> bool:
        """
        Check if current budget can afford the requested number of connections.
        
        Args:
            num_connections: Number of connections to check
            neuron_pool_size: Size of neuron pool (for adaptive cost calculation)
            
        Returns:
            Boolean indicating if connections are affordable
        """
        if num_connections <= 0:
            return True
        
        cost_per_connection = self._compute_connection_cost(neuron_pool_size)
        total_cost = num_connections * cost_per_connection
        
        return self._current_budget.item() >= total_cost
    
    def spend_budget(self, num_connections: int, neuron_pool_size: Optional[int] = None) -> float:
        """
        Spend budget on connections and return actual cost.
        
        Args:
            num_connections: Number of connections to make
            neuron_pool_size: Size of neuron pool (for adaptive cost)
            
        Returns:
            Actual cost spent
        """
        if num_connections <= 0:
            return 0.0
        
        cost_per_connection = self._compute_connection_cost(neuron_pool_size)
        total_cost = num_connections * cost_per_connection
        
        # Clip to available budget
        actual_cost = min(total_cost, self._current_budget.item() - self.min_budget)
        actual_cost = max(0.0, actual_cost)  # Ensure non-negative
        
        # Update budget
        self._current_budget -= actual_cost
        self._total_spent += actual_cost
        self._total_connections += num_connections
        
        # Record history
        self._budget_history.append(self._current_budget.item())
        self._connection_history.append(num_connections)
        self._cost_history.append(actual_cost)
        
        return actual_cost
    
    def _compute_connection_cost(self, neuron_pool_size: Optional[int] = None) -> float:
        """Compute cost per connection, potentially adapting based on pool size."""
        cost = self.base_connection_cost
        
        if self.adaptive_cost and neuron_pool_size is not None:
            # Scale cost based on pool size - larger pools have lower per-connection cost
            # This encourages exploration when many options are available
            scale_factor = math.log(neuron_pool_size + 1) / math.log(100)  # Normalize to ~100 neurons
            cost = self.base_connection_cost * (0.5 + 0.5 * scale_factor)
        
        return cost
    
    def update_budget(self, time_step: Optional[int] = None):
        """
        Update budget based on time-based decay and recovery.
        Call this at each forward pass to simulate biological dynamics.
        """
        if time_step is not None:
            self._time_step = torch.tensor(time_step)
        else:
            self._time_step += 1
        
        current_budget = self._current_budget.item()
        
        # Apply decay
        if self.budget_decay_rate > 0:
            decay_amount = current_budget * self.budget_decay_rate
            current_budget -= decay_amount
        
        # Apply recovery (moves toward initial budget)
        if self.budget_recovery_rate > 0:
            recovery_target = self.initial_budget
            recovery_amount = (recovery_target - current_budget) * self.budget_recovery_rate
            current_budget += recovery_amount
        
        # Enforce bounds
        current_budget = max(self.min_budget, min(self.initial_budget, current_budget))
        
        self._current_budget = torch.tensor(current_budget)
    
    def get_current_budget(self) -> float:
        """Get current available budget."""
        return self._current_budget.item()
    
    def get_budget_utilization(self) -> float:
        """Get fraction of budget currently used (0.0 = full, 1.0 = depleted)."""
        return 1.0 - (self._current_budget.item() / self.initial_budget)
    
    def reset_budget(self):
        """Reset budget to initial state."""
        self._current_budget = torch.tensor(self.initial_budget)
        self._time_step = torch.tensor(0)
        self._total_spent = torch.tensor(0.0)
        self._total_connections = torch.tensor(0)
        
        # Clear history
        self._budget_history.clear()
        self._connection_history.clear()
        self._cost_history.clear()
    
    def get_budget_info(self) -> Dict[str, Any]:
        """
        Get comprehensive budget information for analysis.
        
        Returns:
            Dictionary with budget statistics
        """
        return {
            'current_budget': self._current_budget.item(),
            'initial_budget': self.initial_budget,
            'budget_utilization': self.get_budget_utilization(),
            'total_spent': self._total_spent.item(),
            'total_connections': self._total_connections.item(),
            'time_step': self._time_step.item(),
            'avg_cost_per_connection': (
                self._total_spent.item() / max(1, self._total_connections.item())
            ),
            'budget_efficiency': (
                self._total_connections.item() / max(0.001, self._total_spent.item())
            )
        }
    
    def get_budget_history(self) -> Dict[str, List]:
        """Get historical budget data for visualization."""
        return {
            'budget_history': self._budget_history.copy(),
            'connection_history': self._connection_history.copy(),
            'cost_history': self._cost_history.copy()
        }
    
    def set_budget_parameters(self, **kwargs):
        """
        Update budget parameters during training.
        
        Useful for curriculum learning or adaptive training.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}")
    
    def get_max_affordable_connections(self, neuron_pool_size: Optional[int] = None) -> int:
        """
        Calculate maximum number of connections affordable with current budget.
        
        Args:
            neuron_pool_size: Size of neuron pool (for adaptive cost)
            
        Returns:
            Maximum number of affordable connections
        """
        cost_per_connection = self._compute_connection_cost(neuron_pool_size)
        available_budget = self._current_budget.item() - self.min_budget
        
        if cost_per_connection <= 0:
            return float('inf')  # Unlimited if cost is zero
        
        return int(available_budget / cost_per_connection)
    
    def simulate_budget_trajectory(self, num_steps: int, connections_per_step: List[int]) -> List[float]:
        """
        Simulate budget trajectory for planning.
        
        Args:
            num_steps: Number of time steps to simulate
            connections_per_step: List of connection counts for each step
            
        Returns:
            List of budget values at each step
        """
        # Save current state
        original_budget = self._current_budget.item()
        original_time = self._time_step.item()
        
        trajectory = [original_budget]
        
        for step, num_conn in enumerate(connections_per_step[:num_steps]):
            self.spend_budget(num_conn)
            self.update_budget()
            trajectory.append(self._current_budget.item())
        
        # Restore original state
        self._current_budget = torch.tensor(original_budget)
        self._time_step = torch.tensor(original_time)
        
        return trajectory
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'initial_budget={self.initial_budget}, '
                f'connection_cost={self.base_connection_cost}, '
                f'current_budget={self._current_budget.item():.3f}')


class AdaptiveBudgetManager(BudgetManager):
    """
    Extended budget manager with adaptive budget allocation based on performance.
    
    This version can adjust budget parameters based on learning progress,
    implementing a form of meta-learning for resource allocation.
    """
    
    def __init__(self, adaptation_rate: float = 0.01, performance_window: int = 100, **kwargs):
        super().__init__(**kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        
        # Performance tracking
        self._performance_history: List[float] = []
        self._adaptation_history: List[Dict[str, float]] = []
    
    def update_based_on_performance(self, performance_metric: float):
        """
        Adapt budget parameters based on performance feedback.
        
        Args:
            performance_metric: Recent performance measure (higher = better)
        """
        self._performance_history.append(performance_metric)
        
        # Keep only recent history
        if len(self._performance_history) > self.performance_window:
            self._performance_history.pop(0)
        
        if len(self._performance_history) < 2:
            return  # Need at least 2 points for trend
        
        # Compute performance trend
        recent_performance = sum(self._performance_history[-10:]) / min(10, len(self._performance_history))
        older_performance = sum(self._performance_history[:-10]) / max(1, len(self._performance_history) - 10)
        
        performance_trend = recent_performance - older_performance
        
        # Adapt budget parameters
        if performance_trend > 0:  # Performance improving
            # Slightly increase budget to explore more
            self.initial_budget = min(2.0, self.initial_budget * (1 + self.adaptation_rate))
        else:  # Performance declining
            # Slightly decrease budget to force more selective connections
            self.initial_budget = max(0.1, self.initial_budget * (1 - self.adaptation_rate))
        
        # Record adaptation
        self._adaptation_history.append({
            'performance_trend': performance_trend,
            'new_budget': self.initial_budget,
            'time_step': self._time_step.item()
        })


# Example usage and testing
if __name__ == "__main__":
    print("Testing BudgetManager...")
    
    # Create budget manager
    budget = BudgetManager(
        initial_budget=1.0,
        connection_cost=0.1,
        budget_decay_rate=0.01,
        budget_recovery_rate=0.005
    )
    
    print(f"Initial budget: {budget.get_current_budget()}")
    
    # Test affordability check
    can_afford = budget.can_afford_connections(5)
    print(f"Can afford 5 connections: {can_afford}")
    
    # Spend some budget
    if can_afford:
        cost = budget.spend_budget(5)
        print(f"Spent {cost} on 5 connections")
    
    print(f"Budget after spending: {budget.get_current_budget()}")
    
    # Simulate time progression
    for step in range(5):
        budget.update_budget()
        connections = 2 if step % 2 == 0 else 0
        if budget.can_afford_connections(connections):
            budget.spend_budget(connections)
        print(f"Step {step}: budget={budget.get_current_budget():.3f}, "
              f"utilization={budget.get_budget_utilization():.3f}")
    
    # Get comprehensive info
    info = budget.get_budget_info()
    print(f"Budget info: {info}")
    
    # Test adaptive version
    print("\nTesting AdaptiveBudgetManager...")
    adaptive_budget = AdaptiveBudgetManager(adaptation_rate=0.05)
    
    # Simulate performance feedback
    for i in range(20):
        performance = 0.5 + 0.02 * i + 0.1 * (i % 3)  # Improving with noise
        adaptive_budget.update_based_on_performance(performance)
    
    print(f"Final adaptive budget: {adaptive_budget.initial_budget:.3f}")
    
    print("BudgetManager tests passed!")