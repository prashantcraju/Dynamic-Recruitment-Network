"""
drn/layers/population_layer.py

Population-based processing layer that implements the foundational population dynamics
for Dynamic Recruitment Networks. This layer manages groups of neurons (populations)
that work together to process information and support dynamic recruitment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import math


class Population(nn.Module):
    """
    Represents a single population of neurons with shared dynamics.
    
    A population is a group of neurons that:
    - Share similar response properties
    - Can be recruited as a unit
    - Have internal recurrent connections
    - Contribute to population-level representations
    
    Args:
        size (int): Number of neurons in the population
        input_size (int): Size of input to this population
        activation (str): Activation function ('relu', 'tanh', 'sigmoid')
        recurrent_strength (float): Strength of recurrent connections within population
        inhibition_strength (float): Lateral inhibition between neurons
        adaptation_rate (float): Rate of neural adaptation
        noise_level (float): Amount of neural noise
    """
    
    def __init__(
        self,
        size: int,
        input_size: int,
        activation: str = 'relu',
        recurrent_strength: float = 0.1,
        inhibition_strength: float = 0.05,
        adaptation_rate: float = 0.01,
        noise_level: float = 0.0
    ):
        super(Population, self).__init__()
        
        if size <= 0 or input_size <= 0:
            raise ValueError("size and input_size must be positive")
        if not 0.0 <= recurrent_strength <= 1.0:
            raise ValueError("recurrent_strength must be between 0 and 1")
        if not 0.0 <= inhibition_strength <= 1.0:
            raise ValueError("inhibition_strength must be between 0 and 1")
        
        self.size = size
        self.input_size = input_size
        self.activation_name = activation
        self.recurrent_strength = recurrent_strength
        self.inhibition_strength = inhibition_strength
        self.adaptation_rate = adaptation_rate
        self.noise_level = noise_level
        
        # Input transformation
        self.input_transform = nn.Linear(input_size, size)
        
        # Recurrent connections within population
        self.recurrent_weights = nn.Parameter(torch.zeros(size, size))
        
        # Inhibitory connections (lateral inhibition)
        self.inhibition_weights = nn.Parameter(torch.zeros(size, size))
        
        # Adaptation variables (learnable per-neuron adaptation)
        self.adaptation_weights = nn.Parameter(torch.zeros(size))
        
        # Population state tracking
        self.register_buffer('previous_activation', torch.zeros(1, size))
        self.register_buffer('adaptation_state', torch.zeros(1, size))
        self.register_buffer('activity_history', torch.zeros(10, size))  # Short-term history
        self.history_pointer = 0
        
        # Population identity (learned representation of what this population encodes)
        self.population_identity = nn.Parameter(torch.randn(size) * 0.1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize population weights."""
        # Input transformation
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.zeros_(self.input_transform.bias)
        
        # Recurrent connections - positive, sparse
        nn.init.normal_(self.recurrent_weights, 0, 0.1)
        with torch.no_grad():
            self.recurrent_weights.fill_diagonal_(0)  # No self-connections
            # Make sparse by zeroing small weights
            mask = torch.abs(self.recurrent_weights) < 0.05
            self.recurrent_weights[mask] = 0
        
        # Inhibition weights - negative, denser
        nn.init.normal_(self.inhibition_weights, -0.05, 0.02)
        with torch.no_grad():
            self.inhibition_weights.fill_diagonal_(0)  # No self-inhibition
            self.inhibition_weights.clamp_(max=0)  # Ensure inhibitory
        
        # Adaptation weights
        nn.init.normal_(self.adaptation_weights, 0, 0.01)
    
    def forward(
        self, 
        x: torch.Tensor, 
        external_excitation: Optional[torch.Tensor] = None,
        reset_state: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through population.
        
        Args:
            x: Input tensor [batch_size, input_size]
            external_excitation: Additional excitation [batch_size, size]
            reset_state: Whether to reset population state
            
        Returns:
            (population_output, population_info)
        """
        batch_size = x.shape[0]
        
        # Reset state if requested
        if reset_state:
            self._reset_state(batch_size)
        
        # Ensure state tensors match batch size
        if self.previous_activation.shape[0] != batch_size:
            self._resize_state_buffers(batch_size)
        
        # Input transformation
        input_drive = self.input_transform(x)
        
        # Add external excitation if provided
        if external_excitation is not None:
            input_drive = input_drive + external_excitation
        
        # Recurrent input from previous activation
        recurrent_input = torch.matmul(
            self.previous_activation, 
            self.recurrent_weights * self.recurrent_strength
        )
        
        # Lateral inhibition
        inhibitory_input = torch.matmul(
            self.previous_activation,
            self.inhibition_weights * self.inhibition_strength
        )
        
        # Adaptation (reduces response to repeated stimuli)
        adaptation_effect = self.adaptation_state * self.adaptation_weights.unsqueeze(0)
        
        # Combined drive
        total_drive = (
            input_drive + 
            recurrent_input + 
            inhibitory_input - 
            adaptation_effect
        )
        
        # Add noise if specified
        if self.noise_level > 0 and self.training:
            noise = torch.randn_like(total_drive) * self.noise_level
            total_drive = total_drive + noise
        
        # Apply activation function
        if self.activation_name == 'relu':
            population_output = F.relu(total_drive)
        elif self.activation_name == 'tanh':
            population_output = torch.tanh(total_drive)
        elif self.activation_name == 'sigmoid':
            population_output = torch.sigmoid(total_drive)
        else:
            population_output = total_drive
        
        # Update adaptation state
        self.adaptation_state = (
            (1 - self.adaptation_rate) * self.adaptation_state + 
            self.adaptation_rate * population_output.detach()
        )
        
        # Update activity history
        with torch.no_grad():
            self.activity_history[self.history_pointer] = population_output.mean(0)
            self.history_pointer = (self.history_pointer + 1) % self.activity_history.shape[0]
        
        # Store current activation for next timestep
        self.previous_activation = population_output.detach()
        
        # Compute population info
        population_info = self._compute_population_info(
            population_output, input_drive, recurrent_input, inhibitory_input
        )
        
        return population_output, population_info
    
    def _reset_state(self, batch_size: int):
        """Reset population state."""
        self.previous_activation = torch.zeros(batch_size, self.size, device=self.input_transform.weight.device)
        self.adaptation_state = torch.zeros(batch_size, self.size, device=self.input_transform.weight.device)
        self.activity_history.zero_()
        self.history_pointer = 0
    
    def _resize_state_buffers(self, batch_size: int):
        """Resize state buffers to match batch size."""
        device = self.input_transform.weight.device
        self.previous_activation = torch.zeros(batch_size, self.size, device=device)
        if self.adaptation_state.shape[0] != batch_size:
            self.adaptation_state = torch.zeros(batch_size, self.size, device=device)
    
    def _compute_population_info(
        self, 
        output: torch.Tensor,
        input_drive: torch.Tensor,
        recurrent_input: torch.Tensor,
        inhibitory_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute population-level information."""
        
        # Activity statistics
        mean_activity = output.mean().item()
        max_activity = output.max().item()
        activity_variance = output.var().item()
        
        # Sparsity (fraction of neurons significantly active)
        active_threshold = 0.1 * max_activity if max_activity > 0 else 0.01
        sparsity = (output > active_threshold).float().mean().item()
        
        # Dynamics measures
        recurrent_strength = torch.norm(recurrent_input).item()
        inhibition_strength = torch.abs(inhibitory_input).mean().item()
        
        # Population coherence (how synchronized the population is)
        if output.shape[0] > 1:  # Need multiple samples
            neuron_correlations = torch.corrcoef(output.T)
            coherence = neuron_correlations[~torch.eye(self.size, dtype=bool)].mean().item()
        else:
            coherence = 0.0
        
        # Adaptation level
        adaptation_level = self.adaptation_state.mean().item()
        
        # Historical activity pattern
        activity_pattern_stability = self._compute_pattern_stability()
        
        return {
            'mean_activity': mean_activity,
            'max_activity': max_activity,
            'activity_variance': activity_variance,
            'sparsity': sparsity,
            'recurrent_strength': recurrent_strength,
            'inhibition_strength': inhibition_strength,
            'coherence': coherence,
            'adaptation_level': adaptation_level,
            'pattern_stability': activity_pattern_stability,
            'population_size': self.size
        }
    
    def _compute_pattern_stability(self) -> float:
        """Compute stability of activity patterns over recent history."""
        if self.activity_history.abs().sum() < 1e-6:
            return 0.0
        
        # Compute correlation between recent patterns
        recent_patterns = self.activity_history[-5:]  # Last 5 patterns
        if recent_patterns.shape[0] < 2:
            return 0.0
        
        correlations = []
        for i in range(len(recent_patterns) - 1):
            pattern1 = recent_patterns[i]
            pattern2 = recent_patterns[i + 1]
            
            # Normalize patterns
            norm1 = torch.norm(pattern1)
            norm2 = torch.norm(pattern2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                correlation = torch.dot(pattern1 / norm1, pattern2 / norm2).item()
                correlations.append(correlation)
        
        return sum(correlations) / max(1, len(correlations))
    
    def get_population_state(self) -> Dict[str, torch.Tensor]:
        """Get current population state."""
        return {
            'previous_activation': self.previous_activation.clone(),
            'adaptation_state': self.adaptation_state.clone(),
            'activity_history': self.activity_history.clone(),
            'population_identity': self.population_identity.clone()
        }


class PopulationLayer(nn.Module):
    """
    Layer that manages multiple populations with inter-population dynamics.
    
    This layer creates multiple populations that can:
    - Process different aspects of the input
    - Compete for representation
    - Collaborate through excitatory connections
    - Be selectively recruited based on input patterns
    
    Args:
        input_size (int): Size of layer input
        num_populations (int): Number of populations in the layer
        population_size (int): Size of each population
        output_size (int): Size of layer output
        competition_strength (float): Strength of competition between populations
        collaboration_strength (float): Strength of collaboration between populations
        selection_threshold (float): Threshold for population recruitment
        population_specialization (bool): Whether populations should specialize
    """
    
    def __init__(
        self,
        input_size: int,
        num_populations: int = 4,
        population_size: int = 32,
        output_size: Optional[int] = None,
        competition_strength: float = 0.1,
        collaboration_strength: float = 0.05,
        selection_threshold: float = 0.1,
        population_specialization: bool = True,
        **population_kwargs
    ):
        super(PopulationLayer, self).__init__()
        
        if input_size <= 0 or num_populations <= 0 or population_size <= 0:
            raise ValueError("All size parameters must be positive")
        
        self.input_size = input_size
        self.num_populations = num_populations
        self.population_size = population_size
        self.output_size = output_size or (num_populations * population_size)
        self.competition_strength = competition_strength
        self.collaboration_strength = collaboration_strength
        self.selection_threshold = selection_threshold
        self.population_specialization = population_specialization
        
        # Create populations
        self.populations = nn.ModuleList([
            Population(
                size=population_size,
                input_size=input_size,
                **population_kwargs
            )
            for _ in range(num_populations)
        ])
        
        # Inter-population connections
        if num_populations > 1:
            # Competition matrix (inhibitory)
            self.competition_matrix = nn.Parameter(
                torch.zeros(num_populations, num_populations)
            )
            
            # Collaboration matrix (excitatory)
            self.collaboration_matrix = nn.Parameter(
                torch.zeros(num_populations, num_populations)
            )
        
        # Population selection mechanism
        self.population_selector = nn.Linear(input_size, num_populations)
        
        # Output projection
        total_population_output = num_populations * population_size
        if self.output_size != total_population_output:
            self.output_projection = nn.Linear(total_population_output, self.output_size)
        else:
            self.output_projection = None
        
        # Population specialization encouragement
        if population_specialization:
            self.specialization_centers = nn.Parameter(
                torch.randn(num_populations, input_size) * 0.1
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize layer weights."""
        # Population selector
        nn.init.xavier_uniform_(self.population_selector.weight)
        nn.init.zeros_(self.population_selector.bias)
        
        # Inter-population matrices
        if self.num_populations > 1:
            # Competition - negative, no self-competition
            nn.init.normal_(self.competition_matrix, -0.1, 0.02)
            with torch.no_grad():
                self.competition_matrix.fill_diagonal_(0)
                self.competition_matrix.clamp_(max=0)
            
            # Collaboration - positive, no self-collaboration
            nn.init.normal_(self.collaboration_matrix, 0.05, 0.02)
            with torch.no_grad():
                self.collaboration_matrix.fill_diagonal_(0)
                self.collaboration_matrix.clamp_(min=0)
        
        # Output projection
        if self.output_projection is not None:
            nn.init.xavier_uniform_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        force_all_populations: bool = False,
        reset_populations: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through population layer.
        
        Args:
            x: Input tensor [batch_size, input_size]
            force_all_populations: Whether to activate all populations
            reset_populations: Whether to reset population states
            
        Returns:
            (layer_output, layer_info)
        """
        batch_size = x.shape[0]
        
        # Compute population selection scores
        selection_scores = self.population_selector(x)
        selection_probs = torch.softmax(selection_scores, dim=-1)
        
        # Determine which populations to activate
        if force_all_populations:
            active_populations = torch.ones_like(selection_probs)
        else:
            active_populations = (selection_probs > self.selection_threshold).float()
            # Ensure at least one population is active
            if active_populations.sum(dim=-1).min() == 0:
                max_indices = selection_probs.argmax(dim=-1)
                active_populations[torch.arange(batch_size), max_indices] = 1.0
        
        # Apply population specialization bias
        if self.population_specialization:
            specialization_scores = self._compute_specialization_scores(x)
            selection_probs = selection_probs * specialization_scores
        
        # Process through each population
        population_outputs = []
        population_infos = []
        
        for i, population in enumerate(self.populations):
            # Check if this population should be active
            pop_active = active_populations[:, i].unsqueeze(-1)
            
            # Compute inter-population influence
            inter_pop_input = None
            if self.num_populations > 1 and len(population_outputs) > 0:
                inter_pop_input = self._compute_inter_population_influence(
                    population_outputs, i, pop_active
                )
            
            # Forward through population
            pop_input = x * pop_active  # Gate input by activation
            pop_output, pop_info = population(
                pop_input,
                external_excitation=inter_pop_input,
                reset_state=reset_populations
            )
            
            # Scale output by selection probability
            pop_output = pop_output * selection_probs[:, i].unsqueeze(-1)
            
            population_outputs.append(pop_output)
            population_infos.append(pop_info)
        
        # Combine population outputs
        combined_output = torch.cat(population_outputs, dim=-1)
        
        # Apply output projection if needed
        if self.output_projection is not None:
            layer_output = self.output_projection(combined_output)
        else:
            layer_output = combined_output
        
        # Compile layer information
        layer_info = self._compile_layer_info(
            population_infos, selection_probs, active_populations
        )
        
        return layer_output, layer_info
    
    def _compute_specialization_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute how well each population's specialization matches the input."""
        # Distance from input to each specialization center
        distances = torch.cdist(
            x.unsqueeze(1), 
            self.specialization_centers.unsqueeze(0)
        ).squeeze(1)
        
        # Convert distances to similarity scores
        similarities = torch.exp(-distances / (2 * 0.5**2))  # Gaussian similarity
        
        # Normalize to probabilities
        return F.softmax(similarities, dim=-1)
    
    def _compute_inter_population_influence(
        self, 
        population_outputs: List[torch.Tensor], 
        current_pop: int,
        activation_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute influence from other populations on current population."""
        if not population_outputs:
            return None
        
        batch_size = population_outputs[0].shape[0]
        influence = torch.zeros(batch_size, self.population_size, device=population_outputs[0].device)
        
        for i, other_output in enumerate(population_outputs):
            if i == current_pop:
                continue
            
            # Pool other population's activity
            other_activity = other_output.mean(dim=-1, keepdim=True)  # [batch_size, 1]
            
            # Competition (inhibition)
            competition_weight = self.competition_matrix[current_pop, i]
            competition_effect = competition_weight * other_activity * self.competition_strength
            
            # Collaboration (excitation)
            collaboration_weight = self.collaboration_matrix[current_pop, i]
            collaboration_effect = collaboration_weight * other_activity * self.collaboration_strength
            
            # Total influence
            total_influence = competition_effect + collaboration_effect
            influence += total_influence.expand(-1, self.population_size)
        
        return influence * activation_mask
    
    def _compile_layer_info(
        self, 
        population_infos: List[Dict],
        selection_probs: torch.Tensor,
        active_populations: torch.Tensor
    ) -> Dict[str, Any]:
        """Compile comprehensive layer information."""
        
        # Aggregate population metrics
        total_populations = len(population_infos)
        active_count = active_populations.sum(dim=-1).mean().item()
        
        # Population statistics
        mean_activities = [info['mean_activity'] for info in population_infos]
        sparsities = [info['sparsity'] for info in population_infos]
        coherences = [info['coherence'] for info in population_infos]
        
        # Selection statistics
        selection_entropy = -torch.sum(
            selection_probs * torch.log(selection_probs + 1e-10), dim=-1
        ).mean().item()
        
        # Specialization measure
        specialization_score = self._compute_specialization_measure(selection_probs)
        
        layer_info = {
            # Population activity
            'num_populations': total_populations,
            'active_populations': active_count,
            'population_activities': mean_activities,
            'population_sparsities': sparsities,
            'population_coherences': coherences,
            
            # Selection dynamics
            'selection_entropy': selection_entropy,
            'selection_probabilities': selection_probs.mean(0).tolist(),
            'specialization_score': specialization_score,
            
            # Layer-level metrics
            'layer_sparsity': sum(sparsities) / len(sparsities),
            'layer_coherence': sum(coherences) / len(coherences),
            'population_utilization': active_count / total_populations,
            
            # Competition/collaboration
            'competition_strength': self.competition_strength,
            'collaboration_strength': self.collaboration_strength,
            
            # Individual population info
            'population_infos': population_infos
        }
        
        return layer_info
    
    def _compute_specialization_measure(self, selection_probs: torch.Tensor) -> float:
        """Compute how specialized the populations are."""
        # High specialization = low entropy in selection probabilities
        # across the batch (each input should prefer specific populations)
        
        # Mean selection probability for each population across batch
        mean_probs = selection_probs.mean(0)
        
        # Entropy of mean probabilities (lower = more specialized)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10)).item()
        max_entropy = math.log(self.num_populations)
        
        # Specialization score (1 = fully specialized, 0 = no specialization)
        specialization = 1 - (entropy / max_entropy)
        
        return specialization
    
    def reset_all_populations(self):
        """Reset all population states."""
        for population in self.populations:
            population._reset_state(1)  # Reset with batch size 1
    
    def get_population_states(self) -> List[Dict[str, torch.Tensor]]:
        """Get states of all populations."""
        return [pop.get_population_state() for pop in self.populations]
    
    def __repr__(self) -> str:
        return (f"PopulationLayer(input_size={self.input_size}, "
                f"num_populations={self.num_populations}, "
                f"population_size={self.population_size}, "
                f"output_size={self.output_size})")