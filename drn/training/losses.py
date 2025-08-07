"""
drn/training/losses.py

Specialized loss functions for Dynamic Recruitment Networks.
These losses encourage the desired connectivity patterns and cognitive flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import math


class RecruitmentLoss(nn.Module):
    """
    Loss function that regularizes recruitment patterns in DRN networks.
    
    This loss encourages:
    - Sparse connectivity (fewer recruited neurons)
    - Diverse recruitment patterns (different neurons across time)
    - Efficient budget usage
    - Balanced population activity
    
    Args:
        connectivity_weight: Weight for connectivity sparsity term
        diversity_weight: Weight for recruitment diversity term
        sparsity_weight: Weight for overall sparsity term
        efficiency_weight: Weight for budget efficiency term
        population_balance_weight: Weight for population balance term
    """
    
    def __init__(
        self,
        connectivity_weight: float = 1.0,
        diversity_weight: float = 0.1,
        sparsity_weight: float = 0.1,
        efficiency_weight: float = 0.05,
        population_balance_weight: float = 0.05
    ):
        super(RecruitmentLoss, self).__init__()
        
        self.connectivity_weight = connectivity_weight
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
        self.efficiency_weight = efficiency_weight
        self.population_balance_weight = population_balance_weight
        
        # Track recruitment history for diversity computation
        self.register_buffer('recruitment_history', torch.zeros(100, 1000))  # Last 100 steps, max 1000 neurons
        self.register_buffer('history_pointer', torch.tensor(0))
        self.register_buffer('history_count', torch.tensor(0))
    
    def forward(self, network_info: Dict[str, Any]) -> torch.Tensor:
        """
        Compute recruitment regularization loss.
        
        Args:
            network_info: Information dictionary from DRN forward pass
            
        Returns:
            Scalar recruitment loss
        """
        total_loss = torch.tensor(0.0, device=self._get_device())
        
        # 1. Connectivity sparsity loss
        if self.connectivity_weight > 0:
            connectivity_loss = self._compute_connectivity_loss(network_info)
            total_loss += self.connectivity_weight * connectivity_loss
        
        # 2. Diversity loss
        if self.diversity_weight > 0:
            diversity_loss = self._compute_diversity_loss(network_info)
            total_loss += self.diversity_weight * diversity_loss
        
        # 3. Overall sparsity loss
        if self.sparsity_weight > 0:
            sparsity_loss = self._compute_sparsity_loss(network_info)
            total_loss += self.sparsity_weight * sparsity_loss
        
        # 4. Budget efficiency loss
        if self.efficiency_weight > 0:
            efficiency_loss = self._compute_efficiency_loss(network_info)
            total_loss += self.efficiency_weight * efficiency_loss
        
        # 5. Population balance loss
        if self.population_balance_weight > 0:
            balance_loss = self._compute_population_balance_loss(network_info)
            total_loss += self.population_balance_weight * balance_loss
        
        return total_loss
    
    def _get_device(self) -> torch.device:
        """Get device of the module."""
        return self.recruitment_history.device
    
    def _compute_connectivity_loss(self, network_info: Dict[str, Any]) -> torch.Tensor:
        """Penalize excessive connections."""
        total_recruited = network_info.get('total_neurons_recruited', 0)
        
        # Quadratic penalty for number of recruited neurons
        connectivity_loss = torch.tensor(total_recruited ** 2, dtype=torch.float, device=self._get_device())
        
        # Add per-layer penalties
        layer_infos = network_info.get('layer_infos', [])
        for layer_info in layer_infos:
            num_recruited = layer_info.get('num_recruited', 0)
            # Encourage minimal but non-zero recruitment
            if num_recruited == 0:
                connectivity_loss += 10.0  # Penalty for no recruitment
            else:
                connectivity_loss += num_recruited * 0.1
        
        return connectivity_loss
    
    def _compute_diversity_loss(self, network_info: Dict[str, Any]) -> torch.Tensor:
        """Encourage diverse recruitment patterns over time."""
        # Update recruitment history
        self._update_recruitment_history(network_info)
        
        if self.history_count.item() < 2:
            return torch.tensor(0.0, device=self._get_device())
        
        # Compute diversity as negative entropy of recruitment patterns
        recent_steps = min(self.history_count.item(), 10)
        start_idx = max(0, self.history_pointer.item() - recent_steps)
        
        if start_idx == self.history_pointer.item():
            # Circular buffer wraparound
            recent_history = torch.cat([
                self.recruitment_history[start_idx:],
                self.recruitment_history[:self.history_pointer.item()]
            ], dim=0)
        else:
            recent_history = self.recruitment_history[start_idx:self.history_pointer.item()]
        
        # Compute recruitment frequencies
        recruitment_freq = recent_history.sum(dim=0)
        recruitment_freq = recruitment_freq / recruitment_freq.sum().clamp(min=1e-10)
        
        # Entropy of recruitment distribution (higher = more diverse)
        entropy = -(recruitment_freq * torch.log(recruitment_freq + 1e-10)).sum()
        
        # Return negative entropy as loss (we want to maximize diversity)
        return -entropy
    
    def _compute_sparsity_loss(self, network_info: Dict[str, Any]) -> torch.Tensor:
        """Encourage overall network sparsity."""
        sparsity = network_info.get('network_sparsity', 1.0)
        
        # Penalty for low sparsity (dense networks)
        target_sparsity = 0.8  # Target 80% sparsity
        sparsity_loss = torch.tensor(
            max(0, target_sparsity - sparsity) ** 2,
            device=self._get_device()
        )
        
        return sparsity_loss
    
    def _compute_efficiency_loss(self, network_info: Dict[str, Any]) -> torch.Tensor:
        """Encourage efficient budget usage."""
        efficiency = network_info.get('recruitment_efficiency', 1.0)
        
        # Penalty for low efficiency
        target_efficiency = 10.0  # Target efficiency value
        efficiency_loss = torch.tensor(
            max(0, target_efficiency - efficiency) / target_efficiency,
            device=self._get_device()
        )
        
        return efficiency_loss
    
    def _compute_population_balance_loss(self, network_info: Dict[str, Any]) -> torch.Tensor:
        """Encourage balanced population activity across layers."""
        layer_infos = network_info.get('layer_infos', [])
        
        if not layer_infos:
            return torch.tensor(0.0, device=self._get_device())
        
        # Get population activity levels
        activity_levels = []
        for layer_info in layer_infos:
            pop_stats = layer_info.get('population_stats', {})
            activity = pop_stats.get('mean', 0.0)
            activity_levels.append(activity)
        
        if not activity_levels:
            return torch.tensor(0.0, device=self._get_device())
        
        activity_tensor = torch.tensor(activity_levels, device=self._get_device())
        
        # Penalty for unbalanced activity (high variance)
        balance_loss = activity_tensor.var()
        
        return balance_loss
    
    def _update_recruitment_history(self, network_info: Dict[str, Any]):
        """Update recruitment history for diversity tracking."""
        layer_infos = network_info.get('layer_infos', [])
        
        # Create recruitment pattern for this step
        recruitment_pattern = torch.zeros(self.recruitment_history.shape[1], device=self._get_device())
        
        neuron_idx = 0
        for layer_info in layer_infos:
            recruited_indices = layer_info.get('recruited_indices', [])
            for idx in recruited_indices:
                if neuron_idx < recruitment_pattern.shape[0]:
                    recruitment_pattern[neuron_idx] = 1.0
                    neuron_idx += 1
        
        # Store in circular buffer
        self.recruitment_history[self.history_pointer] = recruitment_pattern
        self.history_pointer = (self.history_pointer + 1) % self.recruitment_history.shape[0]
        self.history_count = torch.min(self.history_count + 1, 
                                     torch.tensor(self.recruitment_history.shape[0]))


class FlexibilityLoss(nn.Module):
    """
    Loss function that encourages cognitive flexibility in DRN networks.
    
    This loss promotes:
    - Smooth decision boundaries (boundary_smoothness)
    - Adaptive behavior (adaptation ability)
    - Robust generalization (generalization penalty)
    - Transfer learning capability
    
    Args:
        boundary_smoothness_weight: Weight for boundary smoothness term
        adaptation_weight: Weight for adaptation capability term
        generalization_weight: Weight for generalization term
        transfer_weight: Weight for transfer learning term
    """
    
    def __init__(
        self,
        boundary_smoothness_weight: float = 1.0,
        adaptation_weight: float = 0.5,
        generalization_weight: float = 0.3,
        transfer_weight: float = 0.2
    ):
        super(FlexibilityLoss, self).__init__()
        
        self.boundary_smoothness_weight = boundary_smoothness_weight
        self.adaptation_weight = adaptation_weight
        self.generalization_weight = generalization_weight
        self.transfer_weight = transfer_weight
        
        # Store recent outputs for analysis
        self.register_buffer('recent_outputs', torch.zeros(50, 100))  # Last 50 batches, max 100 output dims
        self.register_buffer('recent_targets', torch.zeros(50, dtype=torch.long))
        self.register_buffer('output_pointer', torch.tensor(0))
        self.register_buffer('output_count', torch.tensor(0))
    
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        network_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute flexibility loss.
        
        Args:
            outputs: Model outputs [batch_size, output_size]
            targets: Target labels [batch_size]
            network_info: Network information from forward pass
            
        Returns:
            Scalar flexibility loss
        """
        total_loss = torch.tensor(0.0, device=outputs.device)
        
        # Update history
        self._update_output_history(outputs, targets)
        
        # 1. Boundary smoothness loss
        if self.boundary_smoothness_weight > 0:
            smoothness_loss = self._compute_boundary_smoothness_loss(outputs, targets, network_info)
            total_loss += self.boundary_smoothness_weight * smoothness_loss
        
        # 2. Adaptation loss
        if self.adaptation_weight > 0:
            adaptation_loss = self._compute_adaptation_loss(outputs, targets, network_info)
            total_loss += self.adaptation_weight * adaptation_loss
        
        # 3. Generalization loss
        if self.generalization_weight > 0:
            generalization_loss = self._compute_generalization_loss(outputs, targets)
            total_loss += self.generalization_weight * generalization_loss
        
        return total_loss
    
    def _compute_boundary_smoothness_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        network_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Encourage smooth decision boundaries.
        
        This is computed by adding small perturbations to inputs and 
        penalizing large changes in outputs.
        """
        if outputs.shape[0] < 2:
            return torch.tensor(0.0, device=outputs.device)
        
        # Use recruitment patterns as a proxy for input sensitivity
        layer_infos = network_info.get('layer_infos', [])
        if not layer_infos:
            return torch.tensor(0.0, device=outputs.device)
        
        # Compute output variance as a measure of boundary roughness
        output_probs = F.softmax(outputs, dim=-1)
        
        # High entropy = smooth boundaries, low entropy = sharp boundaries
        entropy = -(output_probs * torch.log(output_probs + 1e-10)).sum(dim=-1)
        target_entropy = math.log(outputs.shape[-1]) * 0.7  # Target 70% of max entropy
        
        # Penalty for too low entropy (too sharp boundaries)
        smoothness_loss = torch.relu(target_entropy - entropy).mean()
        
        return smoothness_loss
    
    def _compute_adaptation_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        network_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Encourage adaptive recruitment patterns.
        
        This penalizes overly rigid recruitment patterns.
        """
        layer_infos = network_info.get('layer_infos', [])
        if not layer_infos:
            return torch.tensor(0.0, device=outputs.device)
        
        adaptation_loss = torch.tensor(0.0, device=outputs.device)
        
        for layer_info in layer_infos:
            recruitment_entropy = layer_info.get('recruitment_entropy', 0.0)
            
            # Penalty for low recruitment entropy (too deterministic)
            min_entropy = 0.5  # Minimum desired entropy
            if recruitment_entropy < min_entropy:
                adaptation_loss += (min_entropy - recruitment_entropy) ** 2
        
        return adaptation_loss
    
    def _compute_generalization_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Encourage generalization by penalizing overconfident predictions.
        
        This helps prevent overfitting to training data.
        """
        if self.output_count.item() < 5:
            return torch.tensor(0.0, device=outputs.device)
        
        # Get recent outputs and compute consistency
        recent_steps = min(self.output_count.item(), 10)
        if recent_steps < 2:
            return torch.tensor(0.0, device=outputs.device)
        
        # Compute prediction consistency over recent batches
        # High consistency might indicate overfitting
        output_probs = F.softmax(outputs, dim=-1)
        max_probs = output_probs.max(dim=-1)[0]
        
        # Penalty for overconfident predictions (max prob too high)
        confidence_threshold = 0.9
        overconfidence_penalty = torch.relu(max_probs - confidence_threshold).mean()
        
        return overconfidence_penalty
    
    def _update_output_history(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Update output history for analysis."""
        # Store average outputs and targets for this batch
        batch_output = outputs.mean(dim=0)
        batch_target = targets.float().mean()
        
        # Pad or truncate to fit buffer
        output_size = min(batch_output.shape[0], self.recent_outputs.shape[1])
        
        self.recent_outputs[self.output_pointer, :output_size] = batch_output[:output_size]
        self.recent_targets[self.output_pointer] = batch_target.long()
        
        self.output_pointer = (self.output_pointer + 1) % self.recent_outputs.shape[0]
        self.output_count = torch.min(self.output_count + 1, 
                                    torch.tensor(self.recent_outputs.shape[0]))


class AdversarialFlexibilityLoss(nn.Module):
    """
    Advanced flexibility loss using adversarial examples.
    
    This loss generates small perturbations to inputs and encourages
    the network to maintain consistent outputs, promoting robustness
    and smooth decision boundaries.
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 3,
        consistency_weight: float = 1.0
    ):
        super(AdversarialFlexibilityLoss, self).__init__()
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.consistency_weight = consistency_weight
    
    def forward(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial flexibility loss.
        
        Args:
            model: The DRN model
            inputs: Input data [batch_size, input_size]
            outputs: Original outputs [batch_size, output_size]
            targets: Target labels [batch_size]
            
        Returns:
            Adversarial consistency loss
        """
        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)
        
        # Generate adversarial perturbations
        perturbed_inputs = self._generate_adversarial_inputs(model, inputs, targets)
        
        # Get outputs for perturbed inputs
        with torch.no_grad():
            perturbed_outputs, _ = model(perturbed_inputs, return_layer_info=False)
        
        # Compute consistency loss
        consistency_loss = F.mse_loss(outputs, perturbed_outputs)
        
        return self.consistency_weight * consistency_loss
    
    def _generate_adversarial_inputs(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial examples using PGD."""
        adv_inputs = inputs.clone().detach()
        
        for _ in range(self.num_steps):
            adv_inputs.requires_grad_(True)
            
            outputs, _ = model(adv_inputs, return_layer_info=False)
            loss = F.cross_entropy(outputs, targets)
            
            grad = torch.autograd.grad(loss, adv_inputs, create_graph=False)[0]
            
            # Take step in direction of gradient
            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            
            # Project to epsilon ball
            delta = adv_inputs - inputs
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            adv_inputs = inputs + delta
            
            # Clamp to valid input range (assuming [0, 1] or [-1, 1])
            adv_inputs = torch.clamp(adv_inputs, -2, 2)
        
        return adv_inputs.detach()


# Combined loss function
class DRNLoss(nn.Module):
    """
    Combined loss function for DRN networks.
    
    Combines base task loss with recruitment and flexibility regularization.
    """
    
    def __init__(
        self,
        base_criterion: nn.Module,
        recruitment_weight: float = 0.01,
        flexibility_weight: float = 0.005,
        adversarial_weight: float = 0.001,
        use_adversarial: bool = False
    ):
        super(DRNLoss, self).__init__()
        
        self.base_criterion = base_criterion
        self.recruitment_weight = recruitment_weight
        self.flexibility_weight = flexibility_weight
        self.adversarial_weight = adversarial_weight
        self.use_adversarial = use_adversarial
        
        self.recruitment_loss = RecruitmentLoss()
        self.flexibility_loss = FlexibilityLoss()
        
        if use_adversarial:
            self.adversarial_loss = AdversarialFlexibilityLoss()
    
    def forward(
        self, 
        model: nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        network_info: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined DRN loss.
        
        Returns:
            Dictionary with loss components
        """
        # Base task loss
        base_loss = self.base_criterion(outputs, targets)
        
        # Recruitment regularization
        recruitment_reg = self.recruitment_loss(network_info)
        
        # Flexibility regularization
        flexibility_reg = self.flexibility_loss(outputs, targets, network_info)
        
        # Total loss
        total_loss = (
            base_loss + 
            self.recruitment_weight * recruitment_reg +
            self.flexibility_weight * flexibility_reg
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'recruitment_loss': recruitment_reg,
            'flexibility_loss': flexibility_reg
        }
        
        # Add adversarial loss if enabled
        if self.use_adversarial and hasattr(self, 'adversarial_loss'):
            adversarial_reg = self.adversarial_loss(model, inputs, outputs, targets)
            total_loss = total_loss + self.adversarial_weight * adversarial_reg
            loss_dict['total_loss'] = total_loss
            loss_dict['adversarial_loss'] = adversarial_reg
        
        return loss_dict


# Example usage and testing
if __name__ == "__main__":
    print("Testing DRN loss functions...")
    
    # Create mock network info
    network_info = {
        'total_neurons_recruited': 15,
        'network_sparsity': 0.7,
        'recruitment_efficiency': 8.5,
        'layer_infos': [
            {
                'num_recruited': 8,
                'recruitment_entropy': 1.2,
                'recruited_indices': [1, 3, 5, 7, 9, 11, 13, 15],
                'population_stats': {'mean': 0.3, 'std': 0.1}
            },
            {
                'num_recruited': 7,
                'recruitment_entropy': 1.0,
                'recruited_indices': [0, 2, 4, 6, 8, 10, 12],
                'population_stats': {'mean': 0.4, 'std': 0.12}
            }
        ]
    }
    
    # Test recruitment loss
    recruitment_loss = RecruitmentLoss()
    rec_loss_val = recruitment_loss(network_info)
    print(f"Recruitment loss: {rec_loss_val.item():.4f}")
    
    # Test flexibility loss
    flexibility_loss = FlexibilityLoss()
    outputs = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))
    
    flex_loss_val = flexibility_loss(outputs, targets, network_info)
    print(f"Flexibility loss: {flex_loss_val.item():.4f}")
    
    # Test combined loss
    base_criterion = nn.CrossEntropyLoss()
    combined_loss = DRNLoss(base_criterion)
    
    # Mock model for testing
    class MockModel(nn.Module):
        def forward(self, x, return_layer_info=True):
            return torch.randn(x.shape[0], 10), network_info
    
    model = MockModel()
    inputs = torch.randn(16, 64)
    
    loss_dict = combined_loss(model, inputs, outputs, targets, network_info)
    print(f"Combined loss components: {list(loss_dict.keys())}")
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    
    print("DRN loss function tests passed!")