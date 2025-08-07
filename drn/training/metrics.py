"""
drn/training/metrics.py

Specialized metrics for evaluating cognitive flexibility and connectivity patterns
in Dynamic Recruitment Networks. These metrics help quantify the key innovations
of DRN compared to traditional neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import math
from collections import defaultdict
import warnings


class FlexibilityMetrics:
    """
    Metrics for evaluating cognitive flexibility in neural networks.
    
    These metrics assess how well the network can:
    - Adapt to new data patterns
    - Generalize across different contexts
    - Maintain smooth decision boundaries
    - Transfer knowledge to new tasks
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_flexibility_score(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        network_infos: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute comprehensive flexibility score.
        
        Args:
            model: The neural network model
            outputs: Model outputs [batch_size, output_size]
            targets: Target labels [batch_size]
            network_infos: List of network info dictionaries
            
        Returns:
            Dictionary with flexibility metrics
        """
        flexibility_scores = {}
        
        # 1. Boundary smoothness
        boundary_smoothness = self.compute_boundary_smoothness(model, outputs[:32])  # Sample subset
        flexibility_scores['boundary_smoothness'] = boundary_smoothness
        
        # 2. Prediction entropy (measure of uncertainty/flexibility)
        prediction_entropy = self.compute_prediction_entropy(outputs)
        flexibility_scores['prediction_entropy'] = prediction_entropy
        
        # 3. Recruitment diversity (DRN-specific)
        if network_infos:
            recruitment_diversity = self.compute_recruitment_diversity(network_infos)
            flexibility_scores['recruitment_diversity'] = recruitment_diversity
        
        # 4. Adaptation capability
        adaptation_score = self.compute_adaptation_capability(network_infos)
        flexibility_scores['adaptation_capability'] = adaptation_score
        
        # 5. Overall flexibility score (weighted combination)
        overall_score = (
            0.3 * boundary_smoothness +
            0.2 * prediction_entropy +
            0.3 * flexibility_scores.get('recruitment_diversity', 0.5) +
            0.2 * adaptation_score
        )
        flexibility_scores['overall_flexibility'] = overall_score
        
        return flexibility_scores
    
    def compute_boundary_smoothness(
        self, 
        model: nn.Module, 
        sample_inputs: torch.Tensor,
        num_samples: int = 50,
        perturbation_scale: float = 0.1
    ) -> float:
        """
        Compute smoothness of decision boundaries.
        
        This measures how much the model's outputs change when inputs
        are slightly perturbed, indicating boundary smoothness.
        
        Args:
            model: Neural network model
            sample_inputs: Sample input data [batch_size, input_size]
            num_samples: Number of perturbation samples
            perturbation_scale: Scale of input perturbations
            
        Returns:
            Boundary smoothness score (higher = smoother boundaries)
        """
        if sample_inputs.shape[0] == 0:
            return 0.5  # Default neutral score
        
        model.eval()
        sample_inputs = sample_inputs.to(self.device)
        
        with torch.no_grad():
            # Get original outputs
            if hasattr(model, 'forward') and 'return_layer_info' in model.forward.__code__.co_varnames:
                original_outputs, _ = model(sample_inputs, return_layer_info=False)
            else:
                original_outputs = model(sample_inputs)
            
            # Generate perturbations and compute output changes
            smoothness_scores = []
            
            for _ in range(num_samples):
                # Add small random perturbations
                noise = torch.randn_like(sample_inputs) * perturbation_scale
                perturbed_inputs = sample_inputs + noise
                
                # Get perturbed outputs
                if hasattr(model, 'forward') and 'return_layer_info' in model.forward.__code__.co_varnames:
                    perturbed_outputs, _ = model(perturbed_inputs, return_layer_info=False)
                else:
                    perturbed_outputs = model(perturbed_inputs)
                
                # Compute output difference
                output_diff = torch.norm(original_outputs - perturbed_outputs, dim=-1)
                input_diff = torch.norm(noise, dim=-1)
                
                # Smoothness = inverse of output change / input change ratio
                smoothness = 1.0 / (1.0 + (output_diff / (input_diff + 1e-8)).mean().item())
                smoothness_scores.append(smoothness)
        
        return float(np.mean(smoothness_scores))
    
    def compute_prediction_entropy(self, outputs: torch.Tensor) -> float:
        """
        Compute entropy of model predictions.
        
        Higher entropy indicates more uncertainty and potentially more flexibility.
        
        Args:
            outputs: Model outputs [batch_size, output_size]
            
        Returns:
            Average prediction entropy
        """
        # Convert to probabilities
        probs = F.softmax(outputs, dim=-1)
        
        # Compute entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log(outputs.shape[-1])
        normalized_entropy = entropy.mean().item() / max_entropy
        
        return normalized_entropy
    
    def compute_recruitment_diversity(self, network_infos: List[Dict[str, Any]]) -> float:
        """
        Compute diversity of neuron recruitment patterns (DRN-specific).
        
        This measures how varied the recruitment patterns are across
        different forward passes.
        
        Args:
            network_infos: List of network information dictionaries
            
        Returns:
            Recruitment diversity score (higher = more diverse)
        """
        if not network_infos:
            return 0.5  # Default score
        
        # Collect all recruited indices across all layers and time steps
        all_recruited = []
        max_neuron_idx = 0
        
        for info in network_infos:
            layer_infos = info.get('layer_infos', [])
            for layer_info in layer_infos:
                recruited_indices = layer_info.get('recruited_indices', [])
                all_recruited.extend(recruited_indices)
                if recruited_indices:
                    max_neuron_idx = max(max_neuron_idx, max(recruited_indices))
        
        if not all_recruited or max_neuron_idx == 0:
            return 0.5
        
        # Compute recruitment frequency distribution
        recruitment_counts = np.zeros(max_neuron_idx + 1)
        for idx in all_recruited:
            if idx < len(recruitment_counts):
                recruitment_counts[idx] += 1
        
        # Normalize to probabilities
        recruitment_probs = recruitment_counts / (recruitment_counts.sum() + 1e-10)
        
        # Compute entropy of recruitment distribution
        entropy = -np.sum(recruitment_probs * np.log(recruitment_probs + 1e-10))
        max_entropy = math.log(len(recruitment_probs))
        
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return normalized_entropy
    
    def compute_adaptation_capability(self, network_infos: List[Dict[str, Any]]) -> float:
        """
        Compute network's adaptation capability based on recruitment pattern changes.
        
        Args:
            network_infos: List of network information dictionaries
            
        Returns:
            Adaptation capability score (higher = more adaptive)
        """
        if len(network_infos) < 2:
            return 0.5  # Default score
        
        # Analyze how recruitment patterns change over time
        recruitment_changes = []
        
        for i in range(1, len(network_infos)):
            prev_info = network_infos[i-1]
            curr_info = network_infos[i]
            
            # Get recruitment patterns for both time steps
            prev_recruited = set()
            curr_recruited = set()
            
            for layer_info in prev_info.get('layer_infos', []):
                prev_recruited.update(layer_info.get('recruited_indices', []))
            
            for layer_info in curr_info.get('layer_infos', []):
                curr_recruited.update(layer_info.get('recruited_indices', []))
            
            # Compute Jaccard similarity (overlap)
            if prev_recruited or curr_recruited:
                intersection = len(prev_recruited & curr_recruited)
                union = len(prev_recruited | curr_recruited)
                similarity = intersection / union if union > 0 else 1.0
                change = 1.0 - similarity  # Higher change = more adaptation
                recruitment_changes.append(change)
        
        if not recruitment_changes:
            return 0.5
        
        # Adaptation capability = average amount of change
        adaptation_score = np.mean(recruitment_changes)
        
        return adaptation_score
    
    def compute_adaptation_speed(
        self, 
        model: nn.Module, 
        adaptation_loader: torch.utils.data.DataLoader,
        num_steps: int = 10
    ) -> float:
        """
        Compute how quickly the model adapts to new data.
        
        Args:
            model: Neural network model
            adaptation_loader: DataLoader with adaptation task data
            num_steps: Number of adaptation steps to evaluate
            
        Returns:
            Adaptation speed score (higher = faster adaptation)
        """
        model.eval()
        
        initial_performance = []
        final_performance = []
        
        # Get initial performance
        with torch.no_grad():
            for i, (data, targets) in enumerate(adaptation_loader):
                if i >= 3:  # Just use first few batches
                    break
                
                data, targets = data.to(self.device), targets.to(self.device)
                
                if hasattr(model, 'forward') and 'return_layer_info' in model.forward.__code__.co_varnames:
                    outputs, _ = model(data, return_layer_info=False)
                else:
                    outputs = model(data)
                
                # Compute accuracy or loss
                if len(targets.shape) == 1:  # Classification
                    acc = (outputs.argmax(dim=-1) == targets).float().mean().item()
                    initial_performance.append(acc)
                else:
                    loss = F.mse_loss(outputs, targets).item()
                    initial_performance.append(1.0 / (1.0 + loss))  # Convert to "performance"
        
        # Simulate adaptation (would normally involve actual training steps)
        # For this metric, we'll use recruitment pattern changes as a proxy
        network_infos = []
        with torch.no_grad():
            for i, (data, targets) in enumerate(adaptation_loader):
                if i >= num_steps:
                    break
                
                data = data.to(self.device)
                
                if hasattr(model, 'forward') and 'return_layer_info' in model.forward.__code__.co_varnames:
                    outputs, info = model(data, return_layer_info=True)
                    network_infos.append(info)
                    
                    if i >= num_steps - 3:  # Last few steps
                        if len(targets.shape) == 1:
                            acc = (outputs.argmax(dim=-1) == targets.to(self.device)).float().mean().item()
                            final_performance.append(acc)
                        else:
                            loss = F.mse_loss(outputs, targets.to(self.device)).item()
                            final_performance.append(1.0 / (1.0 + loss))
        
        # Compute adaptation speed as improvement rate
        if initial_performance and final_performance:
            initial_avg = np.mean(initial_performance)
            final_avg = np.mean(final_performance)
            improvement = final_avg - initial_avg
            adaptation_speed = max(0.0, improvement)  # Only positive improvements
        else:
            adaptation_speed = 0.0
        
        return adaptation_speed
    
    def compute_transfer_capability(
        self,
        model: nn.Module,
        source_loader: torch.utils.data.DataLoader,
        target_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Compute transfer learning capability.
        
        Args:
            model: Neural network model
            source_loader: Source task data
            target_loader: Target task data
            
        Returns:
            Transfer capability score
        """
        # This is a simplified version - full implementation would require
        # actual transfer learning evaluation
        
        model.eval()
        
        # Get representations on source task
        source_representations = []
        with torch.no_grad():
            for i, (data, _) in enumerate(source_loader):
                if i >= 5:  # Limit samples
                    break
                data = data.to(self.device)
                
                if hasattr(model, 'forward') and 'return_layer_info' in model.forward.__code__.co_varnames:
                    outputs, _ = model(data, return_layer_info=False)
                else:
                    outputs = model(data)
                
                source_representations.append(outputs)
        
        # Get representations on target task
        target_representations = []
        with torch.no_grad():
            for i, (data, _) in enumerate(target_loader):
                if i >= 5:  # Limit samples
                    break
                data = data.to(self.device)
                
                if hasattr(model, 'forward') and 'return_layer_info' in model.forward.__code__.co_varnames:
                    outputs, _ = model(data, return_layer_info=False)
                else:
                    outputs = model(data)
                
                target_representations.append(outputs)
        
        if not source_representations or not target_representations:
            return 0.5
        
        # Compute similarity between source and target representations
        source_mean = torch.cat(source_representations, dim=0).mean(dim=0)
        target_mean = torch.cat(target_representations, dim=0).mean(dim=0)
        
        similarity = F.cosine_similarity(source_mean, target_mean, dim=0).item()
        transfer_score = (similarity + 1.0) / 2.0  # Normalize to [0, 1]
        
        return transfer_score


class ConnectivityMetrics:
    """
    Metrics for analyzing connectivity patterns in neural networks,
    particularly DRN-specific patterns.
    """
    
    def __init__(self):
        pass
    
    def analyze_network_connectivity(self, network_infos: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Comprehensive analysis of network connectivity patterns.
        
        Args:
            network_infos: List of network information dictionaries
            
        Returns:
            Dictionary with connectivity metrics
        """
        if not network_infos:
            return {'connectivity_error': 1.0}
        
        metrics = {}
        
        # 1. Sparsity metrics
        sparsity_metrics = self.compute_sparsity_metrics(network_infos)
        metrics.update(sparsity_metrics)
        
        # 2. Recruitment pattern analysis
        recruitment_metrics = self.compute_recruitment_metrics(network_infos)
        metrics.update(recruitment_metrics)
        
        # 3. Budget usage analysis
        budget_metrics = self.compute_budget_metrics(network_infos)
        metrics.update(budget_metrics)
        
        # 4. Temporal dynamics
        temporal_metrics = self.compute_temporal_dynamics(network_infos)
        metrics.update(temporal_metrics)
        
        return metrics
    
    def compute_sparsity_metrics(self, network_infos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute various sparsity-related metrics."""
        sparsities = [info.get('network_sparsity', 0.0) for info in network_infos]
        
        if not sparsities:
            return {'sparsity_mean': 0.0, 'sparsity_std': 0.0}
        
        return {
            'sparsity_mean': float(np.mean(sparsities)),
            'sparsity_std': float(np.std(sparsities)),
            'sparsity_min': float(np.min(sparsities)),
            'sparsity_max': float(np.max(sparsities))
        }
    
    def compute_recruitment_metrics(self, network_infos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute recruitment pattern metrics."""
        total_recruited = [info.get('total_neurons_recruited', 0) for info in network_infos]
        
        if not total_recruited:
            return {'recruitment_mean': 0.0}
        
        # Per-layer recruitment analysis
        layer_recruitment = defaultdict(list)
        for info in network_infos:
            layer_infos = info.get('layer_infos', [])
            for i, layer_info in enumerate(layer_infos):
                layer_recruitment[i].append(layer_info.get('num_recruited', 0))
        
        metrics = {
            'recruitment_mean': float(np.mean(total_recruited)),
            'recruitment_std': float(np.std(total_recruited)),
            'recruitment_cv': float(np.std(total_recruited) / (np.mean(total_recruited) + 1e-8))
        }
        
        # Layer-specific metrics
        for layer_idx, recruitments in layer_recruitment.items():
            if recruitments:
                metrics[f'layer_{layer_idx}_recruitment_mean'] = float(np.mean(recruitments))
                metrics[f'layer_{layer_idx}_recruitment_std'] = float(np.std(recruitments))
        
        return metrics
    
    def compute_budget_metrics(self, network_infos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute budget usage metrics."""
        budget_spent = [info.get('total_budget_spent', 0.0) for info in network_infos]
        budget_remaining = [info.get('avg_budget_remaining', 1.0) for info in network_infos]
        efficiency = [info.get('recruitment_efficiency', 1.0) for info in network_infos]
        
        metrics = {}
        
        if budget_spent:
            metrics.update({
                'budget_spent_mean': float(np.mean(budget_spent)),
                'budget_spent_std': float(np.std(budget_spent)),
            })
        
        if budget_remaining:
            metrics.update({
                'budget_remaining_mean': float(np.mean(budget_remaining)),
                'budget_remaining_std': float(np.std(budget_remaining)),
            })
        
        if efficiency:
            metrics.update({
                'efficiency_mean': float(np.mean(efficiency)),
                'efficiency_std': float(np.std(efficiency)),
            })
        
        return metrics
    
    def compute_temporal_dynamics(self, network_infos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute temporal dynamics of connectivity patterns."""
        if len(network_infos) < 2:
            return {'temporal_stability': 1.0}
        
        # Analyze how recruitment patterns change over time
        pattern_changes = []
        sparsity_changes = []
        
        for i in range(1, len(network_infos)):
            prev_info = network_infos[i-1]
            curr_info = network_infos[i]
            
            # Pattern change
            prev_recruited = prev_info.get('total_neurons_recruited', 0)
            curr_recruited = curr_info.get('total_neurons_recruited', 0)
            pattern_change = abs(curr_recruited - prev_recruited)
            pattern_changes.append(pattern_change)
            
            # Sparsity change
            prev_sparsity = prev_info.get('network_sparsity', 0.0)
            curr_sparsity = curr_info.get('network_sparsity', 0.0)
            sparsity_change = abs(curr_sparsity - prev_sparsity)
            sparsity_changes.append(sparsity_change)
        
        metrics = {
            'pattern_change_mean': float(np.mean(pattern_changes)) if pattern_changes else 0.0,
            'pattern_stability': 1.0 - (float(np.mean(pattern_changes)) / 10.0) if pattern_changes else 1.0,
            'sparsity_change_mean': float(np.mean(sparsity_changes)) if sparsity_changes else 0.0,
        }
        
        return metrics
    
    def compute_connectivity_diversity(self, network_infos: List[Dict[str, Any]]) -> float:
        """
        Compute overall connectivity diversity across the network.
        
        Args:
            network_infos: List of network information dictionaries
            
        Returns:
            Connectivity diversity score (higher = more diverse)
        """
        if not network_infos:
            return 0.5
        
        # Collect all recruitment patterns
        all_patterns = []
        
        for info in network_infos:
            layer_infos = info.get('layer_infos', [])
            pattern = []
            for layer_info in layer_infos:
                recruited = layer_info.get('recruited_indices', [])
                pattern.extend(recruited)
            all_patterns.append(set(pattern))
        
        if len(all_patterns) < 2:
            return 0.5
        
        # Compute average pairwise Jaccard distance
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(all_patterns)):
            for j in range(i + 1, len(all_patterns)):
                set_i, set_j = all_patterns[i], all_patterns[j]
                
                if not set_i and not set_j:
                    distance = 0.0
                elif not set_i or not set_j:
                    distance = 1.0
                else:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    jaccard_similarity = intersection / union
                    distance = 1.0 - jaccard_similarity
                
                total_distance += distance
                num_pairs += 1
        
        if num_pairs == 0:
            return 0.5
        
        average_distance = total_distance / num_pairs
        return average_distance


class ComparisonMetrics:
    """
    Metrics for comparing DRN with traditional neural networks.
    """
    
    def __init__(self):
        self.flexibility_metrics = FlexibilityMetrics()
        self.connectivity_metrics = ConnectivityMetrics()
    
    def compare_models(
        self,
        drn_model: nn.Module,
        traditional_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare DRN model with traditional model on various metrics.
        
        Args:
            drn_model: DRN model
            traditional_model: Traditional neural network
            test_loader: Test data loader
            device: Device for computation
            
        Returns:
            Comparison results dictionary
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Evaluate both models
        drn_results = self._evaluate_model(drn_model, test_loader, device, is_drn=True)
        traditional_results = self._evaluate_model(traditional_model, test_loader, device, is_drn=False)
        
        # Compute differences
        improvements = {}
        for key in drn_results:
            if key in traditional_results:
                improvement = drn_results[key] - traditional_results[key]
                improvements[f'{key}_improvement'] = improvement
        
        return {
            'drn_results': drn_results,
            'traditional_results': traditional_results,
            'improvements': improvements
        }
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        is_drn: bool
    ) -> Dict[str, float]:
        """Evaluate a single model."""
        model.eval()
        
        all_outputs = []
        all_targets = []
        all_network_infos = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                
                if is_drn and hasattr(model, 'forward'):
                    try:
                        outputs, network_info = model(data, return_layer_info=True)
                        all_network_infos.append(network_info)
                    except:
                        outputs = model(data)
                        all_network_infos.append({})
                else:
                    outputs = model(data)
                    all_network_infos.append({})
                
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Combine results
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        results = {}
        
        # Basic performance
        if len(all_targets.shape) == 1:  # Classification
            accuracy = (all_outputs.argmax(dim=-1) == all_targets).float().mean().item()
            results['accuracy'] = accuracy
        
        # Flexibility metrics
        if all_network_infos and any(info for info in all_network_infos):
            flexibility_scores = self.flexibility_metrics.compute_flexibility_score(
                model, all_outputs, all_targets, all_network_infos
            )
            results.update(flexibility_scores)
        
        # Connectivity metrics (for DRN)
        if is_drn and all_network_infos:
            connectivity_scores = self.connectivity_metrics.analyze_network_connectivity(all_network_infos)
            results.update(connectivity_scores)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing DRN metrics...")
    
    # Create mock data
    outputs = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    
    network_infos = [
        {
            'total_neurons_recruited': 15,
            'network_sparsity': 0.7,
            'recruitment_efficiency': 8.5,
            'total_budget_spent': 1.5,
            'avg_budget_remaining': 0.3,
            'layer_infos': [
                {
                    'num_recruited': 8,
                    'recruitment_entropy': 1.2,
                    'recruited_indices': [1, 3, 5, 7, 9, 11, 13, 15]
                },
                {
                    'num_recruited': 7,
                    'recruitment_entropy': 1.0,
                    'recruited_indices': [0, 2, 4, 6, 8, 10, 12]
                }
            ]
        }
    ] * 10  # Simulate 10 forward passes
    
    # Test flexibility metrics
    flexibility_metrics = FlexibilityMetrics()
    
    # Create a simple mock model for testing
    class MockModel(nn.Module):
        def forward(self, x, return_layer_info=False):
            output = torch.randn(x.shape[0], 10)
            if return_layer_info:
                return output, network_infos[0]
            return output
    
    model = MockModel()
    
    flexibility_scores = flexibility_metrics.compute_flexibility_score(
        model, outputs, targets, network_infos
    )
    print(f"Flexibility scores: {list(flexibility_scores.keys())}")
    print(f"Overall flexibility: {flexibility_scores['overall_flexibility']:.3f}")
    
    # Test connectivity metrics
    connectivity_metrics = ConnectivityMetrics()
    connectivity_scores = connectivity_metrics.analyze_network_connectivity(network_infos)
    print(f"Connectivity metrics: {list(connectivity_scores.keys())}")
    
    # Test boundary smoothness
    sample_inputs = torch.randn(8, 64)
    boundary_smoothness = flexibility_metrics.compute_boundary_smoothness(model, sample_inputs)
    print(f"Boundary smoothness: {boundary_smoothness:.3f}")
    
    # Test prediction entropy
    pred_entropy = flexibility_metrics.compute_prediction_entropy(outputs)
    print(f"Prediction entropy: {pred_entropy:.3f}")
    
    # Test recruitment diversity
    recruitment_diversity = flexibility_metrics.compute_recruitment_diversity(network_infos)
    print(f"Recruitment diversity: {recruitment_diversity:.3f}")
    
    print("DRN metrics tests passed!")