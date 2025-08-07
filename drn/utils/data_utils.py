"""
drn/utils/data_utils.py

Data processing utilities for Dynamic Recruitment Networks research.
This module provides specialized data loaders, preprocessing functions, and
synthetic data generation for concept learning, cognitive flexibility, and
autism modeling experiments.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Iterator
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
import warnings
from collections import defaultdict, deque
import random
import math


class ConceptLearningDataset(Dataset):
    """
    Dataset for concept learning experiments with DRN.
    
    This dataset creates structured learning tasks where concepts
    can be learned hierarchically and incrementally.
    
    Args:
        num_concepts (int): Number of distinct concepts
        samples_per_concept (int): Samples per concept
        feature_dim (int): Dimensionality of input features
        concept_complexity (str): 'simple', 'medium', 'complex'
        hierarchical (bool): Whether concepts have hierarchical structure
        noise_level (float): Amount of noise to add to features
        concept_drift (bool): Whether concepts change over time
    """
    
    def __init__(
        self,
        num_concepts: int = 4,
        samples_per_concept: int = 200,
        feature_dim: int = 64,
        concept_complexity: str = 'medium',
        hierarchical: bool = True,
        noise_level: float = 0.1,
        concept_drift: bool = False
    ):
        self.num_concepts = num_concepts
        self.samples_per_concept = samples_per_concept
        self.feature_dim = feature_dim
        self.concept_complexity = concept_complexity
        self.hierarchical = hierarchical
        self.noise_level = noise_level
        self.concept_drift = concept_drift
        
        # Generate concept data
        self.data, self.labels, self.concept_info = self._generate_concept_data()
        
        # Create metadata
        self.total_samples = len(self.data)
        self.concept_centers = self._compute_concept_centers()
        self.concept_boundaries = self._compute_concept_boundaries()
    
    def _generate_concept_data(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Generate structured concept learning data."""
        all_data = []
        all_labels = []
        concept_info = {
            'concept_descriptions': {},
            'hierarchical_structure': {},
            'feature_importance': {},
            'concept_relationships': {}
        }
        
        # Define complexity parameters
        complexity_params = {
            'simple': {'num_clusters': 1, 'feature_groups': 2, 'nonlinearity': 0.1},
            'medium': {'num_clusters': 2, 'feature_groups': 4, 'nonlinearity': 0.3},
            'complex': {'num_clusters': 3, 'feature_groups': 8, 'nonlinearity': 0.5}
        }
        params = complexity_params[self.concept_complexity]
        
        # Generate base concept centers
        concept_centers = self._generate_concept_centers(params)
        
        for concept_id in range(self.num_concepts):
            concept_data, concept_meta = self._generate_single_concept(
                concept_id, concept_centers[concept_id], params
            )
            
            concept_labels = torch.full((len(concept_data),), concept_id, dtype=torch.long)
            
            all_data.append(concept_data)
            all_labels.append(concept_labels)
            concept_info['concept_descriptions'][concept_id] = concept_meta
        
        # Combine all data
        data = torch.cat(all_data, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Add hierarchical structure if enabled
        if self.hierarchical:
            concept_info['hierarchical_structure'] = self._create_hierarchical_structure()
        
        # Add noise
        if self.noise_level > 0:
            noise = torch.randn_like(data) * self.noise_level
            data = data + noise
        
        return data, labels, concept_info
    
    def _generate_concept_centers(self, params: Dict[str, Any]) -> List[torch.Tensor]:
        """Generate centers for each concept in feature space."""
        centers = []
        
        # Ensure concepts are well-separated
        for i in range(self.num_concepts):
            center = torch.randn(self.feature_dim) * 2.0
            
            # Add structure based on concept relationships
            if self.hierarchical and i > 0:
                # Make some concepts related to previous ones
                parent_concept = i // 2
                if parent_concept < len(centers):
                    # Inherit some features from parent concept
                    inheritance_weight = 0.3
                    center = (inheritance_weight * centers[parent_concept] + 
                             (1 - inheritance_weight) * center)
            
            centers.append(center)
        
        return centers
    
    def _generate_single_concept(
        self, 
        concept_id: int, 
        center: torch.Tensor, 
        params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate data for a single concept."""
        
        # Create clusters around concept center
        concept_data = []
        for cluster in range(params['num_clusters']):
            cluster_center = center + torch.randn(self.feature_dim) * 0.5
            cluster_size = self.samples_per_concept // params['num_clusters']
            
            # Generate samples around cluster center
            cluster_samples = torch.randn(cluster_size, self.feature_dim) * 0.3
            cluster_samples = cluster_samples + cluster_center.unsqueeze(0)
            
            concept_data.append(cluster_samples)
        
        concept_data = torch.cat(concept_data, dim=0)
        
        # Add concept-specific transformations
        concept_data = self._apply_concept_transformations(
            concept_data, concept_id, params
        )
        
        # Create metadata
        concept_meta = {
            'center': center,
            'cluster_count': params['num_clusters'],
            'feature_groups': params['feature_groups'],
            'complexity_level': self.concept_complexity,
            'transformation_type': f'concept_{concept_id}_transform'
        }
        
        return concept_data, concept_meta
    
    def _apply_concept_transformations(
        self, 
        data: torch.Tensor, 
        concept_id: int, 
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply concept-specific transformations to make concepts distinctive."""
        
        # Feature group transformations
        group_size = self.feature_dim // params['feature_groups']
        
        for group in range(params['feature_groups']):
            start_idx = group * group_size
            end_idx = min((group + 1) * group_size, self.feature_dim)
            
            # Apply different transformations based on concept and group
            transform_type = (concept_id + group) % 4
            
            if transform_type == 0:
                # Amplification
                data[:, start_idx:end_idx] *= (1 + concept_id * 0.2)
            elif transform_type == 1:
                # Rotation (simple linear combination)
                if end_idx - start_idx >= 2:
                    angle = concept_id * np.pi / 4
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    x = data[:, start_idx].clone()
                    y = data[:, start_idx + 1].clone()
                    data[:, start_idx] = cos_a * x - sin_a * y
                    data[:, start_idx + 1] = sin_a * x + cos_a * y
            elif transform_type == 2:
                # Nonlinear transformation
                data[:, start_idx:end_idx] = torch.tanh(
                    data[:, start_idx:end_idx] * (1 + concept_id * 0.1)
                )
            else:
                # Selective suppression
                mask = torch.rand(data.shape[0], end_idx - start_idx) > 0.3
                data[:, start_idx:end_idx] *= mask.float()
        
        return data
    
    def _create_hierarchical_structure(self) -> Dict[str, Any]:
        """Create hierarchical relationships between concepts."""
        structure = {
            'levels': {},
            'parent_child_relationships': {},
            'sibling_relationships': {}
        }
        
        # Create simple binary tree structure
        for concept_id in range(self.num_concepts):
            level = int(np.log2(concept_id + 1))
            if level not in structure['levels']:
                structure['levels'][level] = []
            structure['levels'][level].append(concept_id)
            
            # Parent-child relationships
            if concept_id > 0:
                parent = (concept_id - 1) // 2
                structure['parent_child_relationships'][concept_id] = parent
            
            # Sibling relationships
            if concept_id > 0:
                sibling = concept_id + 1 if concept_id % 2 == 1 else concept_id - 1
                if sibling < self.num_concepts:
                    structure['sibling_relationships'][concept_id] = sibling
        
        return structure
    
    def _compute_concept_centers(self) -> Dict[int, torch.Tensor]:
        """Compute actual centers of generated concepts."""
        centers = {}
        for concept_id in range(self.num_concepts):
            concept_mask = self.labels == concept_id
            concept_data = self.data[concept_mask]
            centers[concept_id] = concept_data.mean(dim=0)
        return centers
    
    def _compute_concept_boundaries(self) -> Dict[str, Any]:
        """Compute decision boundaries between concepts."""
        boundaries = {
            'separability_matrix': torch.zeros(self.num_concepts, self.num_concepts),
            'margin_distances': {},
            'overlap_regions': {}
        }
        
        # Compute pairwise separability
        for i in range(self.num_concepts):
            for j in range(i + 1, self.num_concepts):
                data_i = self.data[self.labels == i]
                data_j = self.data[self.labels == j]
                
                # Simple separability measure: distance between centers / average spread
                center_distance = torch.norm(data_i.mean(0) - data_j.mean(0))
                avg_spread = (data_i.std() + data_j.std()) / 2
                
                separability = center_distance / (avg_spread + 1e-8)
                boundaries['separability_matrix'][i, j] = separability
                boundaries['separability_matrix'][j, i] = separability
        
        return boundaries
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
    
    def get_concept_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the generated concepts."""
        return {
            'concept_info': self.concept_info,
            'concept_centers': self.concept_centers,
            'concept_boundaries': self.concept_boundaries,
            'dataset_stats': {
                'total_samples': self.total_samples,
                'num_concepts': self.num_concepts,
                'feature_dim': self.feature_dim,
                'samples_per_concept': self.samples_per_concept
            }
        }


class TaskSwitchingDataset(Dataset):
    """
    Dataset for task switching and cognitive flexibility experiments.
    
    Creates sequences of different tasks that require cognitive flexibility
    to switch between different rules or processing modes.
    
    Args:
        task_types (List[str]): Types of tasks to include
        sequence_length (int): Length of task sequences
        switch_probability (float): Probability of task switch at each step
        num_sequences (int): Number of task sequences to generate
        task_difficulty (str): 'easy', 'medium', 'hard'
    """
    
    def __init__(
        self,
        task_types: Optional[List[str]] = None,
        sequence_length: int = 100,
        switch_probability: float = 0.3,
        num_sequences: int = 50,
        task_difficulty: str = 'medium'
    ):
        if task_types is None:
            task_types = ['classification', 'regression', 'sequence', 'spatial']
        
        self.task_types = task_types
        self.sequence_length = sequence_length
        self.switch_probability = switch_probability
        self.num_sequences = num_sequences
        self.task_difficulty = task_difficulty
        
        # Generate task switching sequences
        self.sequences = self._generate_task_sequences()
        self.task_data = self._generate_task_data()
        
        # Create flat dataset for easy iteration
        self.flat_data = self._flatten_sequences()
    
    def _generate_task_sequences(self) -> List[List[Dict[str, Any]]]:
        """Generate sequences of task switches."""
        sequences = []
        
        for seq_idx in range(self.num_sequences):
            sequence = []
            current_task = random.choice(self.task_types)
            
            for step in range(self.sequence_length):
                # Decide whether to switch tasks
                if step > 0 and random.random() < self.switch_probability:
                    # Switch to different task
                    available_tasks = [t for t in self.task_types if t != current_task]
                    current_task = random.choice(available_tasks)
                
                # Create task specification
                task_spec = {
                    'task_type': current_task,
                    'task_id': f'{current_task}_{step}',
                    'is_switch': step > 0 and sequence[-1]['task_type'] != current_task if sequence else False,
                    'sequence_position': step,
                    'sequence_id': seq_idx
                }
                
                sequence.append(task_spec)
            
            sequences.append(sequence)
        
        return sequences
    
    def _generate_task_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate data for each task type."""
        task_data = {}
        
        # Define task parameters based on difficulty
        difficulty_params = {
            'easy': {'input_dim': 32, 'output_dim': 2, 'noise': 0.05},
            'medium': {'input_dim': 64, 'output_dim': 4, 'noise': 0.1},
            'hard': {'input_dim': 128, 'output_dim': 8, 'noise': 0.15}
        }
        params = difficulty_params[self.task_difficulty]
        
        for task_type in self.task_types:
            if task_type == 'classification':
                task_data[task_type] = self._generate_classification_task(params)
            elif task_type == 'regression':
                task_data[task_type] = self._generate_regression_task(params)
            elif task_type == 'sequence':
                task_data[task_type] = self._generate_sequence_task(params)
            elif task_type == 'spatial':
                task_data[task_type] = self._generate_spatial_task(params)
            else:
                # Default to classification
                task_data[task_type] = self._generate_classification_task(params)
        
        return task_data
    
    def _generate_classification_task(self, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate classification task data."""
        n_samples = 1000
        X, y = make_classification(
            n_samples=n_samples,
            n_features=params['input_dim'],
            n_classes=params['output_dim'],
            n_informative=params['input_dim'] // 2,
            n_redundant=params['input_dim'] // 4,
            random_state=42
        )
        
        return {
            'inputs': torch.FloatTensor(X),
            'targets': torch.LongTensor(y),
            'task_type': 'classification'
        }
    
    def _generate_regression_task(self, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate regression task data."""
        n_samples = 1000
        X = torch.randn(n_samples, params['input_dim'])
        
        # Create nonlinear regression target
        y = torch.sum(X[:, :params['output_dim']] ** 2, dim=1) + \
            0.5 * torch.sum(torch.sin(X[:, params['output_dim']:params['output_dim']*2]), dim=1) + \
            params['noise'] * torch.randn(n_samples)
        
        return {
            'inputs': X,
            'targets': y.unsqueeze(1),
            'task_type': 'regression'
        }
    
    def _generate_sequence_task(self, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate sequence processing task data."""
        n_samples = 1000
        seq_length = 10
        
        # Create sequences that need to be classified based on patterns
        inputs = torch.randn(n_samples, seq_length, params['input_dim'] // seq_length)
        
        # Target is based on sequence properties (e.g., increasing/decreasing trend)
        trends = torch.sum(torch.diff(inputs, dim=1), dim=1).sum(dim=1)
        targets = (trends > 0).long()
        
        # Flatten sequences for feedforward processing
        inputs_flat = inputs.view(n_samples, -1)
        
        return {
            'inputs': inputs_flat,
            'targets': targets,
            'task_type': 'sequence'
        }
    
    def _generate_spatial_task(self, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate spatial processing task data."""
        n_samples = 1000
        grid_size = int(np.sqrt(params['input_dim']))
        
        # Create 2D grid data
        inputs = torch.randn(n_samples, grid_size, grid_size)
        
        # Target based on spatial properties (e.g., presence of patterns)
        # Look for cross-like patterns
        targets = torch.zeros(n_samples, dtype=torch.long)
        for i in range(n_samples):
            grid = inputs[i]
            center = grid_size // 2
            # Check for cross pattern (high values in cross shape)
            cross_sum = (grid[center, :].sum() + grid[:, center].sum() - grid[center, center])
            targets[i] = (cross_sum > grid.sum() * 0.3).long()
        
        # Flatten for feedforward processing
        inputs_flat = inputs.view(n_samples, -1)
        
        return {
            'inputs': inputs_flat,
            'targets': targets,
            'task_type': 'spatial'
        }
    
    def _flatten_sequences(self) -> List[Dict[str, Any]]:
        """Flatten task sequences into individual trials."""
        flat_data = []
        
        for sequence in self.sequences:
            for trial in sequence:
                task_type = trial['task_type']
                task_data = self.task_data[task_type]
                
                # Sample random data point from this task
                sample_idx = random.randint(0, len(task_data['inputs']) - 1)
                
                flat_trial = {
                    'input': task_data['inputs'][sample_idx],
                    'target': task_data['targets'][sample_idx],
                    'task_id': trial['task_type'],
                    'is_switch': trial['is_switch'],
                    'sequence_position': trial['sequence_position'],
                    'sequence_id': trial['sequence_id']
                }
                
                flat_data.append(flat_trial)
        
        return flat_data
    
    def __len__(self) -> int:
        return len(self.flat_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.flat_data[idx]
    
    def get_sequence(self, sequence_id: int) -> List[Dict[str, Any]]:
        """Get a complete task switching sequence."""
        sequence_data = []
        for trial in self.flat_data:
            if trial['sequence_id'] == sequence_id:
                sequence_data.append(trial)
        
        # Sort by sequence position
        sequence_data.sort(key=lambda x: x['sequence_position'])
        return sequence_data


class InterferenceDataset(Dataset):
    """
    Dataset for interference and cognitive control experiments.
    
    Creates Stroop-like interference tasks where multiple sources of
    information compete for processing resources.
    
    Args:
        interference_type (str): Type of interference ('stroop', 'flanker', 'dual_task')
        num_samples (int): Total number of samples
        interference_probability (float): Probability of interference on each trial
        feature_dim (int): Dimensionality of input features
    """
    
    def __init__(
        self,
        interference_type: str = 'stroop',
        num_samples: int = 2000,
        interference_probability: float = 0.5,
        feature_dim: int = 64
    ):
        self.interference_type = interference_type
        self.num_samples = num_samples
        self.interference_probability = interference_probability
        self.feature_dim = feature_dim
        
        # Generate interference data
        self.data = self._generate_interference_data()
    
    def _generate_interference_data(self) -> List[Dict[str, Any]]:
        """Generate interference task data."""
        data = []
        
        for i in range(self.num_samples):
            if self.interference_type == 'stroop':
                trial = self._generate_stroop_trial()
            elif self.interference_type == 'flanker':
                trial = self._generate_flanker_trial()
            elif self.interference_type == 'dual_task':
                trial = self._generate_dual_task_trial()
            else:
                trial = self._generate_stroop_trial()  # Default
            
            trial['trial_id'] = i
            data.append(trial)
        
        return data
    
    def _generate_stroop_trial(self) -> Dict[str, Any]:
        """Generate Stroop-like interference trial."""
        # Target task: classify based on first half of features
        target_features = torch.randn(self.feature_dim // 2)
        target_class = (target_features.sum() > 0).long()
        
        # Distractor task: interfering information in second half
        distractor_features = torch.randn(self.feature_dim // 2)
        distractor_class = (distractor_features.sum() > 0).long()
        
        # Create interference condition
        is_interference = random.random() < self.interference_probability
        
        if is_interference:
            # Conflicting information (distractor suggests different response)
            if target_class == 0:
                distractor_features = torch.abs(distractor_features) + 0.5  # Make clearly positive
            else:
                distractor_features = -torch.abs(distractor_features) - 0.5  # Make clearly negative
        else:
            # Congruent information (distractor supports same response)
            if target_class == 0:
                distractor_features = -torch.abs(distractor_features) - 0.5
            else:
                distractor_features = torch.abs(distractor_features) + 0.5
        
        # Combine features
        full_input = torch.cat([target_features, distractor_features])
        
        return {
            'input': full_input,
            'target': target_class,
            'is_interference': is_interference,
            'interference_type': 'stroop',
            'target_features': target_features,
            'distractor_features': distractor_features
        }
    
    def _generate_flanker_trial(self) -> Dict[str, Any]:
        """Generate flanker-like interference trial."""
        # Central target
        target_features = torch.randn(self.feature_dim // 3)
        target_class = (target_features.sum() > 0).long()
        
        # Flanking distractors
        flanker_size = (self.feature_dim - len(target_features)) // 2
        left_flanker = torch.randn(flanker_size)
        right_flanker = torch.randn(flanker_size)
        
        # Create interference
        is_interference = random.random() < self.interference_probability
        
        if is_interference:
            # Incompatible flankers
            flanker_sign = -1 if target_class == 1 else 1
        else:
            # Compatible flankers
            flanker_sign = 1 if target_class == 1 else -1
        
        left_flanker = flanker_sign * torch.abs(left_flanker)
        right_flanker = flanker_sign * torch.abs(right_flanker)
        
        # Combine features
        full_input = torch.cat([left_flanker, target_features, right_flanker])
        
        return {
            'input': full_input,
            'target': target_class,
            'is_interference': is_interference,
            'interference_type': 'flanker',
            'target_features': target_features,
            'flanker_features': torch.cat([left_flanker, right_flanker])
        }
    
    def _generate_dual_task_trial(self) -> Dict[str, Any]:
        """Generate dual-task interference trial."""
        # Primary task
        primary_features = torch.randn(self.feature_dim // 2)
        primary_class = (primary_features.sum() > 0).long()
        
        # Secondary task (when present, creates cognitive load)
        secondary_features = torch.randn(self.feature_dim // 2)
        has_secondary = random.random() < self.interference_probability
        
        if not has_secondary:
            secondary_features = torch.zeros_like(secondary_features)
        
        # Combine tasks
        full_input = torch.cat([primary_features, secondary_features])
        
        return {
            'input': full_input,
            'target': primary_class,
            'is_interference': has_secondary,
            'interference_type': 'dual_task',
            'primary_features': primary_features,
            'secondary_features': secondary_features
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class AutismModelingDataset(Dataset):
    """
    Dataset for autism vs neurotypical modeling experiments.
    
    Creates data with different patterns that reflect hypothesized
    differences in autism processing (local vs global processing,
    pattern completion, sensory processing differences).
    
    Args:
        condition (str): 'neurotypical' or 'autism'
        processing_style (str): 'local', 'global', 'mixed'
        pattern_completion_bias (float): Bias towards pattern completion
        sensory_processing_difference (float): Level of sensory processing difference
        num_samples (int): Number of samples to generate
    """
    
    def __init__(
        self,
        condition: str = 'neurotypical',
        processing_style: str = 'mixed',
        pattern_completion_bias: float = 0.5,
        sensory_processing_difference: float = 0.1,
        num_samples: int = 1000
    ):
        self.condition = condition
        self.processing_style = processing_style
        self.pattern_completion_bias = pattern_completion_bias
        self.sensory_processing_difference = sensory_processing_difference
        self.num_samples = num_samples
        
        # Generate autism-specific data patterns
        self.data = self._generate_autism_data()
    
    def _generate_autism_data(self) -> List[Dict[str, Any]]:
        """Generate data reflecting autism vs neurotypical processing differences."""
        data = []
        
        for i in range(self.num_samples):
            if self.condition == 'autism':
                trial = self._generate_autism_trial()
            else:
                trial = self._generate_neurotypical_trial()
            
            trial['trial_id'] = i
            trial['condition'] = self.condition
            data.append(trial)
        
        return data
    
    def _generate_autism_trial(self) -> Dict[str, Any]:
        """Generate trial with autism-like processing characteristics."""
        # Local processing bias: focus on details
        if self.processing_style in ['local', 'mixed']:
            # Create data with strong local features
            local_features = self._create_local_pattern_features()
            global_features = self._create_global_pattern_features() * 0.3  # Reduced global
        else:
            local_features = self._create_local_pattern_features() * 0.7
            global_features = self._create_global_pattern_features()
        
        # Pattern completion differences
        pattern_features = self._create_pattern_completion_features(
            bias=self.pattern_completion_bias * 0.7  # Reduced pattern completion
        )
        
        # Sensory processing differences
        sensory_noise = torch.randn_like(local_features) * self.sensory_processing_difference * 1.5
        
        # Combine features
        full_features = torch.cat([
            local_features + sensory_noise,
            global_features,
            pattern_features
        ])
        
        # Target based on local features (autism-like processing)
        target = (local_features.sum() > 0).long()
        
        return {
            'input': full_features,
            'target': target,
            'local_features': local_features,
            'global_features': global_features,
            'pattern_features': pattern_features,
            'processing_style_bias': 'local'
        }
    
    def _generate_neurotypical_trial(self) -> Dict[str, Any]:
        """Generate trial with neurotypical processing characteristics."""
        # Balanced local/global processing
        local_features = self._create_local_pattern_features()
        global_features = self._create_global_pattern_features()
        
        # Strong pattern completion
        pattern_features = self._create_pattern_completion_features(
            bias=self.pattern_completion_bias * 1.3
        )
        
        # Normal sensory processing
        sensory_noise = torch.randn_like(local_features) * self.sensory_processing_difference
        
        # Combine features with global bias
        full_features = torch.cat([
            local_features + sensory_noise,
            global_features * 1.2,  # Enhanced global processing
            pattern_features
        ])
        
        # Target based on global features (neurotypical processing)
        target = (global_features.sum() > 0).long()
        
        return {
            'input': full_features,
            'target': target,
            'local_features': local_features,
            'global_features': global_features,
            'pattern_features': pattern_features,
            'processing_style_bias': 'global'
        }
    
    def _create_local_pattern_features(self) -> torch.Tensor:
        """Create features representing local/detail processing."""
        # High-frequency, detailed patterns
        features = torch.randn(16)
        
        # Add detailed structure
        for i in range(0, len(features), 2):
            if i + 1 < len(features):
                # Create local correlations
                features[i + 1] = features[i] * 0.8 + torch.randn(1) * 0.2
        
        return features
    
    def _create_global_pattern_features(self) -> torch.Tensor:
        """Create features representing global/gestalt processing."""
        # Smooth, global patterns
        features = torch.randn(16)
        
        # Apply smoothing to create global structure
        for i in range(1, len(features) - 1):
            features[i] = 0.5 * features[i] + 0.25 * (features[i-1] + features[i+1])
        
        return features
    
    def _create_pattern_completion_features(self, bias: float) -> torch.Tensor:
        """Create features for pattern completion tasks."""
        # Create partial patterns that could be completed
        full_pattern = torch.sin(torch.linspace(0, 2*np.pi, 16))
        
        # Remove some parts based on pattern completion bias
        mask = torch.rand(16) < bias
        incomplete_pattern = full_pattern * mask.float()
        
        return incomplete_pattern
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class DRNDataLoader:
    """
    Specialized data loader for DRN training and evaluation.
    
    Provides utilities for creating batches optimized for DRN training,
    including budget management, curriculum learning, and flexible
    batch sampling strategies.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        curriculum_learning: bool = False,
        budget_aware_sampling: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.curriculum_learning = curriculum_learning
        self.budget_aware_sampling = budget_aware_sampling
        
        # Create underlying DataLoader
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate_drn_batch
        )
        
        # Curriculum learning state
        self.current_difficulty = 0.0
        self.difficulty_schedule = self._create_difficulty_schedule()
    
    def _collate_drn_batch(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """Custom collate function for DRN batches."""
        if isinstance(batch[0], dict):
            # Handle complex batch structure
            collated = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], torch.Tensor):
                    collated[key] = torch.stack([item[key] for item in batch])
                else:
                    collated[key] = [item[key] for item in batch]
            return collated
        else:
            # Handle simple (input, target) structure
            inputs = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            return {'inputs': inputs, 'targets': targets}
    
    def _create_difficulty_schedule(self) -> Callable[[int], float]:
        """Create curriculum learning difficulty schedule."""
        def schedule(epoch: int) -> float:
            # Simple linear increase in difficulty
            return min(1.0, epoch / 100.0)
        return schedule
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_concept_learning_experiment(
    num_concepts: int = 4,
    concept_complexity: str = 'medium',
    hierarchical: bool = True,
    train_ratio: float = 0.7
) -> Tuple[ConceptLearningDataset, ConceptLearningDataset, ConceptLearningDataset]:
    """
    Create train/validation/test splits for concept learning experiments.
    
    Args:
        num_concepts: Number of concepts to learn
        concept_complexity: Complexity level
        hierarchical: Whether to use hierarchical structure
        train_ratio: Proportion of data for training
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Create full dataset
    full_dataset = ConceptLearningDataset(
        num_concepts=num_concepts,
        concept_complexity=concept_complexity,
        hierarchical=hierarchical,
        samples_per_concept=300
    )
    
    # Split data
    data, labels = full_dataset.data, full_dataset.labels
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, train_size=train_ratio, stratify=labels, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42
    )
    
    # Create dataset objects
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    return train_dataset, val_dataset, test_dataset


def create_flexibility_benchmark(
    include_tasks: Optional[List[str]] = None,
    difficulty: str = 'medium',
    sequence_length: int = 100
) -> Tuple[TaskSwitchingDataset, InterferenceDataset]:
    """
    Create comprehensive cognitive flexibility benchmark.
    
    Args:
        include_tasks: Tasks to include in switching dataset
        difficulty: Overall difficulty level
        sequence_length: Length of task switching sequences
        
    Returns:
        (task_switching_dataset, interference_dataset)
    """
    if include_tasks is None:
        include_tasks = ['classification', 'regression', 'sequence', 'spatial']
    
    # Task switching dataset
    switching_dataset = TaskSwitchingDataset(
        task_types=include_tasks,
        sequence_length=sequence_length,
        task_difficulty=difficulty,
        switch_probability=0.3
    )
    
    # Interference dataset
    interference_dataset = InterferenceDataset(
        interference_type='stroop',
        num_samples=2000,
        interference_probability=0.5,
        feature_dim=64 if difficulty == 'medium' else (32 if difficulty == 'easy' else 128)
    )
    
    return switching_dataset, interference_dataset


def create_autism_comparison_datasets(
    num_samples: int = 1000,
    processing_differences: bool = True
) -> Tuple[AutismModelingDataset, AutismModelingDataset]:
    """
    Create datasets for autism vs neurotypical comparison studies.
    
    Args:
        num_samples: Number of samples per condition
        processing_differences: Whether to simulate processing differences
        
    Returns:
        (neurotypical_dataset, autism_dataset)
    """
    # Neurotypical dataset
    neurotypical_dataset = AutismModelingDataset(
        condition='neurotypical',
        processing_style='mixed',
        pattern_completion_bias=0.7,
        sensory_processing_difference=0.1,
        num_samples=num_samples
    )
    
    # Autism dataset
    autism_params = {
        'condition': 'autism',
        'processing_style': 'local',
        'pattern_completion_bias': 0.3,
        'sensory_processing_difference': 0.3,
        'num_samples': num_samples
    }
    
    if not processing_differences:
        # Reduce differences for control condition
        autism_params['processing_style'] = 'mixed'
        autism_params['pattern_completion_bias'] = 0.6
        autism_params['sensory_processing_difference'] = 0.15
    
    autism_dataset = AutismModelingDataset(**autism_params)
    
    return neurotypical_dataset, autism_dataset


def visualize_dataset(dataset: Dataset, num_samples: int = 100) -> None:
    """
    Visualize dataset characteristics using dimensionality reduction.
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to plot
    """
    # Sample data
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    if hasattr(dataset, 'data') and hasattr(dataset, 'labels'):
        # Direct access to data and labels
        X = dataset.data[indices].numpy()
        y = dataset.labels[indices].numpy()
    else:
        # Sample through __getitem__
        samples = [dataset[i] for i in indices]
        if isinstance(samples[0], dict):
            X = torch.stack([s['input'] for s in samples]).numpy()
            y = torch.stack([s['target'] for s in samples]).numpy()
        else:
            X = torch.stack([s[0] for s in samples]).numpy()
            y = torch.stack([s[1] for s in samples]).numpy()
    
    # Dimensionality reduction
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.title('PCA Visualization')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.colorbar(scatter)
    
    # t-SNE (if not too many samples)
    if len(X) <= 500:
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()