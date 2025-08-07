"""
drn/analysis/flexibility_metrics.py

Cognitive flexibility metrics for Dynamic Recruitment Networks.
This module provides comprehensive measures of cognitive flexibility inspired by
psychological and neuroscientific research, particularly relevant for autism
research and concept learning studies.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import cosine
import warnings


class FlexibilityMetrics:
    """
    Comprehensive cognitive flexibility metrics for DRN analysis.
    
    This class implements various measures of cognitive flexibility including:
    - Task switching costs and efficiency
    - Concept formation and adaptation
    - Interference resistance and cognitive control
    - Set-shifting and rule learning
    - Working memory flexibility
    
    Args:
        track_detailed_history (bool): Whether to maintain detailed metric history
        baseline_window (int): Window size for baseline performance calculation
        adaptation_threshold (float): Threshold for detecting adaptation events
    """
    
    def __init__(
        self,
        track_detailed_history: bool = True,
        baseline_window: int = 50,
        adaptation_threshold: float = 0.1
    ):
        self.track_detailed_history = track_detailed_history
        self.baseline_window = baseline_window
        self.adaptation_threshold = adaptation_threshold
        
        # Metric tracking
        self.task_performance_history = defaultdict(list)
        self.switching_costs = []
        self.adaptation_events = []
        self.cognitive_control_measures = {}
        
        # Task-specific histories
        self.current_task = None
        self.task_switches = []
        self.interference_trials = []
        self.set_shifting_performance = []
        
        # Detailed analysis results
        self.flexibility_profile = {}
        self.comparative_analysis = {}
        
        # Running statistics
        self.running_stats = {
            'mean_switching_cost': deque(maxlen=100),
            'adaptation_speed': deque(maxlen=100),
            'interference_resistance': deque(maxlen=100),
            'cognitive_stability': deque(maxlen=100)
        }
    
    def measure_task_switching_flexibility(
        self,
        model: torch.nn.Module,
        task_sequence: List[Dict[str, Any]],
        performance_metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Measure cognitive flexibility through task switching paradigms.
        
        Args:
            model: DRN model to evaluate
            task_sequence: Sequence of tasks with 'task_id', 'data', 'labels'
            performance_metric: Metric to use ('accuracy', 'loss', 'custom')
            
        Returns:
            Dictionary of task switching flexibility metrics
        """
        switching_results = {
            'switching_costs': [],
            'task_performances': {},
            'adaptation_curves': {},
            'switch_vs_repeat_analysis': {},
            'cognitive_load_effects': {}
        }
        
        model.eval()
        previous_task = None
        
        for trial_idx, task_trial in enumerate(task_sequence):
            task_id = task_trial['task_id']
            data = task_trial['data']
            labels = task_trial['labels']
            
            # Detect task switch
            is_switch = previous_task is not None and previous_task != task_id
            
            # Measure performance on current trial
            with torch.no_grad():
                outputs, network_info = model(data, return_layer_info=True)
                
                if performance_metric == 'accuracy':
                    predicted = torch.argmax(outputs, dim=-1)
                    performance = (predicted == labels).float().mean().item()
                else:
                    # Could implement other metrics
                    performance = torch.norm(outputs - labels).item()
            
            # Track performance
            if task_id not in switching_results['task_performances']:
                switching_results['task_performances'][task_id] = []
            switching_results['task_performances'][task_id].append(performance)
            
            # Calculate switching cost if this is a switch trial
            if is_switch:
                switching_cost = self._calculate_switching_cost(
                    task_id, previous_task, performance, switching_results['task_performances']
                )
                switching_results['switching_costs'].append({
                    'trial': trial_idx,
                    'from_task': previous_task,
                    'to_task': task_id,
                    'cost': switching_cost,
                    'performance': performance,
                    'network_state': self._extract_network_state(network_info)
                })
                
                self.switching_costs.append(switching_cost)
                self.running_stats['mean_switching_cost'].append(switching_cost)
            
            # Analyze recruitment patterns for flexibility
            recruitment_flexibility = self._analyze_recruitment_flexibility(
                network_info, task_id, is_switch
            )
            
            # Store detailed trial information
            if self.track_detailed_history:
                self.task_performance_history[task_id].append({
                    'trial': trial_idx,
                    'performance': performance,
                    'is_switch': is_switch,
                    'recruitment_pattern': recruitment_flexibility,
                    'network_info': network_info
                })
            
            previous_task = task_id
        
        # Analyze adaptation curves
        switching_results['adaptation_curves'] = self._analyze_adaptation_curves(
            switching_results['task_performances']
        )
        
        # Compare switch vs repeat trials
        switching_results['switch_vs_repeat_analysis'] = self._analyze_switch_vs_repeat(
            task_sequence, switching_results['task_performances']
        )
        
        # Measure cognitive load effects
        switching_results['cognitive_load_effects'] = self._measure_cognitive_load_effects(
            switching_results['switching_costs']
        )
        
        return switching_results
    
    def _calculate_switching_cost(
        self,
        current_task: str,
        previous_task: str,
        current_performance: float,
        task_performances: Dict[str, List[float]]
    ) -> float:
        """Calculate switching cost as performance difference."""
        if current_task not in task_performances or len(task_performances[current_task]) < 2:
            return 0.0
        
        # Compare current performance to baseline performance on this task
        baseline_performance = np.mean(task_performances[current_task][:-1])
        switching_cost = baseline_performance - current_performance
        
        return switching_cost
    
    def _extract_network_state(self, network_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant network state for flexibility analysis."""
        return {
            'total_recruited': network_info.get('total_neurons_recruited', 0),
            'network_sparsity': network_info.get('network_sparsity', 0),
            'recruitment_efficiency': network_info.get('recruitment_efficiency', 0),
            'layer_activities': [info.get('num_recruited', 0) 
                               for info in network_info.get('layer_infos', [])]
        }
    
    def _analyze_recruitment_flexibility(
        self,
        network_info: Dict[str, Any],
        task_id: str,
        is_switch: bool
    ) -> Dict[str, Any]:
        """Analyze how recruitment patterns reflect cognitive flexibility."""
        flexibility_measures = {
            'recruitment_diversity': 0.0,
            'pattern_novelty': 0.0,
            'adaptation_efficiency': 0.0,
            'population_reconfiguration': 0.0
        }
        
        # Extract recruitment pattern
        if 'layer_infos' in network_info:
            layer_recruitments = [info.get('num_recruited', 0) for info in network_info['layer_infos']]
            
            # Recruitment diversity (entropy-like measure)
            if layer_recruitments and sum(layer_recruitments) > 0:
                normalized_recruitments = np.array(layer_recruitments) / sum(layer_recruitments)
                flexibility_measures['recruitment_diversity'] = -np.sum(
                    normalized_recruitments * np.log(normalized_recruitments + 1e-10)
                )
            
            # Pattern novelty (if we have previous patterns for this task)
            if task_id in self.task_performance_history:
                previous_patterns = [
                    trial['recruitment_pattern']['recruitment_diversity']
                    for trial in self.task_performance_history[task_id]
                    if 'recruitment_pattern' in trial
                ]
                
                if previous_patterns:
                    current_diversity = flexibility_measures['recruitment_diversity']
                    mean_previous = np.mean(previous_patterns)
                    flexibility_measures['pattern_novelty'] = abs(current_diversity - mean_previous)
        
        # Population reconfiguration analysis
        if 'connectivity_patterns' in network_info:
            flexibility_measures['population_reconfiguration'] = self._measure_population_reconfiguration(
                network_info['connectivity_patterns']
            )
        
        return flexibility_measures
    
    def _measure_population_reconfiguration(self, connectivity_patterns: Dict[str, Any]) -> float:
        """Measure how much population connectivity reconfigures."""
        # This would analyze population-level connectivity changes
        # For now, return a simplified measure
        recruitment_dist = connectivity_patterns.get('recruitment_distribution', [])
        if recruitment_dist:
            return float(np.std(recruitment_dist))
        return 0.0
    
    def _analyze_adaptation_curves(self, task_performances: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze learning and adaptation curves for each task."""
        adaptation_analysis = {}
        
        for task_id, performances in task_performances.items():
            if len(performances) < 3:
                continue
            
            # Fit learning curve
            x = np.arange(len(performances))
            y = np.array(performances)
            
            # Simple exponential fit: y = a + b * exp(-c * x)
            try:
                from scipy.optimize import curve_fit
                
                def exp_model(x, a, b, c):
                    return a + b * np.exp(-c * x)
                
                popt, _ = curve_fit(exp_model, x, y, maxfev=1000)
                
                adaptation_analysis[task_id] = {
                    'initial_performance': performances[0],
                    'final_performance': performances[-1],
                    'improvement': performances[-1] - performances[0],
                    'learning_rate': popt[2] if len(popt) > 2 else 0.0,
                    'adaptation_speed': self._calculate_adaptation_speed(performances),
                    'stability_index': self._calculate_stability_index(performances)
                }
            except:
                # Fallback to simple metrics
                adaptation_analysis[task_id] = {
                    'initial_performance': performances[0],
                    'final_performance': performances[-1],
                    'improvement': performances[-1] - performances[0],
                    'adaptation_speed': self._calculate_adaptation_speed(performances),
                    'stability_index': self._calculate_stability_index(performances)
                }
        
        return adaptation_analysis
    
    def _calculate_adaptation_speed(self, performances: List[float]) -> float:
        """Calculate how quickly performance adapts to optimal level."""
        if len(performances) < 3:
            return 0.0
        
        # Find the trial where performance reaches 90% of final performance
        final_perf = np.mean(performances[-5:])  # Average of last 5 trials
        target_perf = 0.9 * final_perf
        
        for i, perf in enumerate(performances):
            if perf >= target_perf:
                return 1.0 / (i + 1)  # Faster adaptation = higher score
        
        return 1.0 / len(performances)
    
    def _calculate_stability_index(self, performances: List[float]) -> float:
        """Calculate how stable performance is after initial adaptation."""
        if len(performances) < 10:
            return 0.0
        
        # Use last 50% of trials to measure stability
        stable_period = performances[len(performances)//2:]
        return 1.0 / (np.std(stable_period) + 1e-8)
    
    def _analyze_switch_vs_repeat(
        self,
        task_sequence: List[Dict[str, Any]],
        task_performances: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compare performance on switch vs repeat trials."""
        switch_trials = []
        repeat_trials = []
        
        previous_task = None
        trial_idx = 0
        
        for task_trial in task_sequence:
            current_task = task_trial['task_id']
            
            if previous_task is not None:
                performance = task_performances[current_task][trial_idx - 1] if trial_idx > 0 else 0
                
                if previous_task != current_task:
                    switch_trials.append(performance)
                else:
                    repeat_trials.append(performance)
            
            previous_task = current_task
            trial_idx += 1
        
        analysis = {
            'switch_performance': {
                'mean': np.mean(switch_trials) if switch_trials else 0,
                'std': np.std(switch_trials) if switch_trials else 0,
                'trials': len(switch_trials)
            },
            'repeat_performance': {
                'mean': np.mean(repeat_trials) if repeat_trials else 0,
                'std': np.std(repeat_trials) if repeat_trials else 0,
                'trials': len(repeat_trials)
            }
        }
        
        # Calculate switch cost
        if switch_trials and repeat_trials:
            analysis['switch_cost'] = analysis['repeat_performance']['mean'] - analysis['switch_performance']['mean']
            
            # Statistical significance test
            if len(switch_trials) > 5 and len(repeat_trials) > 5:
                t_stat, p_value = stats.ttest_ind(repeat_trials, switch_trials)
                analysis['statistical_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return analysis
    
    def _measure_cognitive_load_effects(self, switching_costs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Measure how cognitive load affects flexibility."""
        if len(switching_costs) < 5:
            return {}
        
        costs = [sc['cost'] for sc in switching_costs]
        
        return {
            'mean_switching_cost': np.mean(costs),
            'switching_cost_variance': np.var(costs),
            'load_adaptation': self._measure_load_adaptation(costs),
            'cost_distribution': {
                'percentiles': [np.percentile(costs, p) for p in [25, 50, 75, 90, 95]],
                'outliers': len([c for c in costs if abs(c - np.mean(costs)) > 2 * np.std(costs)])
            }
        }
    
    def _measure_load_adaptation(self, switching_costs: List[float]) -> float:
        """Measure how switching costs change over time (adaptation to cognitive load)."""
        if len(switching_costs) < 10:
            return 0.0
        
        # Compare early vs late switching costs
        early_costs = switching_costs[:len(switching_costs)//3]
        late_costs = switching_costs[-len(switching_costs)//3:]
        
        early_mean = np.mean(early_costs)
        late_mean = np.mean(late_costs)
        
        # Positive value means costs decreased (better adaptation)
        adaptation = (early_mean - late_mean) / (early_mean + 1e-8)
        return adaptation
    
    def measure_interference_resistance(
        self,
        model: torch.nn.Module,
        primary_task_data: torch.Tensor,
        interference_task_data: torch.Tensor,
        primary_labels: torch.Tensor,
        interference_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Measure resistance to interference using dual-task paradigms.
        
        This implements cognitive control measures similar to Stroop or
        dual-task interference paradigms used in psychology.
        """
        interference_results = {
            'baseline_performance': 0.0,
            'interference_performance': 0.0,
            'interference_cost': 0.0,
            'control_efficiency': 0.0,
            'selective_attention': {},
            'cognitive_control': {}
        }
        
        model.eval()
        
        # Measure baseline performance (primary task only)
        with torch.no_grad():
            primary_outputs, primary_info = model(primary_task_data, return_layer_info=True)
            primary_predictions = torch.argmax(primary_outputs, dim=-1)
            baseline_accuracy = (primary_predictions == primary_labels).float().mean().item()
            interference_results['baseline_performance'] = baseline_accuracy
        
        # Measure interference performance (dual task)
        # Simulate dual task by presenting both stimuli
        combined_input = self._create_dual_task_input(primary_task_data, interference_task_data)
        
        with torch.no_grad():
            dual_outputs, dual_info = model(combined_input, return_layer_info=True)
            dual_predictions = torch.argmax(dual_outputs, dim=-1)
            interference_accuracy = (dual_predictions == primary_labels).float().mean().item()
            interference_results['interference_performance'] = interference_accuracy
        
        # Calculate interference cost
        interference_cost = baseline_accuracy - interference_accuracy
        interference_results['interference_cost'] = interference_cost
        
        # Measure control efficiency
        control_efficiency = self._measure_control_efficiency(primary_info, dual_info)
        interference_results['control_efficiency'] = control_efficiency
        
        # Analyze selective attention
        interference_results['selective_attention'] = self._analyze_selective_attention(
            primary_info, dual_info
        )
        
        # Measure cognitive control mechanisms
        interference_results['cognitive_control'] = self._measure_cognitive_control(
            primary_info, dual_info, interference_cost
        )
        
        # Store for trend analysis
        self.interference_trials.append(interference_results)
        self.running_stats['interference_resistance'].append(1.0 - interference_cost)
        
        return interference_results
    
    def _create_dual_task_input(
        self,
        primary_input: torch.Tensor,
        interference_input: torch.Tensor
    ) -> torch.Tensor:
        """Create combined input for dual-task interference testing."""
        # Simple concatenation - could be made more sophisticated
        if primary_input.shape == interference_input.shape:
            # Weighted combination
            combined = 0.7 * primary_input + 0.3 * interference_input
        else:
            # Pad to same size and combine
            max_size = max(primary_input.shape[-1], interference_input.shape[-1])
            primary_padded = torch.nn.functional.pad(
                primary_input, (0, max_size - primary_input.shape[-1])
            )
            interference_padded = torch.nn.functional.pad(
                interference_input, (0, max_size - interference_input.shape[-1])
            )
            combined = torch.cat([primary_padded, interference_padded], dim=-1)
        
        return combined
    
    def _measure_control_efficiency(
        self,
        primary_info: Dict[str, Any],
        dual_info: Dict[str, Any]
    ) -> float:
        """Measure how efficiently the network maintains control under interference."""
        primary_recruitment = primary_info.get('total_neurons_recruited', 0)
        dual_recruitment = dual_info.get('total_neurons_recruited', 0)
        
        if primary_recruitment == 0:
            return 0.0
        
        # Control efficiency = maintaining similar recruitment under interference
        efficiency = 1.0 - abs(dual_recruitment - primary_recruitment) / primary_recruitment
        return max(0.0, efficiency)
    
    def _analyze_selective_attention(
        self,
        primary_info: Dict[str, Any],
        dual_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze selective attention mechanisms."""
        attention_analysis = {
            'attention_focus': 0.0,
            'distractor_suppression': 0.0,
            'target_enhancement': 0.0
        }
        
        # Compare layer-wise recruitment patterns
        if ('layer_infos' in primary_info and 'layer_infos' in dual_info and
            len(primary_info['layer_infos']) == len(dual_info['layer_infos'])):
            
            primary_pattern = [info.get('num_recruited', 0) for info in primary_info['layer_infos']]
            dual_pattern = [info.get('num_recruited', 0) for info in dual_info['layer_infos']]
            
            # Attention focus = similarity to primary task pattern
            if sum(primary_pattern) > 0 and sum(dual_pattern) > 0:
                primary_norm = np.array(primary_pattern) / sum(primary_pattern)
                dual_norm = np.array(dual_pattern) / sum(dual_pattern)
                
                attention_focus = 1.0 - cosine(primary_norm, dual_norm)
                attention_analysis['attention_focus'] = max(0.0, attention_focus)
        
        return attention_analysis
    
    def _measure_cognitive_control(
        self,
        primary_info: Dict[str, Any],
        dual_info: Dict[str, Any],
        interference_cost: float
    ) -> Dict[str, Any]:
        """Measure cognitive control mechanisms."""
        control_measures = {
            'inhibitory_control': 0.0,
            'working_memory_maintenance': 0.0,
            'cognitive_flexibility': 0.0,
            'conflict_monitoring': 0.0
        }
        
        # Inhibitory control = resistance to interference
        control_measures['inhibitory_control'] = max(0.0, 1.0 - interference_cost)
        
        # Working memory maintenance = stability of recruitment patterns
        primary_sparsity = primary_info.get('network_sparsity', 0)
        dual_sparsity = dual_info.get('network_sparsity', 0)
        
        control_measures['working_memory_maintenance'] = 1.0 - abs(primary_sparsity - dual_sparsity)
        
        # Conflict monitoring = ability to detect interference
        recruitment_change = abs(
            dual_info.get('total_neurons_recruited', 0) - 
            primary_info.get('total_neurons_recruited', 0)
        )
        control_measures['conflict_monitoring'] = min(1.0, recruitment_change / 100.0)
        
        return control_measures
    
    def measure_set_shifting_ability(
        self,
        model: torch.nn.Module,
        rule_learning_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure set-shifting ability using rule learning paradigms.
        
        Similar to Wisconsin Card Sorting Test or other rule-learning tasks.
        """
        set_shifting_results = {
            'rule_acquisition': {},
            'rule_maintenance': {},
            'rule_switching': {},
            'perseverative_errors': 0,
            'set_loss_errors': 0,
            'dimensional_changes': {}
        }
        
        model.eval()
        current_rule = None
        perseverative_count = 0
        
        for task_idx, task in enumerate(rule_learning_tasks):
            rule_id = task['rule_id']
            data = task['data']
            labels = task['labels']
            is_rule_switch = current_rule is not None and current_rule != rule_id
            
            with torch.no_grad():
                outputs, network_info = model(data, return_layer_info=True)
                predictions = torch.argmax(outputs, dim=-1)
                correct = (predictions == labels).float().mean().item()
            
            # Track rule acquisition
            if rule_id not in set_shifting_results['rule_acquisition']:
                set_shifting_results['rule_acquisition'][rule_id] = {
                    'trials_to_criterion': 0,
                    'acquisition_curve': [],
                    'initial_performance': correct
                }
            
            set_shifting_results['rule_acquisition'][rule_id]['acquisition_curve'].append(correct)
            
            # Detect perseverative errors (continuing with old rule)
            if is_rule_switch and correct < 0.5:  # Below chance performance
                perseverative_count += 1
            
            # Analyze network adaptation to rule changes
            if is_rule_switch:
                rule_switch_analysis = self._analyze_rule_switching(
                    network_info, current_rule, rule_id
                )
                set_shifting_results['rule_switching'][f'{current_rule}_to_{rule_id}'] = rule_switch_analysis
            
            current_rule = rule_id
        
        set_shifting_results['perseverative_errors'] = perseverative_count
        
        # Analyze dimensional changes
        set_shifting_results['dimensional_changes'] = self._analyze_dimensional_changes(
            rule_learning_tasks, set_shifting_results['rule_acquisition']
        )
        
        return set_shifting_results
    
    def _analyze_rule_switching(
        self,
        network_info: Dict[str, Any],
        old_rule: str,
        new_rule: str
    ) -> Dict[str, Any]:
        """Analyze how the network adapts when rules change."""
        switching_analysis = {
            'adaptation_speed': 0.0,
            'network_reconfiguration': 0.0,
            'recruitment_change': 0.0
        }
        
        # This would compare network states before and after rule switch
        # For now, return basic metrics based on current state
        switching_analysis['recruitment_change'] = network_info.get('total_neurons_recruited', 0)
        switching_analysis['network_reconfiguration'] = network_info.get('network_sparsity', 0)
        
        return switching_analysis
    
    def _analyze_dimensional_changes(
        self,
        rule_tasks: List[Dict[str, Any]],
        acquisition_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance on different types of dimensional changes."""
        dimensional_analysis = {}
        
        # Group rules by dimension type if available
        for rule_id, rule_data in acquisition_data.items():
            curve = rule_data['acquisition_curve']
            if len(curve) >= 3:
                dimensional_analysis[rule_id] = {
                    'learning_speed': self._calculate_adaptation_speed(curve),
                    'final_performance': np.mean(curve[-3:]),
                    'stability': 1.0 / (np.std(curve[-5:]) + 1e-8) if len(curve) >= 5 else 0
                }
        
        return dimensional_analysis
    
    def compute_flexibility_profile(self) -> Dict[str, Any]:
        """Compute comprehensive cognitive flexibility profile."""
        profile = {
            'overall_flexibility_score': 0.0,
            'domain_specific_scores': {},
            'strengths_and_weaknesses': {},
            'autism_relevance_measures': {},
            'comparative_percentiles': {}
        }
        
        # Task switching flexibility
        if self.switching_costs:
            switching_score = 1.0 / (np.mean(self.switching_costs) + 1.0)
            profile['domain_specific_scores']['task_switching'] = switching_score
        
        # Interference resistance
        if self.running_stats['interference_resistance']:
            interference_score = np.mean(list(self.running_stats['interference_resistance']))
            profile['domain_specific_scores']['interference_resistance'] = interference_score
        
        # Adaptation efficiency
        if self.running_stats['adaptation_speed']:
            adaptation_score = np.mean(list(self.running_stats['adaptation_speed']))
            profile['domain_specific_scores']['adaptation_speed'] = adaptation_score
        
        # Calculate overall flexibility score
        domain_scores = list(profile['domain_specific_scores'].values())
        if domain_scores:
            profile['overall_flexibility_score'] = np.mean(domain_scores)
        
        # Identify strengths and weaknesses
        profile['strengths_and_weaknesses'] = self._identify_flexibility_profile(
            profile['domain_specific_scores']
        )
        
        # Autism-relevant measures
        profile['autism_relevance_measures'] = self._compute_autism_relevant_measures()
        
        return profile
    
    def _identify_flexibility_profile(self, domain_scores: Dict[str, float]) -> Dict[str, Any]:
        """Identify cognitive flexibility strengths and weaknesses."""
        if not domain_scores:
            return {}
        
        scores = list(domain_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        strengths = []
        weaknesses = []
        
        for domain, score in domain_scores.items():
            if score > mean_score + 0.5 * std_score:
                strengths.append(domain)
            elif score < mean_score - 0.5 * std_score:
                weaknesses.append(domain)
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'profile_variability': std_score,
            'balanced_profile': std_score < 0.2  # Low variability = balanced
        }
    
    def _compute_autism_relevant_measures(self) -> Dict[str, Any]:
        """Compute measures particularly relevant to autism research."""
        autism_measures = {
            'cognitive_rigidity_index': 0.0,
            'local_vs_global_processing': 0.0,
            'detail_focus_tendency': 0.0,
            'pattern_completion_preference': 0.0,
            'sensory_processing_differences': 0.0
        }
        
        # Cognitive rigidity = high switching costs + low adaptation
        if self.switching_costs and self.running_stats['adaptation_speed']:
            mean_switching_cost = np.mean(self.switching_costs)
            mean_adaptation = np.mean(list(self.running_stats['adaptation_speed']))
            
            # Higher costs and slower adaptation = more rigidity
            rigidity = (mean_switching_cost / 0.5) + (1.0 - mean_adaptation)
            autism_measures['cognitive_rigidity_index'] = min(2.0, rigidity) / 2.0
        
        # Detail focus = high interference resistance + low global processing
        if self.running_stats['interference_resistance']:
            detail_focus = np.mean(list(self.running_stats['interference_resistance']))
            autism_measures['detail_focus_tendency'] = detail_focus
        
        return autism_measures


class ConnectivityMetrics:
    """
    Metrics for analyzing connectivity patterns in relation to cognitive flexibility.
    
    This class provides measures that connect network connectivity with
    behavioral flexibility measures.
    """
    
    def __init__(self):
        self.connectivity_flexibility_correlations = {}
        self.dynamic_connectivity_measures = {}
    
    def correlate_connectivity_with_flexibility(
        self,
        connectivity_data: List[Dict[str, Any]],
        flexibility_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Correlate connectivity patterns with flexibility measures."""
        correlations = {}
        
        if len(connectivity_data) != len(flexibility_data):
            return correlations
        
        # Extract connectivity features
        sparsity_values = [conn.get('network_sparsity', 0) for conn in connectivity_data]
        recruitment_values = [conn.get('total_neurons_recruited', 0) for conn in connectivity_data]
        
        # Extract flexibility features
        switching_costs = [flex.get('switching_cost', 0) for flex in flexibility_data]
        adaptation_speeds = [flex.get('adaptation_speed', 0) for flex in flexibility_data]
        
        # Calculate correlations
        if len(sparsity_values) > 3:
            try:
                sparsity_switching_corr = np.corrcoef(sparsity_values, switching_costs)[0, 1]
                recruitment_adaptation_corr = np.corrcoef(recruitment_values, adaptation_speeds)[0, 1]
                
                correlations['sparsity_switching_correlation'] = float(sparsity_switching_corr)
                correlations['recruitment_adaptation_correlation'] = float(recruitment_adaptation_corr)
            except:
                pass
        
        return correlations
    
    def measure_dynamic_connectivity_flexibility(
        self,
        connectivity_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Measure how connectivity patterns change to support flexibility."""
        dynamic_measures = {
            'connectivity_variability': 0.0,
            'adaptive_reconfiguration': 0.0,
            'stability_flexibility_tradeoff': 0.0
        }
        
        if len(connectivity_history) < 5:
            return dynamic_measures
        
        # Connectivity variability
        sparsity_series = [conn.get('network_sparsity', 0) for conn in connectivity_history]
        dynamic_measures['connectivity_variability'] = float(np.std(sparsity_series))
        
        # Adaptive reconfiguration (how much connectivity changes in response to task demands)
        recruitment_series = [conn.get('total_neurons_recruited', 0) for conn in connectivity_history]
        recruitment_changes = np.diff(recruitment_series)
        dynamic_measures['adaptive_reconfiguration'] = float(np.mean(np.abs(recruitment_changes)))
        
        return dynamic_measures


class ComparisonMetrics:
    """
    Metrics for comparing DRN flexibility with baseline models.
    
    This class provides standardized comparisons between DRN and traditional
    neural network architectures on flexibility measures.
    """
    
    def __init__(self):
        self.baseline_comparisons = {}
        self.effect_sizes = {}
    
    def compare_flexibility_profiles(
        self,
        drn_profile: Dict[str, Any],
        baseline_profiles: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare DRN flexibility profile with baseline models."""
        comparison = {
            'flexibility_advantages': {},
            'effect_sizes': {},
            'statistical_significance': {},
            'relative_performance': {}
        }
        
        drn_overall = drn_profile.get('overall_flexibility_score', 0)
        
        for model_name, baseline_profile in baseline_profiles.items():
            baseline_overall = baseline_profile.get('overall_flexibility_score', 0)
            
            # Calculate advantage
            advantage = drn_overall - baseline_overall
            comparison['flexibility_advantages'][model_name] = advantage
            
            # Calculate effect size (Cohen's d approximation)
            drn_scores = list(drn_profile.get('domain_specific_scores', {}).values())
            baseline_scores = list(baseline_profile.get('domain_specific_scores', {}).values())
            
            if drn_scores and baseline_scores:
                pooled_std = np.sqrt((np.var(drn_scores) + np.var(baseline_scores)) / 2)
                if pooled_std > 0:
                    effect_size = (np.mean(drn_scores) - np.mean(baseline_scores)) / pooled_std
                    comparison['effect_sizes'][model_name] = effect_size
        
        return comparison
    
    def compute_autism_modeling_comparison(
        self,
        neurotypical_profile: Dict[str, Any],
        autism_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare neurotypical vs autism-like cognitive flexibility profiles."""
        autism_comparison = {
            'cognitive_differences': {},
            'flexibility_deficits': {},
            'preserved_abilities': {},
            'intervention_targets': {}
        }
        
        nt_autism_measures = neurotypical_profile.get('autism_relevance_measures', {})
        autism_autism_measures = autism_profile.get('autism_relevance_measures', {})
        
        for measure, nt_value in nt_autism_measures.items():
            autism_value = autism_autism_measures.get(measure, 0)
            difference = autism_value - nt_value
            
            autism_comparison['cognitive_differences'][measure] = {
                'neurotypical': nt_value,
                'autism_like': autism_value,
                'difference': difference,
                'effect_direction': 'higher' if difference > 0 else 'lower'
            }
            
            # Identify deficits (areas where autism model performs worse)
            if difference > 0.2:  # Threshold for meaningful difference
                autism_comparison['flexibility_deficits'][measure] = difference
            elif difference < -0.2:
                autism_comparison['preserved_abilities'][measure] = abs(difference)
        
        return autism_comparison


def create_flexibility_metrics(
    detailed_tracking: bool = True,
    baseline_window: int = 50
) -> FlexibilityMetrics:
    """
    Factory function to create a FlexibilityMetrics instance.
    
    Args:
        detailed_tracking: Whether to maintain detailed history
        baseline_window: Window size for baseline calculations
        
    Returns:
        Configured FlexibilityMetrics instance
    """
    return FlexibilityMetrics(
        track_detailed_history=detailed_tracking,
        baseline_window=baseline_window
    )


def quick_flexibility_assessment(
    model: torch.nn.Module,
    task_switching_data: List[Dict[str, Any]],
    interference_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
) -> Dict[str, Any]:
    """
    Quick assessment of cognitive flexibility.
    
    Args:
        model: Model to assess
        task_switching_data: Task switching test data
        interference_data: (primary_data, interference_data, primary_labels, interference_labels)
        
    Returns:
        Quick flexibility assessment results
    """
    metrics = create_flexibility_metrics(detailed_tracking=False)
    
    results = {
        'task_switching': {},
        'interference_resistance': {},
        'overall_assessment': {}
    }
    
    # Task switching assessment
    if task_switching_data:
        results['task_switching'] = metrics.measure_task_switching_flexibility(
            model, task_switching_data
        )
    
    # Interference resistance assessment
    if interference_data:
        primary_data, interference_data_tensor, primary_labels, interference_labels = interference_data
        results['interference_resistance'] = metrics.measure_interference_resistance(
            model, primary_data, interference_data_tensor, primary_labels, interference_labels
        )
    
    # Overall assessment
    flexibility_profile = metrics.compute_flexibility_profile()
    results['overall_assessment'] = flexibility_profile
    
    return results