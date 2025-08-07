"""
drn/analysis/connectivity_analyzer.py

Advanced connectivity analysis tools for Dynamic Recruitment Networks.
This module provides comprehensive analysis of recruitment patterns, connectivity
dynamics, and network topology changes during learning and inference.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings


class ConnectivityAnalyzer:
    """
    Main analyzer for DRN connectivity patterns and recruitment dynamics.
    
    This class tracks and analyzes how DRN networks form connections,
    recruit neurons, and adapt their topology over time.
    
    Args:
        track_history (bool): Whether to maintain history of connectivity changes
        max_history_length (int): Maximum number of timesteps to track
        analysis_frequency (int): How often to perform expensive analyses
    """
    
    def __init__(
        self,
        track_history: bool = True,
        max_history_length: int = 1000,
        analysis_frequency: int = 10
    ):
        self.track_history = track_history
        self.max_history_length = max_history_length
        self.analysis_frequency = analysis_frequency
        
        # History tracking
        self.connectivity_history = deque(maxlen=max_history_length)
        self.recruitment_history = deque(maxlen=max_history_length)
        self.network_states = deque(maxlen=max_history_length)
        
        # Analysis results cache
        self._analysis_cache = {}
        self._last_analysis_step = -1
        
        # Pattern detection
        self.recruitment_patterns = defaultdict(list)
        self.connectivity_clusters = {}
        
        # Metrics tracking
        self.running_metrics = {
            'sparsity_trend': deque(maxlen=100),
            'recruitment_efficiency': deque(maxlen=100),
            'pattern_stability': deque(maxlen=100),
            'connectivity_strength': deque(maxlen=100)
        }
    
    def analyze_network_step(
        self, 
        model: torch.nn.Module, 
        network_info: Dict[str, Any],
        inputs: Optional[torch.Tensor] = None,
        step: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze network connectivity for a single forward pass.
        
        Args:
            model: DRN model to analyze
            network_info: Information from forward pass
            inputs: Input data that triggered this connectivity
            step: Current analysis step
            
        Returns:
            Dictionary of connectivity analysis results
        """
        connectivity_analysis = {}
        
        # Extract recruitment information
        recruitment_info = self._extract_recruitment_info(network_info)
        connectivity_analysis['recruitment'] = recruitment_info
        
        # Analyze connectivity patterns
        connectivity_patterns = self._analyze_connectivity_patterns(model, network_info)
        connectivity_analysis['patterns'] = connectivity_patterns
        
        # Measure network topology
        topology_metrics = self._compute_topology_metrics(model, network_info)
        connectivity_analysis['topology'] = topology_metrics
        
        # Analyze population dynamics
        population_dynamics = self._analyze_population_dynamics(network_info)
        connectivity_analysis['population_dynamics'] = population_dynamics
        
        # Track temporal patterns if history is enabled
        if self.track_history:
            temporal_analysis = self._analyze_temporal_patterns(connectivity_analysis, step)
            connectivity_analysis['temporal'] = temporal_analysis
            
            # Store in history
            self.connectivity_history.append(connectivity_analysis.copy())
            self.recruitment_history.append(recruitment_info)
            
            if inputs is not None:
                self.network_states.append({
                    'inputs': inputs.detach().cpu() if isinstance(inputs, torch.Tensor) else inputs,
                    'step': step,
                    'connectivity': connectivity_analysis
                })
        
        # Update running metrics
        self._update_running_metrics(connectivity_analysis)
        
        # Perform expensive analyses periodically
        if step % self.analysis_frequency == 0:
            expensive_analysis = self._perform_expensive_analysis(model, step)
            connectivity_analysis['expensive'] = expensive_analysis
        
        return connectivity_analysis
    
    def _extract_recruitment_info(self, network_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract recruitment information from network info."""
        recruitment_info = {
            'total_recruited': network_info.get('total_neurons_recruited', 0),
            'total_budget_spent': network_info.get('total_budget_spent', 0),
            'recruitment_efficiency': network_info.get('recruitment_efficiency', 0),
            'layer_recruitment': [],
            'population_recruitment': []
        }
        
        # Layer-wise recruitment
        if 'layer_infos' in network_info:
            for i, layer_info in enumerate(network_info['layer_infos']):
                recruitment_info['layer_recruitment'].append({
                    'layer_id': i,
                    'neurons_recruited': layer_info.get('num_recruited', 0),
                    'budget_spent': layer_info.get('budget_spent', 0),
                    'budget_remaining': layer_info.get('budget_remaining', 0),
                    'recruitment_rate': layer_info.get('budget_utilization', 0)
                })
                
                # Population-level recruitment if available
                if 'population_infos' in layer_info:
                    for j, pop_info in enumerate(layer_info['population_infos']):
                        recruitment_info['population_recruitment'].append({
                            'layer_id': i,
                            'population_id': j,
                            'activity_level': pop_info.get('mean_activity', 0),
                            'sparsity': pop_info.get('sparsity', 0),
                            'coherence': pop_info.get('coherence', 0)
                        })
        
        return recruitment_info
    
    def _analyze_connectivity_patterns(
        self, 
        model: torch.nn.Module, 
        network_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze current connectivity patterns in the network."""
        patterns = {
            'sparsity_distribution': [],
            'connection_strengths': [],
            'hub_analysis': {},
            'motifs': {}
        }
        
        # Analyze each layer's connectivity
        if hasattr(model, 'drn_layers'):
            for i, layer in enumerate(model.drn_layers):
                layer_patterns = self._analyze_layer_connectivity(layer, i)
                patterns['sparsity_distribution'].append(layer_patterns['sparsity'])
                patterns['connection_strengths'].append(layer_patterns['strengths'])
        
        # Inter-layer connectivity analysis
        if hasattr(model, 'inter_layer_connections') and model.inter_layer_connections:
            inter_layer_patterns = self._analyze_inter_layer_connectivity(model)
            patterns['inter_layer'] = inter_layer_patterns
        
        # Identify connectivity hubs (highly connected neurons)
        patterns['hub_analysis'] = self._identify_connectivity_hubs(model)
        
        # Detect common connectivity motifs
        patterns['motifs'] = self._detect_connectivity_motifs(model)
        
        return patterns
    
    def _analyze_layer_connectivity(self, layer: torch.nn.Module, layer_id: int) -> Dict[str, Any]:
        """Analyze connectivity patterns within a single layer."""
        patterns = {
            'layer_id': layer_id,
            'sparsity': 0.0,
            'strengths': [],
            'clustering_coefficient': 0.0,
            'small_world_index': 0.0
        }
        
        # Extract weight matrices
        weights = []
        if hasattr(layer, 'neuron_weights'):
            weights.append(layer.neuron_weights.detach().cpu().numpy())
        if hasattr(layer, 'population_weights'):
            weights.append(layer.population_weights.detach().cpu().numpy())
        
        if not weights:
            return patterns
        
        # Combine all weights for analysis
        combined_weights = np.concatenate([w.flatten() for w in weights])
        
        # Sparsity analysis
        threshold = 0.01 * np.max(np.abs(combined_weights)) if np.max(np.abs(combined_weights)) > 0 else 0
        active_connections = np.abs(combined_weights) > threshold
        patterns['sparsity'] = 1.0 - np.mean(active_connections)
        
        # Connection strength distribution
        active_strengths = np.abs(combined_weights[active_connections])
        if len(active_strengths) > 0:
            patterns['strengths'] = {
                'mean': float(np.mean(active_strengths)),
                'std': float(np.std(active_strengths)),
                'max': float(np.max(active_strengths)),
                'percentiles': [float(np.percentile(active_strengths, p)) for p in [25, 50, 75, 90, 95]]
            }
        
        # Graph theory metrics (for largest weight matrix)
        if weights:
            largest_weight_matrix = max(weights, key=lambda x: x.size)
            patterns.update(self._compute_graph_metrics(largest_weight_matrix, threshold))
        
        return patterns
    
    def _compute_graph_metrics(self, weight_matrix: np.ndarray, threshold: float) -> Dict[str, float]:
        """Compute graph theory metrics for a weight matrix."""
        metrics = {
            'clustering_coefficient': 0.0,
            'path_length': 0.0,
            'small_world_index': 0.0,
            'modularity': 0.0
        }
        
        try:
            # Create binary adjacency matrix
            adj_matrix = (np.abs(weight_matrix) > threshold).astype(int)
            
            # Skip if too sparse or too dense
            density = np.mean(adj_matrix)
            if density < 0.01 or density > 0.99:
                return metrics
            
            # Create networkx graph
            G = nx.from_numpy_array(adj_matrix)
            
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                # Clustering coefficient
                metrics['clustering_coefficient'] = nx.average_clustering(G)
                
                # Average path length (for largest connected component)
                if nx.is_connected(G):
                    metrics['path_length'] = nx.average_shortest_path_length(G)
                else:
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc)
                    if subgraph.number_of_nodes() > 1:
                        metrics['path_length'] = nx.average_shortest_path_length(subgraph)
                
                # Small-world index
                # Compare to random graph with same degree distribution
                try:
                    random_clustering = density  # Expected clustering for random graph
                    random_path_length = np.log(G.number_of_nodes()) / np.log(np.mean([d for n, d in G.degree()]))
                    
                    if random_clustering > 0 and random_path_length > 0:
                        clustering_ratio = metrics['clustering_coefficient'] / random_clustering
                        path_ratio = metrics['path_length'] / random_path_length
                        metrics['small_world_index'] = clustering_ratio / path_ratio
                except:
                    pass
                
                # Modularity (using community detection)
                try:
                    communities = nx.community.greedy_modularity_communities(G)
                    metrics['modularity'] = nx.community.modularity(G, communities)
                except:
                    pass
        
        except Exception as e:
            warnings.warn(f"Error computing graph metrics: {e}")
        
        return metrics
    
    def _analyze_inter_layer_connectivity(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze connectivity patterns between layers."""
        inter_layer = {
            'connection_matrix': None,
            'strongest_connections': [],
            'layer_coupling': []
        }
        
        if hasattr(model, 'inter_layer_weights'):
            weights = model.inter_layer_weights.detach().cpu().numpy()
            inter_layer['connection_matrix'] = weights.tolist()
            
            # Find strongest inter-layer connections
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    if i != j and abs(weights[i, j]) > 0.01:
                        inter_layer['strongest_connections'].append({
                            'from_layer': j,
                            'to_layer': i,
                            'strength': float(weights[i, j])
                        })
            
            # Compute layer coupling strength
            for i in range(weights.shape[0]):
                coupling_strength = np.sum(np.abs(weights[i, :])) + np.sum(np.abs(weights[:, i]))
                inter_layer['layer_coupling'].append({
                    'layer_id': i,
                    'total_coupling': float(coupling_strength),
                    'input_coupling': float(np.sum(np.abs(weights[i, :]))),
                    'output_coupling': float(np.sum(np.abs(weights[:, i])))
                })
        
        return inter_layer
    
    def _identify_connectivity_hubs(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Identify highly connected neurons (hubs) in the network."""
        hubs = {
            'layer_hubs': [],
            'global_hubs': [],
            'hub_distribution': {}
        }
        
        if hasattr(model, 'drn_layers'):
            all_connectivities = []
            
            for i, layer in enumerate(model.drn_layers):
                layer_connectivities = self._compute_layer_node_connectivities(layer)
                all_connectivities.extend(layer_connectivities)
                
                # Identify layer-level hubs (top 10% most connected)
                if layer_connectivities:
                    threshold = np.percentile(layer_connectivities, 90)
                    layer_hubs = [j for j, conn in enumerate(layer_connectivities) if conn >= threshold]
                    hubs['layer_hubs'].append({
                        'layer_id': i,
                        'hub_indices': layer_hubs,
                        'hub_connectivities': [layer_connectivities[j] for j in layer_hubs],
                        'mean_connectivity': float(np.mean(layer_connectivities)),
                        'connectivity_std': float(np.std(layer_connectivities))
                    })
            
            # Global hub analysis
            if all_connectivities:
                global_threshold = np.percentile(all_connectivities, 95)
                hubs['global_threshold'] = float(global_threshold)
                hubs['connectivity_stats'] = {
                    'mean': float(np.mean(all_connectivities)),
                    'std': float(np.std(all_connectivities)),
                    'max': float(np.max(all_connectivities)),
                    'gini_coefficient': self._compute_gini_coefficient(all_connectivities)
                }
        
        return hubs
    
    def _compute_layer_node_connectivities(self, layer: torch.nn.Module) -> List[float]:
        """Compute connectivity (degree) for each node in a layer."""
        connectivities = []
        
        # Extract weight matrices and compute node degrees
        if hasattr(layer, 'neuron_weights'):
            weights = layer.neuron_weights.detach().cpu().numpy()
            threshold = 0.01 * np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 0
            
            # Sum of connections for each neuron
            for i in range(weights.shape[0]):
                in_connections = np.sum(np.abs(weights[i, :]) > threshold)
                out_connections = np.sum(np.abs(weights[:, i]) > threshold)
                total_connections = in_connections + out_connections
                connectivities.append(float(total_connections))
        
        return connectivities
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient to measure inequality in connectivity distribution."""
        if not values:
            return 0.0
        
        values = np.array(values)
        values = np.sort(values)
        n = len(values)
        
        if n == 0 or np.sum(values) == 0:
            return 0.0
        
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
        return float(gini)
    
    def _detect_connectivity_motifs(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Detect common connectivity motifs in the network."""
        motifs = {
            'feedforward_chains': [],
            'feedback_loops': [],
            'convergent_patterns': [],
            'divergent_patterns': []
        }
        
        # This is a simplified motif detection - could be expanded
        if hasattr(model, 'inter_layer_weights'):
            weights = model.inter_layer_weights.detach().cpu().numpy()
            
            # Detect feedforward chains
            for i in range(weights.shape[0] - 1):
                if weights[i + 1, i] > 0.05:  # Strong feedforward connection
                    motifs['feedforward_chains'].append({
                        'from_layer': i,
                        'to_layer': i + 1,
                        'strength': float(weights[i + 1, i])
                    })
            
            # Detect feedback loops
            for i in range(weights.shape[0]):
                for j in range(i + 2, weights.shape[1]):  # Skip adjacent layers
                    if weights[i, j] > 0.05:  # Feedback connection
                        motifs['feedback_loops'].append({
                            'from_layer': j,
                            'to_layer': i,
                            'strength': float(weights[i, j]),
                            'skip_distance': j - i
                        })
        
        return motifs
    
    def _compute_topology_metrics(
        self, 
        model: torch.nn.Module, 
        network_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute high-level network topology metrics."""
        topology = {
            'network_density': 0.0,
            'modularity_index': 0.0,
            'hierarchical_organization': 0.0,
            'information_flow_efficiency': 0.0,
            'robustness_measures': {}
        }
        
        # Network density
        total_possible_connections = 0
        total_active_connections = 0
        
        if hasattr(model, 'drn_layers'):
            for layer in model.drn_layers:
                if hasattr(layer, 'neuron_pool_size'):
                    layer_size = layer.neuron_pool_size
                    total_possible_connections += layer_size * layer_size
                    
                    if hasattr(layer, 'neuron_weights'):
                        weights = layer.neuron_weights.detach().cpu().numpy()
                        threshold = 0.01 * np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 0
                        active = np.sum(np.abs(weights) > threshold)
                        total_active_connections += active
        
        if total_possible_connections > 0:
            topology['network_density'] = total_active_connections / total_possible_connections
        
        # Hierarchical organization (based on layer recruitment patterns)
        if 'layer_infos' in network_info:
            layer_activities = [info.get('num_recruited', 0) for info in network_info['layer_infos']]
            if len(layer_activities) > 1:
                # Measure how much activity decreases through layers (hierarchical processing)
                activity_gradient = np.diff(layer_activities)
                topology['hierarchical_organization'] = float(np.mean(activity_gradient))
        
        # Information flow efficiency (simplified measure)
        topology['information_flow_efficiency'] = network_info.get('recruitment_efficiency', 0)
        
        return topology
    
    def _analyze_population_dynamics(self, network_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dynamics between different populations."""
        dynamics = {
            'population_competition': [],
            'population_cooperation': [],
            'specialization_index': 0.0,
            'dynamic_balance': 0.0
        }
        
        if 'layer_infos' in network_info:
            for layer_info in network_info['layer_infos']:
                if 'population_infos' in layer_info:
                    pop_infos = layer_info['population_infos']
                    
                    # Population competition analysis
                    activities = [pop.get('mean_activity', 0) for pop in pop_infos]
                    if activities:
                        competition_index = np.std(activities) / (np.mean(activities) + 1e-8)
                        dynamics['population_competition'].append(float(competition_index))
                    
                    # Population cooperation (correlation in activities)
                    if len(activities) > 1:
                        activity_matrix = np.array(activities).reshape(1, -1)
                        cooperation_index = np.mean(np.corrcoef(activity_matrix)[0, 1:])
                        dynamics['population_cooperation'].append(float(cooperation_index))
        
        # Overall specialization index
        if dynamics['population_competition']:
            dynamics['specialization_index'] = float(np.mean(dynamics['population_competition']))
        
        return dynamics
    
    def _analyze_temporal_patterns(
        self, 
        current_analysis: Dict[str, Any], 
        step: int
    ) -> Dict[str, Any]:
        """Analyze how connectivity patterns change over time."""
        temporal = {
            'recruitment_stability': 0.0,
            'pattern_evolution': {},
            'adaptation_rate': 0.0,
            'memory_effects': {}
        }
        
        if len(self.connectivity_history) > 1:
            prev_analysis = self.connectivity_history[-2]
            
            # Recruitment stability
            current_recruitment = current_analysis['recruitment']['total_recruited']
            prev_recruitment = prev_analysis['recruitment']['total_recruited']
            
            if prev_recruitment > 0:
                stability = 1.0 - abs(current_recruitment - prev_recruitment) / prev_recruitment
                temporal['recruitment_stability'] = float(max(0, stability))
            
            # Pattern evolution (compare sparsity distributions)
            current_sparsity = current_analysis['patterns']['sparsity_distribution']
            prev_sparsity = prev_analysis['patterns']['sparsity_distribution']
            
            if current_sparsity and prev_sparsity and len(current_sparsity) == len(prev_sparsity):
                sparsity_changes = [abs(c - p) for c, p in zip(current_sparsity, prev_sparsity)]
                temporal['pattern_evolution'] = {
                    'mean_change': float(np.mean(sparsity_changes)),
                    'max_change': float(np.max(sparsity_changes)),
                    'layer_changes': sparsity_changes
                }
        
        # Adaptation rate (based on recent history)
        if len(self.recruitment_history) >= 5:
            recent_recruitments = [h['total_recruited'] for h in list(self.recruitment_history)[-5:]]
            temporal['adaptation_rate'] = float(np.std(recent_recruitments))
        
        return temporal
    
    def _update_running_metrics(self, connectivity_analysis: Dict[str, Any]):
        """Update running metrics for trend analysis."""
        recruitment = connectivity_analysis['recruitment']
        patterns = connectivity_analysis['patterns']
        
        # Update trends
        self.running_metrics['recruitment_efficiency'].append(
            recruitment.get('recruitment_efficiency', 0)
        )
        
        if patterns['sparsity_distribution']:
            mean_sparsity = np.mean(patterns['sparsity_distribution'])
            self.running_metrics['sparsity_trend'].append(mean_sparsity)
        
        if 'temporal' in connectivity_analysis:
            self.running_metrics['pattern_stability'].append(
                connectivity_analysis['temporal'].get('recruitment_stability', 0)
            )
    
    def _perform_expensive_analysis(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        """Perform computationally expensive analyses periodically."""
        expensive = {
            'clustering_analysis': {},
            'dimensionality_analysis': {},
            'critical_nodes': {},
            'network_comparison': {}
        }
        
        # Clustering analysis of recruitment patterns
        if len(self.recruitment_history) >= 10:
            expensive['clustering_analysis'] = self._analyze_recruitment_clustering()
        
        # Dimensionality analysis of connectivity space
        expensive['dimensionality_analysis'] = self._analyze_connectivity_dimensionality(model)
        
        # Critical node analysis
        expensive['critical_nodes'] = self._identify_critical_nodes(model)
        
        return expensive
    
    def _analyze_recruitment_clustering(self) -> Dict[str, Any]:
        """Cluster recruitment patterns to identify distinct modes."""
        clustering = {
            'num_clusters': 0,
            'cluster_labels': [],
            'cluster_characteristics': [],
            'silhouette_score': 0.0
        }
        
        # Extract recruitment vectors
        recruitment_vectors = []
        for hist in self.recruitment_history:
            if hist['layer_recruitment']:
                vector = [lr['neurons_recruited'] for lr in hist['layer_recruitment']]
                recruitment_vectors.append(vector)
        
        if len(recruitment_vectors) < 5:
            return clustering
        
        recruitment_matrix = np.array(recruitment_vectors)
        
        # Determine optimal number of clusters
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, min(8, len(recruitment_vectors) // 2)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(recruitment_matrix)
                score = silhouette_score(recruitment_matrix, labels)
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except:
                continue
        
        # Perform final clustering
        try:
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            labels = kmeans.fit_predict(recruitment_matrix)
            
            clustering['num_clusters'] = best_n_clusters
            clustering['cluster_labels'] = labels.tolist()
            clustering['silhouette_score'] = float(best_score)
            
            # Characterize each cluster
            for i in range(best_n_clusters):
                cluster_mask = labels == i
                cluster_data = recruitment_matrix[cluster_mask]
                
                clustering['cluster_characteristics'].append({
                    'cluster_id': i,
                    'size': int(np.sum(cluster_mask)),
                    'mean_recruitment': cluster_data.mean(axis=0).tolist(),
                    'std_recruitment': cluster_data.std(axis=0).tolist(),
                    'total_recruitment': float(cluster_data.sum(axis=1).mean())
                })
        
        except Exception as e:
            warnings.warn(f"Clustering analysis failed: {e}")
        
        return clustering
    
    def _analyze_connectivity_dimensionality(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze the dimensionality of the connectivity space."""
        dimensionality = {
            'effective_dimension': 0,
            'explained_variance_ratio': [],
            'intrinsic_dimension_estimate': 0
        }
        
        # Extract connectivity patterns
        connectivity_vectors = []
        if hasattr(model, 'drn_layers'):
            for layer in model.drn_layers:
                if hasattr(layer, 'neuron_weights'):
                    weights = layer.neuron_weights.detach().cpu().numpy().flatten()
                    connectivity_vectors.append(weights)
        
        if len(connectivity_vectors) < 2:
            return dimensionality
        
        connectivity_matrix = np.array(connectivity_vectors)
        
        try:
            # PCA analysis
            pca = PCA()
            pca.fit(connectivity_matrix)
            
            # Find effective dimensionality (95% variance explained)
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = np.argmax(cumsum_variance >= 0.95) + 1
            
            dimensionality['effective_dimension'] = int(effective_dim)
            dimensionality['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            
            # Estimate intrinsic dimension using participation ratio
            eigenvalues = pca.explained_variance_
            participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            dimensionality['intrinsic_dimension_estimate'] = float(participation_ratio)
        
        except Exception as e:
            warnings.warn(f"Dimensionality analysis failed: {e}")
        
        return dimensionality
    
    def _identify_critical_nodes(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Identify critical nodes whose removal would significantly impact network function."""
        critical_nodes = {
            'high_betweenness_nodes': [],
            'high_degree_nodes': [],
            'bottleneck_nodes': []
        }
        
        # This would require more sophisticated graph analysis
        # For now, return placeholder structure
        return critical_nodes
    
    def get_connectivity_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of connectivity analysis."""
        if not self.connectivity_history:
            return {'error': 'No connectivity data available'}
        
        latest = self.connectivity_history[-1]
        
        summary = {
            'current_state': latest,
            'trends': {
                'sparsity_trend': list(self.running_metrics['sparsity_trend']),
                'recruitment_efficiency_trend': list(self.running_metrics['recruitment_efficiency']),
                'pattern_stability_trend': list(self.running_metrics['pattern_stability'])
            },
            'statistics': {
                'total_analyses': len(self.connectivity_history),
                'mean_recruitment': np.mean([h['recruitment']['total_recruited'] 
                                           for h in self.connectivity_history]),
                'recruitment_variance': np.var([h['recruitment']['total_recruited'] 
                                              for h in self.connectivity_history])
            }
        }
        
        # Add historical patterns if available
        if len(self.connectivity_history) > 10:
            summary['historical_patterns'] = self._compute_historical_patterns()
        
        return summary
    
    def _compute_historical_patterns(self) -> Dict[str, Any]:
        """Compute patterns across the full history."""
        patterns = {
            'recruitment_cycles': [],
            'connectivity_phases': [],
            'adaptation_periods': []
        }
        
        # Extract time series
        recruitments = [h['recruitment']['total_recruited'] for h in self.connectivity_history]
        
        # Simple cycle detection using autocorrelation
        if len(recruitments) > 20:
            autocorr = np.correlate(recruitments, recruitments, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation (potential cycles)
            peaks = []
            for i in range(2, len(autocorr) - 2):
                if (autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and 
                    autocorr[i] > 0.1 * np.max(autocorr)):
                    peaks.append(i)
            
            patterns['recruitment_cycles'] = peaks
        
        return patterns
    
    def export_analysis(self, filepath: str, format: str = 'json'):
        """Export connectivity analysis results."""
        summary = self.get_connectivity_summary()
        
        if format.lower() == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            import pandas as pd
            
            # Convert to flat format for CSV
            flat_data = []
            for i, conn in enumerate(self.connectivity_history):
                row = {
                    'step': i,
                    'total_recruited': conn['recruitment']['total_recruited'],
                    'recruitment_efficiency': conn['recruitment']['recruitment_efficiency'],
                }
                flat_data.append(row)
            
            df = pd.DataFrame(flat_data)
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


def create_connectivity_analyzer(
    track_history: bool = True,
    analysis_frequency: int = 10
) -> ConnectivityAnalyzer:
    """
    Factory function to create a connectivity analyzer.
    
    Args:
        track_history: Whether to maintain analysis history
        analysis_frequency: How often to perform expensive analyses
        
    Returns:
        Configured ConnectivityAnalyzer instance
    """
    return ConnectivityAnalyzer(
        track_history=track_history,
        analysis_frequency=analysis_frequency
    )


def quick_connectivity_analysis(
    model: torch.nn.Module,
    test_inputs: torch.Tensor,
    num_steps: int = 50
) -> Dict[str, Any]:
    """
    Quick connectivity analysis for a model on test inputs.
    
    Args:
        model: DRN model to analyze
        test_inputs: Test input data
        num_steps: Number of analysis steps
        
    Returns:
        Connectivity analysis summary
    """
    analyzer = create_connectivity_analyzer()
    
    model.eval()
    with torch.no_grad():
        for step in range(num_steps):
            # Random batch from test inputs
            batch_idx = torch.randint(0, len(test_inputs), (1,))
            x = test_inputs[batch_idx]
            
            # Forward pass
            output, network_info = model(x, return_layer_info=True)
            
            # Analyze connectivity
            analyzer.analyze_network_step(model, network_info, x, step)
    
    return analyzer.get_connectivity_summary()