"""
drn/analysis/visualization.py

Visualization tools for Dynamic Recruitment Networks.
These tools help analyze and understand the unique connectivity patterns
and dynamics of DRN models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from collections import defaultdict
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some interactive visualizations will be disabled.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Network graph visualizations will be disabled.")


class DRNVisualizer:
    """
    Comprehensive visualization toolkit for Dynamic Recruitment Networks.
    
    Provides static and interactive visualizations for:
    - Recruitment patterns over time
    - Budget usage and efficiency
    - Population dynamics
    - Connectivity graphs
    - Training progress
    - Flexibility metrics comparison
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set style
        if style in plt.style.available:
            plt.style.use(style)
        
        # Color palettes for different visualizations
        self.colors = {
            'recruitment': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            'budget': ['#0D7377', '#14A085', '#A7E6D7', '#FFF4E6'],
            'layers': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
            'metrics': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        }
    
    def plot_recruitment_timeline(
        self, 
        network_infos: List[Dict[str, Any]], 
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot recruitment patterns over time.
        
        Args:
            network_infos: List of network info dictionaries from forward passes
            save_path: Path to save the plot
            interactive: Whether to create interactive plot (requires plotly)
            
        Returns:
            Matplotlib figure if not interactive, None otherwise
        """
        if not network_infos:
            print("No network info available for plotting")
            return None
        
        # Extract data
        time_steps = list(range(len(network_infos)))
        total_recruited = [info.get('total_neurons_recruited', 0) for info in network_infos]
        sparsity = [info.get('network_sparsity', 0.0) for info in network_infos]
        
        # Extract per-layer recruitment
        layer_recruitment = defaultdict(list)
        max_layers = 0
        
        for info in network_infos:
            layer_infos = info.get('layer_infos', [])
            max_layers = max(max_layers, len(layer_infos))
            
            for i, layer_info in enumerate(layer_infos):
                layer_recruitment[i].append(layer_info.get('num_recruited', 0))
            
            # Pad with zeros for missing layers
            for i in range(len(layer_infos), max_layers):
                layer_recruitment[i].append(0)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_recruitment_timeline_interactive(
                time_steps, total_recruited, sparsity, layer_recruitment, save_path
            )
        else:
            return self._plot_recruitment_timeline_static(
                time_steps, total_recruited, sparsity, layer_recruitment, save_path
            )
    
    def _plot_recruitment_timeline_static(
        self, 
        time_steps: List[int], 
        total_recruited: List[int],
        sparsity: List[float],
        layer_recruitment: Dict[int, List[int]],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create static recruitment timeline plot."""
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Plot 1: Total recruitment over time
        axes[0].plot(time_steps, total_recruited, 'o-', color=self.colors['recruitment'][0], linewidth=2)
        axes[0].set_title('Total Neurons Recruited Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Neurons Recruited')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Sparsity over time
        axes[1].plot(time_steps, sparsity, 's-', color=self.colors['recruitment'][1], linewidth=2)
        axes[1].set_title('Network Sparsity Over Time', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Sparsity')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Per-layer recruitment
        for layer_idx, recruitments in layer_recruitment.items():
            if recruitments and max(recruitments) > 0:  # Only plot layers with activity
                color = self.colors['layers'][layer_idx % len(self.colors['layers'])]
                axes[2].plot(time_steps, recruitments, 'o-', label=f'Layer {layer_idx}', 
                           color=color, linewidth=2)
        
        axes[2].set_title('Per-Layer Recruitment Over Time', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Neurons Recruited')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Recruitment timeline saved to {save_path}")
        
        return fig
    
    def _plot_recruitment_timeline_interactive(
        self,
        time_steps: List[int],
        total_recruited: List[int],
        sparsity: List[float],
        layer_recruitment: Dict[int, List[int]],
        save_path: Optional[str]
    ):
        """Create interactive recruitment timeline plot."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Total Neurons Recruited', 'Network Sparsity', 'Per-Layer Recruitment'),
            vertical_spacing=0.1
        )
        
        # Total recruitment
        fig.add_trace(
            go.Scatter(x=time_steps, y=total_recruited, mode='lines+markers', 
                      name='Total Recruited', line=dict(color=self.colors['recruitment'][0])),
            row=1, col=1
        )
        
        # Sparsity
        fig.add_trace(
            go.Scatter(x=time_steps, y=sparsity, mode='lines+markers',
                      name='Sparsity', line=dict(color=self.colors['recruitment'][1])),
            row=2, col=1
        )
        
        # Per-layer recruitment
        for layer_idx, recruitments in layer_recruitment.items():
            if recruitments and max(recruitments) > 0:
                color = self.colors['layers'][layer_idx % len(self.colors['layers'])]
                fig.add_trace(
                    go.Scatter(x=time_steps, y=recruitments, mode='lines+markers',
                              name=f'Layer {layer_idx}', line=dict(color=color)),
                    row=3, col=1
                )
        
        fig.update_layout(height=800, title_text="DRN Recruitment Timeline")
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive recruitment timeline saved to {save_path}")
        
        fig.show()
        return None
    
    def plot_budget_analysis(
        self, 
        network_infos: List[Dict[str, Any]], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot budget usage and efficiency analysis.
        
        Args:
            network_infos: List of network info dictionaries
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not network_infos:
            print("No network info available for plotting")
            return None
        
        # Extract budget data
        time_steps = list(range(len(network_infos)))
        budget_spent = [info.get('total_budget_spent', 0.0) for info in network_infos]
        budget_remaining = [info.get('avg_budget_remaining', 1.0) for info in network_infos]
        efficiency = [info.get('recruitment_efficiency', 1.0) for info in network_infos]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        
        # Budget spent over time
        axes[0, 0].plot(time_steps, budget_spent, 'o-', color=self.colors['budget'][0], linewidth=2)
        axes[0, 0].set_title('Budget Spent Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Budget Spent')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Budget remaining over time
        axes[0, 1].plot(time_steps, budget_remaining, 's-', color=self.colors['budget'][1], linewidth=2)
        axes[0, 1].set_title('Average Budget Remaining', fontweight='bold')
        axes[0, 1].set_ylabel('Budget Remaining')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recruitment efficiency
        axes[1, 0].plot(time_steps, efficiency, '^-', color=self.colors['budget'][2], linewidth=2)
        axes[1, 0].set_title('Recruitment Efficiency', fontweight='bold')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Efficiency (Neurons/Budget)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Budget vs Efficiency scatter
        axes[1, 1].scatter(budget_spent, efficiency, c=time_steps, 
                          cmap='viridis', alpha=0.7, s=50)
        axes[1, 1].set_title('Budget vs Efficiency', fontweight='bold')
        axes[1, 1].set_xlabel('Budget Spent')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar for time
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Time Step')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Budget analysis saved to {save_path}")
        
        return fig
    
    def plot_connectivity_graph(
        self, 
        network_info: Dict[str, Any], 
        save_path: Optional[str] = None,
        layout: str = 'spring'
    ) -> Optional[plt.Figure]:
        """
        Plot network connectivity as a graph.
        
        Args:
            network_info: Single network info dictionary
            save_path: Path to save the plot
            layout: Graph layout ('spring', 'circular', 'kamada_kawai')
            
        Returns:
            Matplotlib figure if NetworkX available, None otherwise
        """
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available. Cannot create connectivity graph.")
            return None
        
        layer_infos = network_info.get('layer_infos', [])
        if not layer_infos:
            print("No layer info available for connectivity graph")
            return None
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        node_id = 0
        layer_nodes = {}
        
        for layer_idx, layer_info in enumerate(layer_infos):
            recruited_indices = layer_info.get('recruited_indices', [])
            layer_nodes[layer_idx] = []
            
            # Add population node
            pop_node = f"Pop_{layer_idx}"
            G.add_node(pop_node, node_type='population', layer=layer_idx)
            layer_nodes[layer_idx].append(pop_node)
            
            # Add recruited neuron nodes
            for neuron_idx in recruited_indices:
                neuron_node = f"N_{layer_idx}_{neuron_idx}"
                G.add_node(neuron_node, node_type='neuron', layer=layer_idx)
                layer_nodes[layer_idx].append(neuron_node)
                
                # Edge from population to neuron
                G.add_edge(pop_node, neuron_node, edge_type='recruitment')
            
            # Connect to next layer population
            if layer_idx < len(layer_infos) - 1:
                next_pop = f"Pop_{layer_idx + 1}"
                for neuron_node in layer_nodes[layer_idx][1:]:  # Skip population node
                    G.add_edge(neuron_node, next_pop, edge_type='forward')
        
        # Create layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw nodes by type
        pop_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'population']
        neuron_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'neuron']
        
        nx.draw_networkx_nodes(G, pos, nodelist=pop_nodes, node_color='red', 
                              node_size=300, alpha=0.8, ax=ax, label='Population')
        nx.draw_networkx_nodes(G, pos, nodelist=neuron_nodes, node_color='blue', 
                              node_size=100, alpha=0.6, ax=ax, label='Recruited Neurons')
        
        # Draw edges by type
        recruitment_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'recruitment']
        forward_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'forward']
        
        nx.draw_networkx_edges(G, pos, edgelist=recruitment_edges, edge_color='green', 
                              alpha=0.6, ax=ax, style='solid', width=2)
        nx.draw_networkx_edges(G, pos, edgelist=forward_edges, edge_color='gray', 
                              alpha=0.4, ax=ax, style='dashed', width=1)
        
        # Add labels for population nodes
        pop_labels = {n: n for n in pop_nodes}
        nx.draw_networkx_labels(G, pos, labels=pop_labels, font_size=8, ax=ax)
        
        ax.set_title('DRN Connectivity Graph', fontsize=16, fontweight='bold')
        ax.legend()
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Connectivity graph saved to {save_path}")
        
        return fig
    
    def plot_training_progress(
        self, 
        training_history: Dict[str, List[float]], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive training progress for DRN.
        
        Args:
            training_history: Dictionary with training metrics over time
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Extract epochs
        epochs = list(range(len(training_history.get('train_total_loss', []))))
        
        if not epochs:
            print("No training history available")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.2))
        
        # Loss curves
        if 'train_total_loss' in training_history and 'val_total_loss' in training_history:
            axes[0, 0].plot(epochs, training_history['train_total_loss'], 
                           label='Train', color=self.colors['metrics'][0], linewidth=2)
            axes[0, 0].plot(epochs, training_history['val_total_loss'], 
                           label='Validation', color=self.colors['metrics'][1], linewidth=2)
            axes[0, 0].set_title('Total Loss', fontweight='bold')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy (if available)
        if 'train_accuracy' in training_history and 'val_accuracy' in training_history:
            axes[0, 1].plot(epochs, training_history['train_accuracy'], 
                           label='Train', color=self.colors['metrics'][2], linewidth=2)
            axes[0, 1].plot(epochs, training_history['val_accuracy'], 
                           label='Validation', color=self.colors['metrics'][3], linewidth=2)
            axes[0, 1].set_title('Accuracy', fontweight='bold')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Recruitment metrics
        if 'val_total_recruited' in training_history:
            axes[0, 2].plot(epochs, training_history['val_total_recruited'], 
                           color=self.colors['recruitment'][0], linewidth=2)
            axes[0, 2].set_title('Neurons Recruited', fontweight='bold')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Network sparsity
        if 'val_network_sparsity' in training_history:
            axes[1, 0].plot(epochs, training_history['val_network_sparsity'], 
                           color=self.colors['recruitment'][1], linewidth=2)
            axes[1, 0].set_title('Network Sparsity', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Sparsity')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Flexibility metrics (if available)
        if 'val_overall_flexibility' in training_history:
            axes[1, 1].plot(epochs, training_history['val_overall_flexibility'], 
                           color=self.colors['metrics'][4], linewidth=2)
            axes[1, 1].set_title('Cognitive Flexibility', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Flexibility Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Recruitment efficiency
        if 'val_recruitment_efficiency' in training_history:
            axes[1, 2].plot(epochs, training_history['val_recruitment_efficiency'], 
                           color=self.colors['budget'][0], linewidth=2)
            axes[1, 2].set_title('Recruitment Efficiency', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Efficiency')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress saved to {save_path}")
        
        return fig
    
    def plot_flexibility_comparison(
        self, 
        drn_results: Dict[str, float], 
        baseline_results: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare flexibility metrics between DRN and baseline models.
        
        Args:
            drn_results: DRN model results
            baseline_results: Baseline model results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Find common metrics
        common_metrics = set(drn_results.keys()) & set(baseline_results.keys())
        flexibility_metrics = [m for m in common_metrics if 'flexibility' in m or 'entropy' in m or 'smoothness' in m]
        
        if not flexibility_metrics:
            print("No common flexibility metrics found")
            return None
        
        # Prepare data
        metrics = []
        drn_values = []
        baseline_values = []
        improvements = []
        
        for metric in flexibility_metrics:
            metrics.append(metric.replace('_', ' ').title())
            drn_val = drn_results[metric]
            baseline_val = baseline_results[metric]
            drn_values.append(drn_val)
            baseline_values.append(baseline_val)
            improvements.append(((drn_val - baseline_val) / baseline_val) * 100)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        
        # Comparison bar plot
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, drn_values, width, label='DRN', 
                       color=self.colors['metrics'][0], alpha=0.8)
        bars2 = ax1.bar(x + width/2, baseline_values, width, label='Baseline', 
                       color=self.colors['metrics'][1], alpha=0.8)
        
        ax1.set_title('Flexibility Metrics Comparison', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement plot
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
        ax2.set_title('DRN Improvement Over Baseline', fontweight='bold')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Flexibility comparison saved to {save_path}")
        
        return fig
    
    def create_dashboard(
        self, 
        network_infos: List[Dict[str, Any]],
        training_history: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            network_infos: List of network info dictionaries
            training_history: Optional training history
            save_path: Path to save the dashboard
            
        Returns:
            Matplotlib figure
        """
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Define grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Recruitment timeline (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        time_steps = list(range(len(network_infos)))
        total_recruited = [info.get('total_neurons_recruited', 0) for info in network_infos]
        ax1.plot(time_steps, total_recruited, 'o-', color=self.colors['recruitment'][0], linewidth=2)
        ax1.set_title('Recruitment Timeline', fontweight='bold')
        ax1.set_ylabel('Neurons Recruited')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sparsity over time (top row, right)
        ax2 = fig.add_subplot(gs[0, 2:])
        sparsity = [info.get('network_sparsity', 0.0) for info in network_infos]
        ax2.plot(time_steps, sparsity, 's-', color=self.colors['recruitment'][1], linewidth=2)
        ax2.set_title('Network Sparsity', fontweight='bold')
        ax2.set_ylabel('Sparsity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Budget analysis (middle row, left)
        ax3 = fig.add_subplot(gs[1, 0])
        budget_spent = [info.get('total_budget_spent', 0.0) for info in network_infos]
        ax3.plot(time_steps, budget_spent, '^-', color=self.colors['budget'][0], linewidth=2)
        ax3.set_title('Budget Spent', fontweight='bold')
        ax3.set_ylabel('Budget')
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency (middle row, center-left)
        ax4 = fig.add_subplot(gs[1, 1])
        efficiency = [info.get('recruitment_efficiency', 1.0) for info in network_infos]
        ax4.plot(time_steps, efficiency, 'D-', color=self.colors['budget'][1], linewidth=2)
        ax4.set_title('Efficiency', fontweight='bold')
        ax4.set_ylabel('Neurons/Budget')
        ax4.grid(True, alpha=0.3)
        
        # 5. Per-layer recruitment heatmap (middle row, center-right to right)
        ax5 = fig.add_subplot(gs[1, 2:])
        layer_data = []
        max_layers = 0
        for info in network_infos:
            layer_infos = info.get('layer_infos', [])
            max_layers = max(max_layers, len(layer_infos))
            layer_row = [layer_info.get('num_recruited', 0) for layer_info in layer_infos]
            # Pad with zeros
            while len(layer_row) < max_layers:
                layer_row.append(0)
            layer_data.append(layer_row)
        
        if layer_data and max_layers > 0:
            layer_array = np.array(layer_data).T  # Transpose for proper orientation
            im = ax5.imshow(layer_array, aspect='auto', cmap='viridis', interpolation='nearest')
            ax5.set_title('Per-Layer Recruitment', fontweight='bold')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Layer')
            ax5.set_yticks(range(max_layers))
            plt.colorbar(im, ax=ax5, label='Neurons Recruited')
        
        # 6. Training loss (bottom row, if available)
        if training_history and 'train_total_loss' in training_history:
            ax6 = fig.add_subplot(gs[2, :2])
            epochs = list(range(len(training_history['train_total_loss'])))
            ax6.plot(epochs, training_history['train_total_loss'], 
                    label='Train', color=self.colors['metrics'][0], linewidth=2)
            if 'val_total_loss' in training_history:
                ax6.plot(epochs, training_history['val_total_loss'], 
                        label='Val', color=self.colors['metrics'][1], linewidth=2)
            ax6.set_title('Training Loss', fontweight='bold')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Loss')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Flexibility metrics summary (bottom row, right)
        if len(network_infos) > 0:
            ax7 = fig.add_subplot(gs[2, 2:])
            
            # Compute some summary statistics
            final_info = network_infos[-1]
            metrics_names = ['Total\nRecruited', 'Sparsity', 'Efficiency']
            metrics_values = [
                final_info.get('total_neurons_recruited', 0),
                final_info.get('network_sparsity', 0.0),
                final_info.get('recruitment_efficiency', 1.0) / 10  # Scale for visibility
            ]
            
            bars = ax7.bar(metrics_names, metrics_values, 
                          color=self.colors['metrics'][:3], alpha=0.7)
            ax7.set_title('Final Metrics', fontweight='bold')
            ax7.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # Main title
        fig.suptitle('Dynamic Recruitment Network - Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        return fig


# Utility functions for quick plotting
def quick_plot_recruitment(network_infos: List[Dict[str, Any]], save_path: Optional[str] = None):
    """Quick function to plot recruitment timeline."""
    visualizer = DRNVisualizer()
    return visualizer.plot_recruitment_timeline(network_infos, save_path)


def quick_plot_training(training_history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Quick function to plot training progress."""
    visualizer = DRNVisualizer()
    return visualizer.plot_training_progress(training_history, save_path)


def quick_dashboard(network_infos: List[Dict[str, Any]], 
                   training_history: Optional[Dict[str, List[float]]] = None,
                   save_path: Optional[str] = None):
    """Quick function to create analysis dashboard."""
    visualizer = DRNVisualizer()
    return visualizer.create_dashboard(network_infos, training_history, save_path)


# Example usage and testing
if __name__ == "__main__":
    print("Testing DRN visualization tools...")
    
    # Create mock data
    network_infos = []
    for i in range(50):
        info = {
            'total_neurons_recruited': 10 + int(5 * np.sin(i/5)) + np.random.randint(-2, 3),
            'network_sparsity': 0.7 + 0.2 * np.sin(i/10) + np.random.normal(0, 0.05),
            'recruitment_efficiency': 8 + 2 * np.cos(i/7) + np.random.normal(0, 0.5),
            'total_budget_spent': 1.0 + 0.3 * np.sin(i/8) + np.random.normal(0, 0.1),
            'avg_budget_remaining': max(0.1, 0.8 - 0.01 * i + np.random.normal(0, 0.05)),
            'layer_infos': [
                {
                    'num_recruited': 5 + np.random.randint(-1, 2),
                    'recruited_indices': list(range(5 + np.random.randint(-1, 2)))
                },
                {
                    'num_recruited': 3 + np.random.randint(-1, 2),
                    'recruited_indices': list(range(3 + np.random.randint(-1, 2)))
                }
            ]
        }
        network_infos.append(info)
    
    # Mock training history
    epochs = 25
    training_history = {
        'train_total_loss': [2.0 * np.exp(-i/10) + np.random.normal(0, 0.1) for i in range(epochs)],
        'val_total_loss': [2.2 * np.exp(-i/10) + np.random.normal(0, 0.15) for i in range(epochs)],
        'train_accuracy': [0.5 + 0.4 * (1 - np.exp(-i/5)) + np.random.normal(0, 0.02) for i in range(epochs)],
        'val_accuracy': [0.45 + 0.4 * (1 - np.exp(-i/5)) + np.random.normal(0, 0.03) for i in range(epochs)],
        'val_total_recruited': [12 + 3 * np.sin(i/3) + np.random.normal(0, 1) for i in range(epochs)],
        'val_network_sparsity': [0.75 + 0.1 * np.sin(i/4) + np.random.normal(0, 0.02) for i in range(epochs)],
        'val_recruitment_efficiency': [8 + 2 * np.cos(i/5) + np.random.normal(0, 0.3) for i in range(epochs)],
    }
    
    # Create visualizer
    visualizer = DRNVisualizer()
    
    # Test recruitment timeline
    print("Creating recruitment timeline...")
    fig1 = visualizer.plot_recruitment_timeline(network_infos)
    plt.show()
    
    # Test budget analysis
    print("Creating budget analysis...")
    fig2 = visualizer.plot_budget_analysis(network_infos)
    plt.show()
    
    # Test training progress
    print("Creating training progress plot...")
    fig3 = visualizer.plot_training_progress(training_history)
    plt.show()
    
    # Test dashboard
    print("Creating dashboard...")
    fig4 = visualizer.create_dashboard(network_infos, training_history)
    plt.show()
    
    # Test connectivity graph (if NetworkX available)
    if NETWORKX_AVAILABLE:
        print("Creating connectivity graph...")
        fig5 = visualizer.plot_connectivity_graph(network_infos[0])
        if fig5:
            plt.show()
    
    print("DRN visualization tests completed!")