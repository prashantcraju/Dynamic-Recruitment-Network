"""
drn/models/baseline_models.py

Baseline neural network models for comparison with Dynamic Recruitment Networks.
These models provide standard architectures to benchmark DRN performance against
traditional approaches in concept learning and cognitive flexibility tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
import math


class BaselineModel(nn.Module):
    """
    Base class for all baseline models to ensure consistent interface with DRN.
    
    All baseline models should inherit from this class and implement the forward
    method with the same signature as DRNNetwork for fair comparison.
    """
    
    def __init__(self, input_size: int, output_size: int):
        super(BaselineModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._forward_count = 0
        self._info_history = []
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_layer_info: bool = True,
        reset_budgets: bool = False  # For interface compatibility
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass. Must be implemented by subclasses.
        Should return same format as DRNNetwork for fair comparison.
        """
        raise NotImplementedError
    
    def reset_all_budgets(self):
        """Compatibility method for DRN interface."""
        pass
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics for analysis."""
        if not self._info_history:
            return {'error': 'No forward passes recorded yet'}
        return {
            'forward_count': self._forward_count,
            'current_state': self._info_history[-1] if self._info_history else {},
            'history_length': len(self._info_history)
        }


class StandardMLP(BaselineModel):
    """
    Standard Multi-Layer Perceptron for comparison with DRN.
    
    This represents the traditional feedforward neural network approach
    with fixed connectivity and no dynamic recruitment mechanisms.
    
    Args:
        input_size (int): Input feature dimension
        hidden_sizes (List[int]): Hidden layer sizes
        output_size (int): Output dimension
        activation (str): Activation function ('relu', 'tanh', 'sigmoid')
        dropout_rate (float): Dropout probability
        batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        batch_norm: bool = False
    ):
        super(StandardMLP, self).__init__(input_size, output_size)
        
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        if activation not in ['relu', 'tanh', 'sigmoid', 'gelu']:
            raise ValueError("Unsupported activation function")
        
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Batch normalization (not for output layer)
            if batch_norm and i < len(layer_sizes) - 2:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            
            # Dropout (not for output layer)
            if dropout_rate > 0 and i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Get activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = F.gelu
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_layer_info: bool = True,
        reset_budgets: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass through MLP."""
        self._forward_count += 1
        
        current = x
        layer_activations = []
        layer_stats = []
        
        # Forward through hidden layers
        for i in range(len(self.layers) - 1):
            # Linear transformation
            current = self.layers[i](current)
            
            # Batch normalization
            if self.batch_norm:
                current = self.batch_norms[i](current)
            
            # Activation
            current = self.activation(current)
            
            # Dropout
            if self.dropout_rate > 0:
                current = self.dropouts[i](current)
            
            layer_activations.append(current)
            
            # Collect layer statistics
            if return_layer_info:
                layer_stats.append({
                    'mean_activation': current.mean().item(),
                    'activation_std': current.std().item(),
                    'sparsity': (current == 0).float().mean().item(),
                    'max_activation': current.max().item()
                })
        
        # Final output layer (no activation)
        output = self.layers[-1](current)
        
        if not return_layer_info:
            return output
        
        # Compile network info
        network_info = {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'num_layers': len(self.layers),
            'forward_count': self._forward_count,
            'layer_stats': layer_stats,
            'output_magnitude': torch.norm(output).item(),
            'architecture': 'StandardMLP',
            'hidden_sizes': self.hidden_sizes,
            
            # For compatibility with DRN analysis
            'total_neurons_recruited': sum(h for h in self.hidden_sizes),
            'network_sparsity': 0.0,  # No sparsity in standard MLP
            'recruitment_efficiency': 1.0,  # All neurons always active
        }
        
        self._info_history.append(network_info)
        if len(self._info_history) > 500:
            self._info_history.pop(0)
        
        return output, network_info


class RecurrentBaseline(BaselineModel):
    """
    Recurrent Neural Network baseline (LSTM/GRU) for sequence processing.
    
    This model processes inputs sequentially and maintains hidden state,
    providing a baseline for temporal processing capabilities.
    
    Args:
        input_size (int): Input feature dimension
        hidden_size (int): Hidden state dimension
        num_layers (int): Number of recurrent layers
        output_size (int): Output dimension
        rnn_type (str): Type of RNN ('lstm', 'gru', 'rnn')
        dropout (float): Dropout between layers
        bidirectional (bool): Whether to use bidirectional RNN
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        rnn_type: str = 'lstm',
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super(RecurrentBaseline, self).__init__(input_size, output_size)
        
        if rnn_type not in ['lstm', 'gru', 'rnn']:
            raise ValueError("rnn_type must be 'lstm', 'gru', or 'rnn'")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # Create RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                dropout=dropout, bidirectional=bidirectional, batch_first=True
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                dropout=dropout, bidirectional=bidirectional, batch_first=True
            )
        else:  # rnn
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers,
                dropout=dropout, bidirectional=bidirectional, batch_first=True
            )
        
        # Output projection
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(rnn_output_size, output_size)
        
        # State tracking
        self.hidden_state = None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize RNN weights."""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def reset_hidden_state(self):
        """Reset hidden state for new sequences."""
        self.hidden_state = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_layer_info: bool = True,
        reset_budgets: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass through RNN."""
        self._forward_count += 1
        
        if reset_budgets:
            self.reset_hidden_state()
        
        batch_size = x.shape[0]
        
        # Treat input as sequence of length 1 if 2D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        
        # Forward through RNN
        rnn_output, self.hidden_state = self.rnn(x, self.hidden_state)
        
        # Use last output for classification
        last_output = rnn_output[:, -1, :]  # [batch_size, hidden_size]
        
        # Project to output size
        output = self.output_projection(last_output)
        
        if not return_layer_info:
            return output
        
        # Compile network info
        network_info = {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'num_layers': self.num_layers,
            'forward_count': self._forward_count,
            'hidden_state_norm': torch.norm(self.hidden_state[0]).item() if self.hidden_state else 0.0,
            'rnn_output_norm': torch.norm(rnn_output).item(),
            'output_magnitude': torch.norm(output).item(),
            'architecture': f'RecurrentBaseline_{self.rnn_type.upper()}',
            'hidden_size': self.hidden_size,
            'bidirectional': self.bidirectional,
            
            # For compatibility with DRN analysis
            'total_neurons_recruited': self.hidden_size * self.num_layers,
            'network_sparsity': 0.0,
            'recruitment_efficiency': 1.0,
        }
        
        self._info_history.append(network_info)
        if len(self._info_history) > 500:
            self._info_history.pop(0)
        
        return output, network_info


class TransformerBaseline(BaselineModel):
    """
    Transformer-based baseline for attention-based processing.
    
    This model uses self-attention mechanisms to process inputs,
    providing a modern baseline for comparison with DRN's
    dynamic recruitment mechanisms.
    
    Args:
        input_size (int): Input feature dimension
        model_dim (int): Model dimension (d_model)
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        output_size (int): Output dimension
        ff_dim (int): Feedforward dimension
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        output_size: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super(TransformerBaseline, self).__init__(input_size, output_size)
        
        if ff_dim is None:
            ff_dim = model_dim * 4
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_size, model_dim)
        
        # Positional encoding (for sequence processing)
        self.pos_encoding = PositionalEncoding(model_dim, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(model_dim, output_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize transformer weights."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_layer_info: bool = True,
        reset_budgets: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass through Transformer."""
        self._forward_count += 1
        
        batch_size = x.shape[0]
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Treat as sequence of length 1 if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, model_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Forward through transformer
        transformer_output = self.transformer(x)
        
        # Use last token for classification (or mean pooling)
        if transformer_output.shape[1] == 1:
            pooled_output = transformer_output.squeeze(1)
        else:
            pooled_output = transformer_output.mean(dim=1)  # Mean pooling
        
        # Project to output size
        output = self.output_projection(pooled_output)
        
        if not return_layer_info:
            return output
        
        # Compile network info
        network_info = {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'num_layers': self.num_layers,
            'forward_count': self._forward_count,
            'attention_output_norm': torch.norm(transformer_output).item(),
            'output_magnitude': torch.norm(output).item(),
            'architecture': 'TransformerBaseline',
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            
            # For compatibility with DRN analysis
            'total_neurons_recruited': self.model_dim * self.num_layers,
            'network_sparsity': 0.0,
            'recruitment_efficiency': 1.0,
        }
        
        self._info_history.append(network_info)
        if len(self._info_history) > 500:
            self._info_history.pop(0)
        
        return output, network_info


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, model_dim: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * 
                           (-math.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class SparseBaseline(BaselineModel):
    """
    Sparse neural network baseline with fixed sparsity patterns.
    
    This model implements fixed sparsity (like pruned networks) to compare
    with DRN's dynamic sparsity. It maintains a constant sparse connectivity
    pattern throughout training and inference.
    
    Args:
        input_size (int): Input feature dimension
        hidden_sizes (List[int]): Hidden layer sizes
        output_size (int): Output dimension
        sparsity_level (float): Fraction of weights to keep (0.1 = 10% weights active)
        sparsity_type (str): Type of sparsity ('random', 'structured')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        sparsity_level: float = 0.1,
        sparsity_type: str = 'random'
    ):
        super(SparseBaseline, self).__init__(input_size, output_size)
        
        if not 0.0 < sparsity_level <= 1.0:
            raise ValueError("sparsity_level must be between 0 and 1")
        if sparsity_type not in ['random', 'structured']:
            raise ValueError("sparsity_type must be 'random' or 'structured'")
        
        self.hidden_sizes = hidden_sizes
        self.sparsity_level = sparsity_level
        self.sparsity_type = sparsity_type
        
        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        self.sparse_masks = []
        
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            
            # Create sparsity mask
            if i < len(layer_sizes) - 2:  # Don't sparsify output layer
                mask = self._create_sparsity_mask(
                    layer_sizes[i], layer_sizes[i + 1], sparsity_level, sparsity_type
                )
                self.sparse_masks.append(mask)
            else:
                # Dense output layer
                mask = torch.ones(layer_sizes[i + 1], layer_sizes[i])
                self.sparse_masks.append(mask)
        
        # Register masks as buffers
        for i, mask in enumerate(self.sparse_masks):
            self.register_buffer(f'mask_{i}', mask)
        
        self._initialize_weights()
        self._apply_sparsity_masks()
    
    def _create_sparsity_mask(
        self, 
        input_size: int, 
        output_size: int, 
        sparsity_level: float,
        sparsity_type: str
    ) -> torch.Tensor:
        """Create sparsity mask for a layer."""
        mask = torch.zeros(output_size, input_size)
        
        if sparsity_type == 'random':
            # Random sparsity
            total_weights = output_size * input_size
            num_active = int(total_weights * sparsity_level)
            
            # Randomly select weights to keep active
            flat_mask = torch.zeros(total_weights)
            indices = torch.randperm(total_weights)[:num_active]
            flat_mask[indices] = 1.0
            mask = flat_mask.view(output_size, input_size)
            
        elif sparsity_type == 'structured':
            # Structured sparsity - each output neuron connects to a subset of inputs
            inputs_per_output = max(1, int(input_size * sparsity_level))
            
            for i in range(output_size):
                # Each output connects to random subset of inputs
                indices = torch.randperm(input_size)[:inputs_per_output]
                mask[i, indices] = 1.0
        
        return mask
    
    def _initialize_weights(self):
        """Initialize weights."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def _apply_sparsity_masks(self):
        """Apply sparsity masks to weights."""
        for i, (layer, mask) in enumerate(zip(self.layers, self.sparse_masks)):
            with torch.no_grad():
                layer.weight.data *= mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_layer_info: bool = True,
        reset_budgets: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass through sparse network."""
        self._forward_count += 1
        
        # Re-apply sparsity masks to ensure sparsity is maintained
        self._apply_sparsity_masks()
        
        current = x
        layer_activations = []
        active_neurons_per_layer = []
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            current = layer(current)
            
            # Apply ReLU activation for hidden layers
            if i < len(self.layers) - 1:
                current = F.relu(current)
            
            layer_activations.append(current)
            
            # Count active neurons
            if return_layer_info:
                active_neurons = (current > 0).float().sum(dim=-1).mean().item()
                active_neurons_per_layer.append(active_neurons)
        
        output = current
        
        if not return_layer_info:
            return output
        
        # Calculate actual sparsity
        total_possible_weights = sum(
            layer.weight.numel() for layer in self.layers[:-1]  # Exclude output layer
        )
        total_active_weights = sum(
            (mask > 0).sum().item() for mask in self.sparse_masks[:-1]
        )
        actual_sparsity = 1.0 - (total_active_weights / max(1, total_possible_weights))
        
        # Compile network info
        network_info = {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'num_layers': len(self.layers),
            'forward_count': self._forward_count,
            'active_neurons_per_layer': active_neurons_per_layer,
            'actual_sparsity': actual_sparsity,
            'target_sparsity': 1.0 - self.sparsity_level,
            'output_magnitude': torch.norm(output).item(),
            'architecture': f'SparseBaseline_{self.sparsity_type}',
            'sparsity_level': self.sparsity_level,
            
            # For compatibility with DRN analysis
            'total_neurons_recruited': total_active_weights,
            'network_sparsity': actual_sparsity,
            'recruitment_efficiency': 1.0,  # Fixed recruitment
        }
        
        self._info_history.append(network_info)
        if len(self._info_history) > 500:
            self._info_history.pop(0)
        
        return output, network_info


def create_baseline_model(
    model_type: str,
    input_size: int,
    output_size: int,
    **kwargs
) -> BaselineModel:
    """
    Factory function to create baseline models.
    
    Args:
        model_type (str): Type of baseline model
        input_size (int): Input dimension
        output_size (int): Output dimension
        **kwargs: Additional model-specific arguments
        
    Returns:
        Configured baseline model
    """
    if model_type.lower() == 'mlp':
        hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
        return StandardMLP(input_size, hidden_sizes, output_size, **kwargs)
    
    elif model_type.lower() in ['lstm', 'gru', 'rnn']:
        hidden_size = kwargs.get('hidden_size', 128)
        num_layers = kwargs.get('num_layers', 2)
        hidden_size = kwargs.pop('hidden_size', 128)
        num_layers = kwargs.pop('num_layers', 2)
        return RecurrentBaseline(
            input_size, hidden_size, num_layers, output_size,
            rnn_type=model_type.lower(), **kwargs
        )
    
    elif model_type.lower() == 'transformer':
        model_dim = kwargs.get('model_dim', 128)
        num_heads = kwargs.get('num_heads', 8)
        num_layers = kwargs.get('num_layers', 4)
        return TransformerBaseline(
            input_size, model_dim, num_heads, num_layers, output_size, **kwargs
        )
    
    elif model_type.lower() == 'sparse':
        hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
        sparsity_level = kwargs.get('sparsity_level', 0.1)
        return SparseBaseline(
            input_size, hidden_sizes, output_size, sparsity_level, **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_comparison_suite(
    input_size: int,
    output_size: int,
    complexity_level: str = 'medium'
) -> Dict[str, BaselineModel]:
    """
    Create a suite of baseline models for comprehensive comparison.
    
    Args:
        input_size (int): Input dimension
        output_size (int): Output dimension
        complexity_level (str): Model complexity ('small', 'medium', 'large')
        
    Returns:
        Dictionary of baseline models for comparison
    """
    if complexity_level == 'small':
        configs = {
            'mlp': {'hidden_sizes': [64, 32]},
            'lstm': {'hidden_size': 64, 'num_layers': 1},
            'transformer': {'model_dim': 64, 'num_heads': 4, 'num_layers': 2},
            'sparse': {'hidden_sizes': [64, 32], 'sparsity_level': 0.2}
        }
    elif complexity_level == 'medium':
        configs = {
            'mlp': {'hidden_sizes': [128, 64]},
            'lstm': {'hidden_size': 128, 'num_layers': 2},
            'transformer': {'model_dim': 128, 'num_heads': 8, 'num_layers': 4},
            'sparse': {'hidden_sizes': [128, 64], 'sparsity_level': 0.1}
        }
    else:  # large
        configs = {
            'mlp': {'hidden_sizes': [256, 128, 64]},
            'lstm': {'hidden_size': 256, 'num_layers': 3},
            'transformer': {'model_dim': 256, 'num_heads': 16, 'num_layers': 6},
            'sparse': {'hidden_sizes': [256, 128, 64], 'sparsity_level': 0.05}
        }
    
    models = {}
    for model_type, config in configs.items():
        try:
            models[model_type] = create_baseline_model(
                model_type, input_size, output_size, **config
            )
        except Exception as e:
            print(f"Warning: Could not create {model_type} model: {e}")
    
    return models