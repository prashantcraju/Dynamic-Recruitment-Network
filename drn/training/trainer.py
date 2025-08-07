"""
drn/training/trainer.py

Training framework specifically designed for Dynamic Recruitment Networks.
Handles the unique aspects of DRN training including budget management,
connectivity regularization, and flexibility metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Callable, Tuple
import time
import logging
from collections import defaultdict
import numpy as np

from ..models.drn_network import DRNNetwork
from .losses import RecruitmentLoss, FlexibilityLoss
from .metrics import FlexibilityMetrics, ConnectivityMetrics


class DRNTrainer:
    """
    Trainer specifically designed for Dynamic Recruitment Networks.
    
    This trainer handles the unique aspects of DRN training:
    - Budget management and regularization
    - Connectivity pattern analysis
    - Flexibility metric computation
    - Adaptive parameter adjustment
    - Multi-objective optimization
    
    Args:
        model: DRN model to train
        optimizer: PyTorch optimizer
        criterion: Base loss function (e.g., CrossEntropyLoss)
        connectivity_weight: Weight for connectivity regularization loss
        flexibility_weight: Weight for flexibility regularization loss
        budget_adaptation: Whether to adapt budgets during training
        device: Device to use for training
        log_interval: How often to log training progress
        
    Example:
        >>> model = DRNNetwork(...)
        >>> trainer = DRNTrainer(model, optimizer, criterion)
        >>> history = trainer.train(train_loader, val_loader, num_epochs=50)
    """
    
    def __init__(
        self,
        model: DRNNetwork,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        connectivity_weight: float = 0.01,
        flexibility_weight: float = 0.005,
        budget_adaptation: bool = True,
        device: Optional[torch.device] = None,
        log_interval: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.base_criterion = criterion
        self.connectivity_weight = connectivity_weight
        self.flexibility_weight = flexibility_weight
        self.budget_adaptation = budget_adaptation
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_interval = log_interval
        
        # Move model to device
        self.model.to(self.device)
        
        # Specialized loss functions
        self.recruitment_loss = RecruitmentLoss(
            connectivity_weight=connectivity_weight,
            diversity_weight=0.01,
            sparsity_weight=0.005
        )
        
        self.flexibility_loss = FlexibilityLoss(
            boundary_smoothness_weight=flexibility_weight,
            adaptation_weight=flexibility_weight * 0.5
        )
        
        # Metrics
        self.flexibility_metrics = FlexibilityMetrics()
        self.connectivity_metrics = ConnectivityMetrics()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float('-inf')
        self.training_history = defaultdict(list)
        
        # Logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for training progress."""
        logger = logging.getLogger('DRNTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def compute_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        network_info: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including base task loss and DRN-specific regularization.
        
        Args:
            outputs: Model outputs [batch_size, output_size]
            targets: Target labels [batch_size, ...]
            network_info: Information from DRN forward pass
            
        Returns:
            Dictionary with loss components
        """
        # Base task loss
        base_loss = self.base_criterion(outputs, targets)
        
        # Recruitment/connectivity loss
        recruitment_loss = self.recruitment_loss(network_info)
        
        # Flexibility loss (encourages smooth decision boundaries)
        flexibility_loss = self.flexibility_loss(outputs, targets, network_info)
        
        # Total loss
        total_loss = (
            base_loss + 
            self.connectivity_weight * recruitment_loss +
            self.flexibility_weight * flexibility_loss
        )
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'recruitment_loss': recruitment_loss,
            'flexibility_loss': flexibility_loss
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with epoch training metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, network_info = self.model(data, return_layer_info=True)
            
            # Compute loss
            loss_dict = self.compute_loss(outputs, targets, network_info)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            batch_size = data.size(0)
            for key, value in loss_dict.items():
                epoch_metrics[key] += value.item() * batch_size
            
            # DRN-specific metrics
            epoch_metrics['total_recruited'] += network_info['total_neurons_recruited'] * batch_size
            epoch_metrics['network_sparsity'] += network_info['network_sparsity'] * batch_size
            epoch_metrics['recruitment_efficiency'] += network_info['recruitment_efficiency'] * batch_size
            
            num_batches += batch_size
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)] '
                    f'Loss: {loss_dict["total_loss"].item():.6f} '
                    f'Recruited: {network_info["total_neurons_recruited"]} '
                    f'Sparsity: {network_info["network_sparsity"]:.3f}'
                )
            
            # Adapt model parameters if enabled
            if self.budget_adaptation and batch_idx % 50 == 0:
                # Use inverse of loss as performance metric
                performance = 1.0 / (1.0 + loss_dict['total_loss'].item())
                self.model.adapt_network_parameters(performance)
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        epoch_metrics['epoch_time'] = time.time() - start_time
        
        return dict(epoch_metrics)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model performance.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_metrics = defaultdict(float)
        num_batches = 0
        
        # For flexibility metrics
        all_outputs = []
        all_targets = []
        all_network_infos = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs, network_info = self.model(data, return_layer_info=True)
                
                # Compute loss
                loss_dict = self.compute_loss(outputs, targets, network_info)
                
                # Update metrics
                batch_size = data.size(0)
                for key, value in loss_dict.items():
                    val_metrics[key] += value.item() * batch_size
                
                # DRN-specific metrics
                val_metrics['total_recruited'] += network_info['total_neurons_recruited'] * batch_size
                val_metrics['network_sparsity'] += network_info['network_sparsity'] * batch_size
                val_metrics['recruitment_efficiency'] += network_info['recruitment_efficiency'] * batch_size
                
                # Accuracy (for classification)
                if len(targets.shape) == 1:  # Classification
                    pred = outputs.argmax(dim=1)
                    val_metrics['accuracy'] += (pred == targets).float().sum().item()
                
                num_batches += batch_size
                
                # Store for flexibility analysis
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_network_infos.append(network_info)
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        # Compute flexibility metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            flexibility_scores = self.flexibility_metrics.compute_flexibility_score(
                self.model, all_outputs, all_targets, all_network_infos
            )
            val_metrics.update(flexibility_scores)
        
        # Compute connectivity metrics
        connectivity_scores = self.connectivity_metrics.analyze_network_connectivity(
            all_network_infos
        )
        val_metrics.update(connectivity_scores)
        
        return dict(val_metrics)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_best_model: bool = True,
        early_stopping_patience: int = 10,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, List[float]]:
        """
        Complete training procedure.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_best_model: Whether to save the best model
            early_stopping_patience: Number of epochs without improvement before stopping
            scheduler: Learning rate scheduler
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_score = float('-inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            # Store history
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            
            # Determine validation score (use accuracy if available, else negative loss)
            val_score = val_metrics.get('accuracy', -val_metrics['total_loss'])
            
            # Check for improvement
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                
                if save_best_model:
                    self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            # Log epoch results
            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs}: '
                f'Train Loss: {train_metrics["total_loss"]:.4f}, '
                f'Val Loss: {val_metrics["total_loss"]:.4f}, '
                f'Val Score: {val_score:.4f}, '
                f'Recruited: {val_metrics["total_recruited"]:.1f}, '
                f'Sparsity: {val_metrics["network_sparsity"]:.3f}'
            )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping after {epoch + 1} epochs')
                break
        
        self.logger.info("Training completed")
        return dict(self.training_history)
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_score': self.best_val_score,
            'training_history': dict(self.training_history),
            'model_config': self.model.get_network_statistics()
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_score = checkpoint['best_val_score']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def evaluate_cognitive_flexibility(
        self, 
        test_loader: DataLoader,
        adaptation_task_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of cognitive flexibility.
        
        Args:
            test_loader: Test data loader
            adaptation_task_loader: Optional loader for adaptation task
            
        Returns:
            Comprehensive flexibility metrics
        """
        self.model.eval()
        
        # Standard evaluation
        test_metrics = self.validate(test_loader)
        
        # Flexibility-specific evaluations
        flexibility_results = {}
        
        with torch.no_grad():
            # 1. Boundary smoothness
            sample_data, sample_targets = next(iter(test_loader))
            sample_data = sample_data.to(self.device)
            
            boundary_smoothness = self.flexibility_metrics.compute_boundary_smoothness(
                self.model, sample_data, num_samples=100
            )
            flexibility_results['boundary_smoothness'] = boundary_smoothness
            
            # 2. Adaptation speed (if adaptation task provided)
            if adaptation_task_loader is not None:
                adaptation_speed = self.flexibility_metrics.compute_adaptation_speed(
                    self.model, adaptation_task_loader, num_steps=10
                )
                flexibility_results['adaptation_speed'] = adaptation_speed
            
            # 3. Connectivity diversity
            all_network_infos = []
            for data, _ in test_loader:
                data = data.to(self.device)
                _, network_info = self.model(data, return_layer_info=True)
                all_network_infos.append(network_info)
            
            connectivity_diversity = self.connectivity_metrics.compute_connectivity_diversity(
                all_network_infos
            )
            flexibility_results['connectivity_diversity'] = connectivity_diversity
        
        # Combine all metrics
        all_metrics = {**test_metrics, **flexibility_results}
        
        return all_metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_history:
            return {'message': 'No training history available'}
        
        # Get network statistics
        network_stats = self.model.get_network_statistics()
        connectivity_summary = self.model.get_connectivity_summary()
        
        return {
            'training_config': {
                'connectivity_weight': self.connectivity_weight,
                'flexibility_weight': self.flexibility_weight,
                'budget_adaptation': self.budget_adaptation,
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
            },
            'final_metrics': {
                key: values[-1] if values else 0.0 
                for key, values in self.training_history.items()
            },
            'network_statistics': network_stats,
            'connectivity_summary': connectivity_summary,
            'training_history_length': len(self.training_history.get('train_total_loss', []))
        }


# Utility functions for training
def create_drn_trainer(
    model: DRNNetwork,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    connectivity_weight: float = 0.01,
    flexibility_weight: float = 0.005,
    criterion: Optional[nn.Module] = None,
    optimizer_type: str = 'adam'
) -> DRNTrainer:
    """
    Create a DRN trainer with reasonable defaults.
    
    Args:
        model: DRN model to train
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        connectivity_weight: Weight for connectivity regularization
        flexibility_weight: Weight for flexibility regularization
        criterion: Loss function (CrossEntropyLoss if None)
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        
    Returns:
        Configured DRNTrainer
    """
    # Default criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return DRNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        connectivity_weight=connectivity_weight,
        flexibility_weight=flexibility_weight,
        budget_adaptation=True
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing DRNTrainer...")
    
    # Create a simple dataset for testing
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, input_dim=64, num_classes=10):
            self.data = torch.randn(size, input_dim)
            self.targets = torch.randint(0, num_classes, (size,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Create model and data
    from ..models.drn_network import create_standard_drn
    
    model = create_standard_drn(
        input_size=64,
        output_size=10,
        hidden_sizes=[32, 16]
    )
    
    train_dataset = SimpleDataset(size=500)
    val_dataset = SimpleDataset(size=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create trainer
    trainer = create_drn_trainer(
        model=model,
        learning_rate=0.001,
        connectivity_weight=0.01,
        flexibility_weight=0.005
    )
    
    print(f"Trainer created with device: {trainer.device}")
    
    # Test single epoch
    print("Testing single epoch...")
    train_metrics = trainer.train_epoch(train_loader)
    print(f"Train metrics: {list(train_metrics.keys())}")
    
    val_metrics = trainer.validate(val_loader)
    print(f"Val metrics: {list(val_metrics.keys())}")
    
    # Test short training
    print("Testing short training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        early_stopping_patience=10
    )
    
    print(f"Training history keys: {list(history.keys())}")
    print(f"Final train loss: {history['train_total_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_total_loss'][-1]:.4f}")
    
    # Test cognitive flexibility evaluation
    print("Testing cognitive flexibility evaluation...")
    flexibility_results = trainer.evaluate_cognitive_flexibility(val_loader)
    print(f"Flexibility metrics: {list(flexibility_results.keys())}")
    
    # Get training summary
    summary = trainer.get_training_summary()
    print(f"Training summary keys: {list(summary.keys())}")
    
    print("DRNTrainer tests passed!")