# advanced_continual_learning.py (FIXED VERSION - Complete)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import seaborn as sns

from population_experiment import PopulationDRN
from drn.models.baseline_models import StandardMLP

class AdvancedContinualLearning:
    """Comprehensive continual learning tests with multiple datasets and protocols"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = defaultdict(list)
        
    def create_disjoint_concept_tasks(self, num_tasks=8):
        """Create completely disjoint conceptual tasks"""
        tasks = []
        
        for task_id in range(num_tasks):
            # Each task has unique input-output mapping
            input_dim = 64
            output_dim = 4
            num_samples = 1000
            
            # Create unique task pattern
            X = torch.randn(num_samples, input_dim)
            
            # Different rules for each task
            if task_id == 0:
                y = (X[:, :16].sum(dim=1) > 0).long()
            elif task_id == 1:
                y = (X[:, 16:32].sum(dim=1) > 0).long()
            elif task_id == 2:
                y = (X[:, 32:48].sum(dim=1) > 0).long()
            elif task_id == 3:
                y = (X[:, 48:].sum(dim=1) > 0).long()
            elif task_id == 4:
                y = (X[:, ::2].sum(dim=1) > 0).long()  # Even indices
            elif task_id == 5:
                y = (X[:, 1::2].sum(dim=1) > 0).long()  # Odd indices
            elif task_id == 6:
                y = (torch.norm(X[:, :32], dim=1) > torch.norm(X[:, 32:], dim=1)).long()
            else:
                y = (X.max(dim=1)[0] > X.min(dim=1)[0] + 2).long()
            
            # Convert to multi-class
            y = y % output_dim
            
            # Split into train/test
            split = int(0.8 * num_samples)
            
            tasks.append({
                'task_id': task_id,
                'train_data': TensorDataset(X[:split], y[:split]),
                'test_data': TensorDataset(X[split:], y[split:]),
                'name': f'Concept_{task_id}'
            })
        
        return tasks
    
    def train_with_adaptive_budget(self, model, task, epochs=10, budget_decay=0.95):
        """Train with adaptive neurotransmitter budget"""
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Adapt budget for DRN
        if hasattr(model, 'layer1') and hasattr(model.layer1, 'budget'):
            current_budget = model.layer1.budget
            model.layer1.budget = current_budget * (budget_decay ** task['task_id'])
        
        model.train()
        loader = DataLoader(task['train_data'], batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self._get_output(model, data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
    def _get_output(self, model, data):
        """Handle different model output types"""
        output = model(data)
        return output[0] if isinstance(output, tuple) else output
    
    def evaluate_all_tasks(self, model, tasks):
        """Evaluate model on all tasks"""
        accuracies = []
        
        for task in tasks:
            loader = DataLoader(task['test_data'], batch_size=32, shuffle=False)
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for data, targets in loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = self._get_output(model, data)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = 100. * correct / total
            accuracies.append(accuracy)
        
        return accuracies
    
    def run_comprehensive_continual_learning(self):
        """Run all continual learning experiments"""
        print("🧠 COMPREHENSIVE CONTINUAL LEARNING TESTS\n" + "="*60)
        
        # Test 1: Disjoint Concepts (easiest)
        print("\n📌 Test 1: Disjoint Concept Learning")
        print("-"*40)
        tasks = self.create_disjoint_concept_tasks(num_tasks=8)
        
        # Create models
        drn_model = PopulationDRN(input_size=64, num_classes=4).to(self.device)
        mlp_model = StandardMLP(input_size=64, hidden_sizes=[128, 64], output_size=4).to(self.device)
        
        drn_results = {'task_accuracies': []}
        mlp_results = {'task_accuracies': []}
        
        # Train on each task sequentially
        for task_idx, task in enumerate(tasks):
            print(f"\n🎯 Training on {task['name']}")
            
            # Train models
            self.train_with_adaptive_budget(drn_model, task, epochs=10)
            self.train_with_adaptive_budget(mlp_model, task, epochs=10)
            
            # Evaluate on all previous tasks
            drn_accs = self.evaluate_all_tasks(drn_model, tasks[:task_idx+1])
            mlp_accs = self.evaluate_all_tasks(mlp_model, tasks[:task_idx+1])
            
            drn_results['task_accuracies'].append(drn_accs)
            mlp_results['task_accuracies'].append(mlp_accs)
            
            # Print current performance
            print(f"  DRN: Current task: {drn_accs[-1]:.1f}%, Avg old tasks: {np.mean(drn_accs[:-1]):.1f}%" if len(drn_accs) > 1 else f"  DRN: {drn_accs[-1]:.1f}%")
            print(f"  MLP: Current task: {mlp_accs[-1]:.1f}%, Avg old tasks: {np.mean(mlp_accs[:-1]):.1f}%" if len(mlp_accs) > 1 else f"  MLP: {mlp_accs[-1]:.1f}%")
        
        # Calculate final metrics
        self._analyze_continual_learning_results(drn_results, mlp_results, tasks)
        
        # Visualize results
        self._visualize_continual_learning(drn_results, mlp_results, tasks)
        
        return drn_results, mlp_results
    
    def _analyze_continual_learning_results(self, drn_results, mlp_results, tasks):
        """Analyze and print continual learning metrics"""
        print("\n" + "="*60)
        print("📊 CONTINUAL LEARNING ANALYSIS")
        print("="*60)
        
        # Calculate average forgetting
        drn_forgetting = []
        mlp_forgetting = []
        
        for task_id in range(len(tasks)-1):
            # Get max accuracy for this task
            max_drn = max([accs[task_id] for accs in drn_results['task_accuracies'][task_id:] if task_id < len(accs)])
            max_mlp = max([accs[task_id] for accs in mlp_results['task_accuracies'][task_id:] if task_id < len(accs)])
            
            # Get final accuracy
            final_drn = drn_results['task_accuracies'][-1][task_id] if task_id < len(drn_results['task_accuracies'][-1]) else 0
            final_mlp = mlp_results['task_accuracies'][-1][task_id] if task_id < len(mlp_results['task_accuracies'][-1]) else 0
            
            drn_forgetting.append(max_drn - final_drn)
            mlp_forgetting.append(max_mlp - final_mlp)
        
        print(f"\n🎯 Average Forgetting:")
        print(f"  DRN: {np.mean(drn_forgetting):.2f}%")
        print(f"  MLP: {np.mean(mlp_forgetting):.2f}%")
        print(f"  DRN Advantage: {np.mean(mlp_forgetting) - np.mean(drn_forgetting):.2f}% less forgetting")
        
        # Forward transfer
        print(f"\n🔄 Forward Transfer (accuracy on first attempt):")
        first_attempt_drn = [accs[-1] for accs in drn_results['task_accuracies']]
        first_attempt_mlp = [accs[-1] for accs in mlp_results['task_accuracies']]
        print(f"  DRN: {np.mean(first_attempt_drn):.1f}%")
        print(f"  MLP: {np.mean(first_attempt_mlp):.1f}%")
        
        # Backward transfer
        print(f"\n🔙 Backward Transfer (final accuracy on task 1):")
        if drn_results['task_accuracies'][-1]:
            print(f"  DRN: {drn_results['task_accuracies'][-1][0]:.1f}%")
            print(f"  MLP: {mlp_results['task_accuracies'][-1][0]:.1f}%")
    
    def _visualize_continual_learning(self, drn_results, mlp_results, tasks):
        """Create comprehensive visualization of continual learning results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        num_tasks = len(tasks)
        
        # Plot 1: Performance over time for each task
        ax = axes[0, 0]
        
        # Create color maps for better visualization
        colors_drn = plt.cm.Blues(np.linspace(0.4, 0.9, num_tasks))
        colors_mlp = plt.cm.Reds(np.linspace(0.4, 0.9, num_tasks))
        
        for task_id in range(min(num_tasks, len(drn_results['task_accuracies']))):
            # Get accuracies for this task across all training steps
            task_accs_drn = []
            task_accs_mlp = []
            steps = []
            
            for step in range(task_id, len(drn_results['task_accuracies'])):
                if task_id < len(drn_results['task_accuracies'][step]):
                    task_accs_drn.append(drn_results['task_accuracies'][step][task_id])
                    task_accs_mlp.append(mlp_results['task_accuracies'][step][task_id])
                    steps.append(step)
            
            if task_accs_drn:
                ax.plot(steps, task_accs_drn, 'o-', color=colors_drn[task_id], 
                       label=f'DRN T{task_id}', linewidth=1.5, markersize=4, alpha=0.7)
                ax.plot(steps, task_accs_mlp, 's--', color=colors_mlp[task_id], 
                       label=f'MLP T{task_id}', linewidth=1, markersize=3, alpha=0.5)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Task Performance Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Plot 2: Forgetting heatmap for DRN
        ax = axes[0, 1]
        
        # Create accuracy matrix
        drn_matrix = np.zeros((num_tasks, num_tasks))
        drn_matrix[:] = np.nan
        
        for i, accs in enumerate(drn_results['task_accuracies']):
            for j, acc in enumerate(accs):
                drn_matrix[j, i] = acc
        
        im = ax.imshow(drn_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax.set_xlabel('Training Progress')
        ax.set_ylabel('Task ID')
        ax.set_title('DRN Accuracy Matrix')
        plt.colorbar(im, ax=ax)
        
        # Plot 3: Forgetting heatmap for MLP
        ax = axes[0, 2]
        
        mlp_matrix = np.zeros((num_tasks, num_tasks))
        mlp_matrix[:] = np.nan
        
        for i, accs in enumerate(mlp_results['task_accuracies']):
            for j, acc in enumerate(accs):
                mlp_matrix[j, i] = acc
        
        im = ax.imshow(mlp_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax.set_xlabel('Training Progress')
        ax.set_ylabel('Task ID')
        ax.set_title('MLP Accuracy Matrix')
        plt.colorbar(im, ax=ax)
        
        # Plot 4: Final accuracy per task
        ax = axes[1, 0]
        
        # Get final accuracies (last training step)
        if drn_results['task_accuracies']:
            final_drn = drn_results['task_accuracies'][-1]
            final_mlp = mlp_results['task_accuracies'][-1]
            
            x = np.arange(len(final_drn))
            width = 0.35
            
            ax.bar(x - width/2, final_drn, width, label='DRN', color='#3498db')
            ax.bar(x + width/2, final_mlp, width, label='MLP', color='#e74c3c')
            
            ax.set_xlabel('Task ID')
            ax.set_ylabel('Final Accuracy (%)')
            ax.set_title('Final Performance on All Tasks')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
        
        # Plot 5: Forgetting comparison
        ax = axes[1, 1]
        
        drn_forgetting = []
        mlp_forgetting = []
        
        for task_id in range(num_tasks-1):
            # Find max accuracy for this task
            max_drn = 0
            max_mlp = 0
            
            for step in range(task_id, len(drn_results['task_accuracies'])):
                if task_id < len(drn_results['task_accuracies'][step]):
                    max_drn = max(max_drn, drn_results['task_accuracies'][step][task_id])
                    max_mlp = max(max_mlp, mlp_results['task_accuracies'][step][task_id])
            
            # Get final accuracy
            if drn_results['task_accuracies'] and task_id < len(drn_results['task_accuracies'][-1]):
                final_drn_acc = drn_results['task_accuracies'][-1][task_id]
                final_mlp_acc = mlp_results['task_accuracies'][-1][task_id]
                
                drn_forgetting.append(max_drn - final_drn_acc)
                mlp_forgetting.append(max_mlp - final_mlp_acc)
        
        if drn_forgetting:
            x = np.arange(len(drn_forgetting))
            width = 0.35
            
            ax.bar(x - width/2, drn_forgetting, width, label='DRN', color='#3498db')
            ax.bar(x + width/2, mlp_forgetting, width, label='MLP', color='#e74c3c')
            
            ax.set_xlabel('Task ID')
            ax.set_ylabel('Forgetting (%)')
            ax.set_title('Catastrophic Forgetting by Task')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Summary metrics
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate summary statistics safely
        if drn_results['task_accuracies'] and drn_results['task_accuracies'][-1]:
            final_drn_list = drn_results['task_accuracies'][-1]
            final_mlp_list = mlp_results['task_accuracies'][-1]
            
            mean_drn_forgetting = np.mean(drn_forgetting) if drn_forgetting else 0
            mean_mlp_forgetting = np.mean(mlp_forgetting) if mlp_forgetting else 0
            mean_final_drn = np.mean(final_drn_list)
            mean_final_mlp = np.mean(final_mlp_list)
            max_final_drn = max(final_drn_list)
            max_final_mlp = max(final_mlp_list)
            
            summary_text = f"""
            Summary Metrics:
            
            Average Forgetting:
              DRN: {mean_drn_forgetting:.1f}%
              MLP: {mean_mlp_forgetting:.1f}%
            
            Final Average Accuracy:
              DRN: {mean_final_drn:.1f}%
              MLP: {mean_final_mlp:.1f}%
            
            Best Single Task:
              DRN: {max_final_drn:.1f}%
              MLP: {max_final_mlp:.1f}%
            
            DRN Advantages:
              {mean_mlp_forgetting - mean_drn_forgetting:.1f}% less forgetting
              {mean_final_drn - mean_final_mlp:.1f}% better retention
            """
        else:
            summary_text = "No data available for summary"
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        plt.suptitle('Comprehensive Continual Learning Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig('advanced_continual_learning.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 Visualization saved as 'advanced_continual_learning.png'")


if __name__ == "__main__":
    continual = AdvancedContinualLearning()
    drn_results, mlp_results = continual.run_comprehensive_continual_learning()
    
    print("\n✅ Advanced continual learning tests complete!")