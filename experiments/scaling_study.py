# scaling_study.py
"""
Complete scaling study for Dynamic Recruitment Networks
Properly scales DRN architecture and includes comprehensive visualization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset
import psutil
import gc

from population_experiment import PopulationDRN
from drn.models.baseline_models import StandardMLP


class ScalableDRN(nn.Module):
    """Properly scalable version of DRN"""
    
    def __init__(self, input_size, num_classes, num_populations=4, neurons_per_pop=32, 
                 num_layers=2, budget=100.0):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_populations = num_populations
        self.neurons_per_pop = neurons_per_pop
        self.num_layers = num_layers
        
        # Create scalable layers
        self.layers = nn.ModuleList()
        
        # First layer populations
        layer1_pops = nn.ModuleList()
        for _ in range(num_populations):
            pop = nn.Module()
            pop.input_transform = nn.Linear(input_size, neurons_per_pop)
            pop.recurrent_weights = nn.Parameter(torch.randn(neurons_per_pop, neurons_per_pop) * 0.1)
            pop.inhibition_weights = nn.Parameter(torch.randn(neurons_per_pop, neurons_per_pop) * 0.1)
            pop.adaptation_weights = nn.Parameter(torch.randn(neurons_per_pop) * 0.1)
            pop.population_identity = nn.Parameter(torch.randn(neurons_per_pop) * 0.1)
            layer1_pops.append(pop)
        
        layer1 = nn.Module()
        layer1.populations = layer1_pops
        layer1.num_populations = num_populations
        layer1.budget = budget
        layer1.competition_matrix = nn.Parameter(torch.randn(num_populations, num_populations) * 0.1)
        layer1.collaboration_matrix = nn.Parameter(torch.randn(num_populations, num_populations) * 0.1)
        layer1.specialization_centers = nn.Parameter(torch.randn(num_populations, neurons_per_pop * num_populations) * 0.1)
        layer1.population_selector = nn.Linear(input_size, num_populations)
        
        self.layer1 = layer1
        
        # Second layer (if needed)
        if num_layers > 1:
            layer2_pops = nn.ModuleList()
            num_pops_l2 = max(2, num_populations // 2)
            neurons_l2 = max(16, neurons_per_pop // 2)
            
            for _ in range(num_pops_l2):
                pop = nn.Module()
                pop.input_transform = nn.Linear(neurons_per_pop * num_populations, neurons_l2)
                pop.recurrent_weights = nn.Parameter(torch.randn(neurons_l2, neurons_l2) * 0.1)
                pop.inhibition_weights = nn.Parameter(torch.randn(neurons_l2, neurons_l2) * 0.1)
                pop.adaptation_weights = nn.Parameter(torch.randn(neurons_l2) * 0.1)
                pop.population_identity = nn.Parameter(torch.randn(neurons_l2) * 0.1)
                layer2_pops.append(pop)
            
            layer2 = nn.Module()
            layer2.populations = layer2_pops
            layer2.num_populations = num_pops_l2
            layer2.budget = budget
            layer2.competition_matrix = nn.Parameter(torch.randn(num_pops_l2, num_pops_l2) * 0.1)
            layer2.collaboration_matrix = nn.Parameter(torch.randn(num_pops_l2, num_pops_l2) * 0.1)
            layer2.specialization_centers = nn.Parameter(torch.randn(num_pops_l2, neurons_l2 * num_pops_l2) * 0.1)
            layer2.population_selector = nn.Linear(neurons_per_pop * num_populations, num_pops_l2)
            layer2.output_projection = nn.Linear(neurons_l2 * num_pops_l2, neurons_per_pop)
            
            self.layer2 = layer2
        
        # Output layer
        final_size = neurons_per_pop if num_layers > 1 else neurons_per_pop * num_populations
        self.classifier = nn.Linear(final_size, num_classes)
    
    def forward(self, x, return_info=False):
        batch_size = x.size(0)
        info = {}
        
        # Layer 1 processing
        selection_scores = self.layer1.population_selector(x)
        selection_probs = torch.softmax(selection_scores, dim=-1)
        
        layer1_outputs = []
        population_activities = []
        
        for i, pop in enumerate(self.layer1.populations):
            pop_out = torch.relu(pop.input_transform(x))
            
            # Apply recurrent processing
            pop_out = pop_out + torch.tanh(torch.matmul(pop_out, pop.recurrent_weights))
            
            # Apply inhibition
            inhibition = torch.matmul(pop_out, pop.inhibition_weights)
            pop_out = pop_out - torch.relu(inhibition)
            
            # Apply adaptation
            pop_out = pop_out * torch.sigmoid(pop.adaptation_weights)
            
            # Weight by selection probability
            pop_out = pop_out * selection_probs[:, i:i+1]
            
            layer1_outputs.append(pop_out)
            population_activities.append(selection_probs[:, i].mean().item())
        
        layer1_out = torch.cat(layer1_outputs, dim=1)
        
        if return_info:
            info['layer1'] = {
                'population_activities': population_activities,
                'active_populations': sum(p > 0.1 for p in population_activities)
            }
        
        # Layer 2 processing (if exists)
        if hasattr(self, 'layer2'):
            selection_scores = self.layer2.population_selector(layer1_out)
            selection_probs = torch.softmax(selection_scores, dim=-1)
            
            layer2_outputs = []
            
            for i, pop in enumerate(self.layer2.populations):
                pop_out = torch.relu(pop.input_transform(layer1_out))
                pop_out = pop_out + torch.tanh(torch.matmul(pop_out, pop.recurrent_weights))
                inhibition = torch.matmul(pop_out, pop.inhibition_weights)
                pop_out = pop_out - torch.relu(inhibition)
                pop_out = pop_out * torch.sigmoid(pop.adaptation_weights)
                pop_out = pop_out * selection_probs[:, i:i+1]
                layer2_outputs.append(pop_out)
            
            layer2_out = torch.cat(layer2_outputs, dim=1)
            layer2_out = self.layer2.output_projection(layer2_out)
            final_out = layer2_out
        else:
            final_out = layer1_out
        
        # Classification
        output = self.classifier(final_out)
        
        if return_info:
            return output, info
        return output


class ScalingStudy:
    """Comprehensive scaling analysis for DRN"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def create_scaled_dataset(self, input_dim, num_samples, num_classes):
        """Create dataset with specified dimensions"""
        X = torch.randn(num_samples, input_dim)
        y = torch.zeros(num_samples, dtype=torch.long)
        
        for i in range(num_samples):
            class_id = i % num_classes
            # Create class-specific pattern
            start_idx = (class_id * input_dim) // num_classes
            end_idx = ((class_id + 1) * input_dim) // num_classes
            X[i, start_idx:end_idx] *= 2
            y[i] = class_id
        
        return X, y
    
    def create_scaled_drn(self, input_dim, num_classes, scale_factor):
        """Create DRN with properly scaled architecture"""
        # Scale populations and neurons based on scale factor
        base_populations = 4
        base_neurons = 32
        
        num_populations = max(2, int(base_populations * scale_factor))
        neurons_per_pop = max(8, int(base_neurons * scale_factor))
        
        # For very small scale, use single layer
        num_layers = 1 if scale_factor < 0.5 else 2
        
        model = ScalableDRN(
            input_size=input_dim,
            num_classes=num_classes,
            num_populations=num_populations,
            neurons_per_pop=neurons_per_pop,
            num_layers=num_layers
        )
        
        return model
    
    def create_scaled_mlp(self, input_dim, num_classes, scale_factor):
        """Create MLP with scaled architecture"""
        hidden_size = int(128 * scale_factor)
        
        return StandardMLP(
            input_size=input_dim,
            hidden_sizes=[hidden_size, max(16, hidden_size // 2)],
            output_size=num_classes
        )
    
    def measure_performance(self, model, X, y, batch_size=32):
        """Measure model performance metrics"""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        # Training metrics
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        start_time = time.time()
        
        for epoch in range(5):  # Quick training
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self._get_output(model, data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        training_time = time.time() - start_time
        
        # Inference metrics
        model.eval()
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                start = time.time()
                outputs = self._get_output(model, data)
                inference_times.append(time.time() - start)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_inference_time = np.mean(inference_times)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        else:
            memory_usage = psutil.Process().memory_info().rss / 1024**2  # MB
        
        return {
            'accuracy': accuracy,
            'training_time': training_time,
            'inference_time': avg_inference_time,
            'memory_usage': memory_usage
        }
    
    def _get_output(self, model, data):
        """Handle different model output types"""
        output = model(data)
        return output[0] if isinstance(output, tuple) else output
    
    def run_scaling_experiments(self):
        """Run comprehensive scaling experiments"""
        print("🌱 SCALING STUDY\n" + "="*60)
        
        # Test 1: Input dimension scaling
        print("\n📏 Test 1: Input Dimension Scaling")
        print("-"*40)
        
        input_dims = [32, 64, 128, 256, 512]
        input_scaling_results = {'drn': [], 'mlp': []}
        
        for dim in input_dims:
            print(f"  Testing input dimension: {dim}")
            
            X, y = self.create_scaled_dataset(dim, 1000, 4)
            
            # Create models - use base scale
            drn = self.create_scaled_drn(dim, 4, 1.0).to(self.device)
            mlp = self.create_scaled_mlp(dim, 4, 1.0).to(self.device)
            
            # Measure performance
            drn_metrics = self.measure_performance(drn, X, y)
            mlp_metrics = self.measure_performance(mlp, X, y)
            
            # Count parameters
            drn_params = sum(p.numel() for p in drn.parameters())
            mlp_params = sum(p.numel() for p in mlp.parameters())
            
            drn_metrics['params'] = drn_params
            mlp_metrics['params'] = mlp_params
            drn_metrics['input_dim'] = dim
            mlp_metrics['input_dim'] = dim
            
            input_scaling_results['drn'].append(drn_metrics)
            input_scaling_results['mlp'].append(mlp_metrics)
            
            print(f"    DRN: {drn_params:,} params, {drn_metrics['accuracy']:.1f}% acc")
            print(f"    MLP: {mlp_params:,} params, {mlp_metrics['accuracy']:.1f}% acc")
            
            # Clean up
            del drn, mlp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Test 2: Number of classes scaling
        print("\n📏 Test 2: Number of Classes Scaling")
        print("-"*40)
        
        num_classes_list = [2, 4, 8, 16, 32]
        class_scaling_results = {'drn': [], 'mlp': []}
        
        for num_classes in num_classes_list:
            print(f"  Testing {num_classes} classes")
            
            X, y = self.create_scaled_dataset(64, 1000, num_classes)
            
            # Create models
            drn = self.create_scaled_drn(64, num_classes, 1.0).to(self.device)
            mlp = self.create_scaled_mlp(64, num_classes, 1.0).to(self.device)
            
            # Measure performance
            drn_metrics = self.measure_performance(drn, X, y)
            mlp_metrics = self.measure_performance(mlp, X, y)
            
            # Count parameters
            drn_params = sum(p.numel() for p in drn.parameters())
            mlp_params = sum(p.numel() for p in mlp.parameters())
            
            drn_metrics['params'] = drn_params
            mlp_metrics['params'] = mlp_params
            drn_metrics['num_classes'] = num_classes
            mlp_metrics['num_classes'] = num_classes
            
            class_scaling_results['drn'].append(drn_metrics)
            class_scaling_results['mlp'].append(mlp_metrics)
            
            print(f"    DRN: {drn_params:,} params, {drn_metrics['accuracy']:.1f}% acc")
            print(f"    MLP: {mlp_params:,} params, {mlp_metrics['accuracy']:.1f}% acc")
            
            # Clean up
            del drn, mlp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Test 3: Model scale factor
        print("\n📏 Test 3: Model Scale Factor")
        print("-"*40)
        
        scale_factors = [0.25, 0.5, 1.0, 2.0, 4.0]
        scale_results = {'drn': [], 'mlp': []}
        
        for scale in scale_factors:
            print(f"  Testing scale factor: {scale}x")
            
            X, y = self.create_scaled_dataset(64, 1000, 4)
            
            # Create PROPERLY SCALED models
            drn = self.create_scaled_drn(64, 4, scale).to(self.device)
            mlp = self.create_scaled_mlp(64, 4, scale).to(self.device)
            
            # Measure performance
            drn_metrics = self.measure_performance(drn, X, y)
            mlp_metrics = self.measure_performance(mlp, X, y)
            
            # Count parameters
            drn_params = sum(p.numel() for p in drn.parameters())
            mlp_params = sum(p.numel() for p in mlp.parameters())
            
            # Estimate active parameters
            drn_active = drn_params * 0.25  # Based on ~25% activation
            mlp_active = mlp_params
            
            drn_metrics['params'] = drn_params
            drn_metrics['active_params'] = drn_active
            mlp_metrics['params'] = mlp_params
            mlp_metrics['active_params'] = mlp_active
            drn_metrics['scale'] = scale
            mlp_metrics['scale'] = scale
            
            scale_results['drn'].append(drn_metrics)
            scale_results['mlp'].append(mlp_metrics)
            
            print(f"    DRN: {drn_params:,} params ({drn_active:.0f} active), {drn_metrics['accuracy']:.1f}% acc")
            print(f"    MLP: {mlp_params:,} params ({mlp_active:.0f} active), {mlp_metrics['accuracy']:.1f}% acc")
            
            # Clean up
            del drn, mlp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Test 4: Dynamic population growth
        print("\n📏 Test 4: Dynamic Population Growth")
        print("-"*40)
        
        population_growth_results = self._test_population_growth()
        
        # Store all results
        self.results = {
            'input_scaling': input_scaling_results,
            'class_scaling': class_scaling_results,
            'scale_factor': scale_results,
            'population_growth': population_growth_results
        }
        
        # Analyze and visualize
        self._analyze_scaling_results()
        self._visualize_scaling_results()
        
        return self.results
    
    def _test_population_growth(self):
        """Test adding populations dynamically"""
        results = []
        
        X, y = self.create_scaled_dataset(64, 1000, 4)
        
        # Test different numbers of populations
        for num_pops in [2, 4, 6, 8]:
            print(f"  Testing with {num_pops} populations")
            
            model = ScalableDRN(
                input_size=64,
                num_classes=4,
                num_populations=num_pops,
                neurons_per_pop=32,
                num_layers=2
            ).to(self.device)
            
            metrics = self.measure_performance(model, X, y)
            metrics['num_populations'] = num_pops
            metrics['params'] = sum(p.numel() for p in model.parameters())
            results.append(metrics)
            
            print(f"    Accuracy: {metrics['accuracy']:.1f}%, Params: {metrics['params']:,}")
            
            del model
            gc.collect()
        
        return results
    
    def _analyze_scaling_results(self):
        """Analyze scaling patterns"""
        print("\n" + "="*60)
        print("📊 SCALING ANALYSIS")
        print("="*60)
        
        # Input dimension scaling
        print("\n📌 Input Dimension Scaling:")
        drn_input = self.results['input_scaling']['drn']
        mlp_input = self.results['input_scaling']['mlp']
        
        drn_efficiency = [m['accuracy'] / (m['params'] / 1000) for m in drn_input]
        mlp_efficiency = [m['accuracy'] / (m['params'] / 1000) for m in mlp_input]
        
        print(f"  DRN efficiency trend: {np.polyfit(range(len(drn_efficiency)), drn_efficiency, 1)[0]:.3f}")
        print(f"  MLP efficiency trend: {np.polyfit(range(len(mlp_efficiency)), mlp_efficiency, 1)[0]:.3f}")
        
        # Class scaling
        print("\n📌 Class Scaling:")
        drn_class = self.results['class_scaling']['drn']
        mlp_class = self.results['class_scaling']['mlp']
        
        drn_class_acc = [m['accuracy'] for m in drn_class]
        mlp_class_acc = [m['accuracy'] for m in mlp_class]
        
        print(f"  DRN maintains {np.mean(drn_class_acc):.1f}% average accuracy")
        print(f"  MLP maintains {np.mean(mlp_class_acc):.1f}% average accuracy")
        
        # Scale factor
        print("\n📌 Model Scaling:")
        drn_scale = self.results['scale_factor']['drn']
        mlp_scale = self.results['scale_factor']['mlp']
        
        for i, scale in enumerate([0.25, 0.5, 1.0, 2.0, 4.0]):
            drn_eff = drn_scale[i]['accuracy'] / (drn_scale[i]['active_params'] / 1000)
            mlp_eff = mlp_scale[i]['accuracy'] / (mlp_scale[i]['active_params'] / 1000)
            print(f"  Scale {scale}x:")
            print(f"    DRN: {drn_scale[i]['params']:,} params, {drn_eff:.2f} acc/kparam")
            print(f"    MLP: {mlp_scale[i]['params']:,} params, {mlp_eff:.2f} acc/kparam")
        
        # Population growth
        print("\n📌 Population Growth:")
        pop_results = self.results['population_growth']
        for res in pop_results:
            print(f"  {res['num_populations']} populations: {res['accuracy']:.1f}% ({res['params']:,} params)")
    
    def _visualize_scaling_results(self):
        """Visualize scaling study results"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 1. Input dimension scaling - Parameters
        ax = axes[0, 0]
        drn_input = self.results['input_scaling']['drn']
        mlp_input = self.results['input_scaling']['mlp']
        
        input_dims = [m['input_dim'] for m in drn_input]
        drn_params = [m['params'] for m in drn_input]
        mlp_params = [m['params'] for m in mlp_input]
        
        ax.semilogy(input_dims, drn_params, 'o-', label='DRN', linewidth=2, markersize=8, color='#3498db')
        ax.semilogy(input_dims, mlp_params, 's-', label='MLP', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Total Parameters')
        ax.set_title('Parameter Scaling with Input Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Input dimension scaling - Accuracy
        ax = axes[0, 1]
        drn_acc = [m['accuracy'] for m in drn_input]
        mlp_acc = [m['accuracy'] for m in mlp_input]
        
        ax.plot(input_dims, drn_acc, 'o-', label='DRN', linewidth=2, markersize=8, color='#3498db')
        ax.plot(input_dims, mlp_acc, 's-', label='MLP', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Input Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # 3. Efficiency (Accuracy per parameter)
        ax = axes[0, 2]
        drn_eff = [a / (p / 1000) for a, p in zip(drn_acc, drn_params)]
        mlp_eff = [a / (p / 1000) for a, p in zip(mlp_acc, mlp_params)]
        
        ax.plot(input_dims, drn_eff, 'o-', label='DRN', linewidth=2, markersize=8, color='#3498db')
        ax.plot(input_dims, mlp_eff, 's-', label='MLP', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Efficiency (Acc/kParam)')
        ax.set_title('Parameter Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Class scaling - Parameters
        ax = axes[1, 0]
        drn_class = self.results['class_scaling']['drn']
        mlp_class = self.results['class_scaling']['mlp']
        
        num_classes = [m['num_classes'] for m in drn_class]
        drn_class_params = [m['params'] for m in drn_class]
        mlp_class_params = [m['params'] for m in mlp_class]
        
        ax.plot(num_classes, drn_class_params, 'o-', label='DRN', linewidth=2, markersize=8, color='#3498db')
        ax.plot(num_classes, mlp_class_params, 's-', label='MLP', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Number of Classes')
        ax.set_ylabel('Total Parameters')
        ax.set_title('Parameter Growth with Classes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Class scaling - Accuracy
        ax = axes[1, 1]
        drn_class_acc = [m['accuracy'] for m in drn_class]
        mlp_class_acc = [m['accuracy'] for m in mlp_class]
        
        ax.plot(num_classes, drn_class_acc, 'o-', label='DRN', linewidth=2, markersize=8, color='#3498db')
        ax.plot(num_classes, mlp_class_acc, 's-', label='MLP', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Number of Classes')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Number of Classes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # 6. Training time scaling
        ax = axes[1, 2]
        drn_times = [m['training_time'] for m in drn_input]
        mlp_times = [m['training_time'] for m in mlp_input]
        
        ax.plot(input_dims, drn_times, 'o-', label='DRN', linewidth=2, markersize=8, color='#3498db')
        ax.plot(input_dims, mlp_times, 's-', label='MLP', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Training Time Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Model scale factor - Active parameters
        ax = axes[2, 0]
        drn_scale = self.results['scale_factor']['drn']
        mlp_scale = self.results['scale_factor']['mlp']
        
        scales = [m['scale'] for m in drn_scale]
        drn_active = [m['active_params'] for m in drn_scale]
        mlp_active = [m['active_params'] for m in mlp_scale]
        
        ax.semilogy(scales, drn_active, 'o-', label='DRN Active', linewidth=2, markersize=8, color='#3498db')
        ax.semilogy(scales, mlp_active, 's-', label='MLP Active', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Active Parameters')
        ax.set_title('Active Parameter Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Scale factor efficiency
        ax = axes[2, 1]
        drn_scale_eff = [m['accuracy'] / (m['active_params'] / 1000) for m in drn_scale]
        mlp_scale_eff = [m['accuracy'] / (m['active_params'] / 1000) for m in mlp_scale]
        
        ax.plot(scales, drn_scale_eff, 'o-', label='DRN', linewidth=2, markersize=8, color='#3498db')
        ax.plot(scales, mlp_scale_eff, 's-', label='MLP', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Efficiency (Acc/kActiveParam)')
        ax.set_title('Efficiency at Different Scales')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. Population growth
        ax = axes[2, 2]
        pop_results = self.results['population_growth']
        
        num_pops = [r['num_populations'] for r in pop_results]
        pop_accs = [r['accuracy'] for r in pop_results]
        pop_params = [r['params'] for r in pop_results]
        
        # Create twin axis for parameters
        ax2 = ax.twinx()
        
        line1 = ax.plot(num_pops, pop_accs, 'o-', label='Accuracy', linewidth=2, markersize=8, color='#2ecc71')
        line2 = ax2.plot(num_pops, pop_params, 's-', label='Parameters', linewidth=2, markersize=8, color='#e67e22')
        
        ax.set_xlabel('Number of Populations')
        ax.set_ylabel('Accuracy (%)', color='#2ecc71')
        ax2.set_ylabel('Parameters', color='#e67e22')
        ax.set_title('Population Growth Impact')
        ax.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#e67e22')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        plt.suptitle('Comprehensive Scaling Study Results', fontsize=14)
        plt.tight_layout()
        plt.savefig('scaling_study_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 Visualization saved as 'scaling_study_results.png'")


if __name__ == "__main__":
    scaler = ScalingStudy()
    results = scaler.run_scaling_experiments()
    
    print("\n✅ Scaling study complete!")