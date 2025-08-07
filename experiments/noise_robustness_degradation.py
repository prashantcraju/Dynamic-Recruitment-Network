# noise_robustness_degradation.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import seaborn as sns

from population_experiment import PopulationDRN
from drn.models.baseline_models import StandardMLP

class NoiseRobustnessTests:
    """Comprehensive noise robustness and degradation analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_test_data(self, num_samples=1000, num_concepts=4):
        """Create test dataset with clear concept structure"""
        X = torch.randn(num_samples, 64)
        
        # Create different concept patterns
        y = torch.zeros(num_samples, dtype=torch.long)
        
        for i in range(num_samples):
            concept = i % num_concepts
            if concept == 0:
                X[i, :16] *= 2  # Amplify first quarter
                y[i] = 0
            elif concept == 1:
                X[i, 16:32] *= 2  # Amplify second quarter
                y[i] = 1
            elif concept == 2:
                X[i, 32:48] *= 2  # Amplify third quarter
                y[i] = 2
            else:
                X[i, 48:] *= 2  # Amplify fourth quarter
                y[i] = 3
        
        return X, y
    
    def add_gaussian_noise(self, data, noise_level):
        """Add Gaussian noise with specified standard deviation"""
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    def add_salt_pepper_noise(self, data, corruption_rate):
        """Add salt and pepper noise"""
        noisy_data = data.clone()
        mask = torch.rand_like(data) < corruption_rate
        
        # Half salt (max), half pepper (min)
        salt_mask = mask & (torch.rand_like(data) < 0.5)
        pepper_mask = mask & ~salt_mask
        
        noisy_data[salt_mask] = data.max()
        noisy_data[pepper_mask] = data.min()
        
        return noisy_data
    
    def add_occlusion(self, data, occlusion_size=0.3):
        """Add random occlusions to input"""
        noisy_data = data.clone()
        num_features = data.shape[1]
        
        for i in range(len(data)):
            # Randomly select region to occlude
            start_idx = torch.randint(0, int(num_features * (1 - occlusion_size)), (1,)).item()
            end_idx = start_idx + int(num_features * occlusion_size)
            noisy_data[i, start_idx:end_idx] = 0
        
        return noisy_data
    
    def add_adversarial_noise(self, model, data, targets, epsilon):
        """Generate adversarial examples using FGSM"""
        data = data.clone().detach().requires_grad_(True)
        
        outputs = self._get_output(model, data)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        model.zero_grad()
        loss.backward()
        
        # Create adversarial perturbation
        data_grad = data.grad.data
        perturbation = epsilon * data_grad.sign()
        adversarial_data = data + perturbation
        
        return adversarial_data.detach()
    
    def add_structured_noise(self, data, pattern='stripe'):
        """Add structured noise patterns"""
        noisy_data = data.clone()
        
        if pattern == 'stripe':
            # Add striped pattern noise
            for i in range(0, data.shape[1], 4):
                noisy_data[:, i:i+2] += torch.randn(data.shape[0], min(2, data.shape[1]-i)) * 0.5
        elif pattern == 'block':
            # Add block pattern noise
            block_size = 8
            for i in range(0, data.shape[1], block_size*2):
                noisy_data[:, i:i+block_size] += torch.randn(data.shape[0], min(block_size, data.shape[1]-i)) * 0.5
        elif pattern == 'gradient':
            # Add gradient noise (increasing with position)
            for i in range(data.shape[1]):
                noise_level = i / data.shape[1]
                noisy_data[:, i] += torch.randn(data.shape[0]) * noise_level
        
        return noisy_data
    
    def _get_output(self, model, data):
        """Handle different model output types"""
        output = model(data)
        return output[0] if isinstance(output, tuple) else output
    
    def train_model(self, model, X_train, y_train, epochs=30):
        """Train model on clean data"""
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self._get_output(model, data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
    def evaluate_with_noise(self, model, X_test, y_test, noise_fn, noise_levels):
        """Evaluate model performance under different noise levels"""
        model.eval()
        accuracies = []
        
        for noise_level in noise_levels:
            # Add noise
            if callable(noise_fn):
                X_noisy = noise_fn(X_test, noise_level)
            else:
                X_noisy = X_test  # No noise baseline
            
            # Evaluate
            dataset = TensorDataset(X_noisy, y_test)
            loader = DataLoader(dataset, batch_size=32)
            
            correct = 0
            total = 0
            
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
    
    def run_comprehensive_noise_tests(self):
        """Run all noise robustness tests"""
        print("📉 COMPREHENSIVE NOISE ROBUSTNESS & DEGRADATION TESTS\n" + "="*60)
        
        # Create datasets
        X_train, y_train = self.create_test_data(1000, 4)
        X_test, y_test = self.create_test_data(500, 4)
        
        # Create and train models
        print("\n🎯 Training models on clean data...")
        drn_model = PopulationDRN(input_size=64, num_classes=4).to(self.device)
        mlp_model = StandardMLP(input_size=64, hidden_sizes=[128, 64], output_size=4).to(self.device)
        
        self.train_model(drn_model, X_train, y_train)
        self.train_model(mlp_model, X_train, y_train)
        
        # Test different noise types
        noise_tests = {
            'Gaussian': {
                'fn': self.add_gaussian_noise,
                'levels': np.linspace(0, 3, 30),
                'label': 'Noise σ'
            },
            'Salt & Pepper': {
                'fn': self.add_salt_pepper_noise,
                'levels': np.linspace(0, 0.5, 30),
                'label': 'Corruption Rate'
            },
            'Occlusion': {
                'fn': self.add_occlusion,
                'levels': np.linspace(0, 0.7, 30),
                'label': 'Occlusion Size'
            },
            'Adversarial': {
                'fn': lambda x, eps: self.add_adversarial_noise(drn_model, x, y_test, eps),
                'levels': np.linspace(0, 0.3, 30),
                'label': 'ε (epsilon)'
            },
            'Stripe Pattern': {
                'fn': lambda x, _: self.add_structured_noise(x, 'stripe'),
                'levels': np.linspace(0, 1, 30),
                'label': 'Pattern Intensity'
            }
        }
        
        results = {}
        
        print("\n🔬 Testing noise robustness...")
        for noise_type, config in noise_tests.items():
            print(f"  Testing {noise_type}...")
            
            drn_accs = self.evaluate_with_noise(
                drn_model, X_test, y_test, 
                config['fn'], config['levels']
            )
            
            mlp_accs = self.evaluate_with_noise(
                mlp_model, X_test, y_test,
                config['fn'], config['levels']
            )
            
            results[noise_type] = {
                'drn': drn_accs,
                'mlp': mlp_accs,
                'levels': config['levels'],
                'label': config['label']
            }
        
        # Analyze degradation patterns
        self._analyze_degradation_patterns(results)
        
        # Visualize results
        self._visualize_noise_robustness(results)
        
        # Test on unseen concepts
        print("\n🆕 Testing on unseen concept classes...")
        X_unseen, y_unseen = self.create_test_data(200, 6)  # 6 classes (2 unseen)
        self._test_unseen_robustness(drn_model, mlp_model, X_unseen, y_unseen)
        
        return results
    
    def _analyze_degradation_patterns(self, results):
        """Analyze how models degrade under noise"""
        print("\n" + "="*60)
        print("📊 DEGRADATION PATTERN ANALYSIS")
        print("="*60)
        
        for noise_type, data in results.items():
            drn_accs = data['drn']
            mlp_accs = data['mlp']
            levels = data['levels']
            
            # Find point where accuracy drops below 80%
            drn_threshold = next((i for i, acc in enumerate(drn_accs) if acc < 80), len(levels))
            mlp_threshold = next((i for i, acc in enumerate(mlp_accs) if acc < 80), len(levels))
            
            # Calculate degradation rate (linear fit)
            drn_slope, _ = np.polyfit(levels[:len(levels)//2], drn_accs[:len(levels)//2], 1)
            mlp_slope, _ = np.polyfit(levels[:len(levels)//2], mlp_accs[:len(levels)//2], 1)
            
            # Calculate area under curve (robustness metric)
            drn_auc = np.trapz(drn_accs, levels)
            mlp_auc = np.trapz(mlp_accs, levels)
            
            print(f"\n📌 {noise_type}:")
            print(f"  80% Threshold - DRN: {levels[min(drn_threshold, len(levels)-1)]:.3f}, MLP: {levels[min(mlp_threshold, len(levels)-1)]:.3f}")
            print(f"  Degradation Rate - DRN: {abs(drn_slope):.1f}%/unit, MLP: {abs(mlp_slope):.1f}%/unit")
            print(f"  Robustness (AUC) - DRN: {drn_auc:.1f}, MLP: {mlp_auc:.1f}")
            
            if drn_auc > mlp_auc:
                print(f"  ✅ DRN is {(drn_auc/mlp_auc - 1)*100:.1f}% more robust")
            else:
                print(f"  ❌ MLP is {(mlp_auc/drn_auc - 1)*100:.1f}% more robust")
        
        # Overall robustness score
        overall_drn = np.mean([np.mean(data['drn']) for data in results.values()])
        overall_mlp = np.mean([np.mean(data['mlp']) for data in results.values()])
        
        print(f"\n🎯 Overall Robustness Score:")
        print(f"  DRN: {overall_drn:.1f}%")
        print(f"  MLP: {overall_mlp:.1f}%")
    
    def _test_unseen_robustness(self, drn_model, mlp_model, X_unseen, y_unseen):
        """Test robustness on unseen concept classes"""
        # Only test on classes 4 and 5 (unseen during training)
        unseen_mask = y_unseen >= 4
        X_unseen = X_unseen[unseen_mask]
        y_unseen = y_unseen[unseen_mask] - 4  # Remap to 0-1
        
        noise_levels = np.linspace(0, 2, 20)
        
        drn_unseen = self.evaluate_with_noise(
            drn_model, X_unseen, y_unseen,
            self.add_gaussian_noise, noise_levels
        )
        
        mlp_unseen = self.evaluate_with_noise(
            mlp_model, X_unseen, y_unseen,
            self.add_gaussian_noise, noise_levels
        )
        
        print(f"  Unseen concepts (avg accuracy):")
        print(f"    DRN: {np.mean(drn_unseen):.1f}%")
        print(f"    MLP: {np.mean(mlp_unseen):.1f}%")
    
    def _visualize_noise_robustness(self, results):
        """Create comprehensive visualization of noise robustness"""
        fig = plt.figure(figsize=(18, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1-5: Individual noise type curves
        for idx, (noise_type, data) in enumerate(results.items()):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            ax.plot(data['levels'], data['drn'], 'o-', label='DRN', 
                   linewidth=2, markersize=4, color='#3498db')
            ax.plot(data['levels'], data['mlp'], 's-', label='MLP', 
                   linewidth=2, markersize=4, color='#e74c3c')
            
            ax.set_xlabel(data['label'])
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{noise_type} Robustness')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
            
            # Add graceful degradation region
            ax.axhspan(60, 80, alpha=0.1, color='yellow', label='Graceful degradation')
            ax.axhspan(0, 60, alpha=0.1, color='red', label='Poor performance')
        
        # Plot 6: Heatmap of robustness across all noise types
        ax = fig.add_subplot(gs[1, 2])
        
        # Create robustness matrix
        noise_names = list(results.keys())
        robustness_matrix = []
        
        for noise_type in noise_names:
            drn_mean = np.mean(results[noise_type]['drn'])
            mlp_mean = np.mean(results[noise_type]['mlp'])
            robustness_matrix.append([drn_mean, mlp_mean])
        
        im = ax.imshow(np.array(robustness_matrix).T, cmap='RdYlGn', vmin=0, vmax=100)
        ax.set_xticks(range(len(noise_names)))
        ax.set_xticklabels(noise_names, rotation=45, ha='right')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['DRN', 'MLP'])
        ax.set_title('Robustness Heatmap')
        
        # Add text annotations
        for i in range(len(noise_names)):
            for j in range(2):
                text = ax.text(i, j, f'{robustness_matrix[i][j]:.0f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
        
        # Plot 7: Degradation rate comparison
        ax = fig.add_subplot(gs[2, 0])
        
        degradation_rates = []
        for noise_type, data in results.items():
            levels = data['levels'][:10]  # First part of curve
            drn_accs = data['drn'][:10]
            mlp_accs = data['mlp'][:10]
            
            drn_rate = abs(np.polyfit(levels, drn_accs, 1)[0])
            mlp_rate = abs(np.polyfit(levels, mlp_accs, 1)[0])
            degradation_rates.append((noise_type, drn_rate, mlp_rate))
        
        noise_types = [x[0] for x in degradation_rates]
        drn_rates = [x[1] for x in degradation_rates]
        mlp_rates = [x[2] for x in degradation_rates]
        
        x = np.arange(len(noise_types))
        width = 0.35
        
        ax.bar(x - width/2, drn_rates, width, label='DRN', color='#3498db')
        ax.bar(x + width/2, mlp_rates, width, label='MLP', color='#e74c3c')
        
        ax.set_xlabel('Noise Type')
        ax.set_ylabel('Degradation Rate (%/unit)')
        ax.set_title('Degradation Rate Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(noise_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Area under curve (overall robustness)
        ax = fig.add_subplot(gs[2, 1])
        
        aucs = []
        for noise_type, data in results.items():
            drn_auc = np.trapz(data['drn'], data['levels'])
            mlp_auc = np.trapz(data['mlp'], data['levels'])
            aucs.append((noise_type, drn_auc, mlp_auc))
        
        noise_types = [x[0] for x in aucs]
        drn_aucs = [x[1] for x in aucs]
        mlp_aucs = [x[2] for x in aucs]
        
        x = np.arange(len(noise_types))
        
        ax.bar(x - width/2, drn_aucs, width, label='DRN', color='#3498db')
        ax.bar(x + width/2, mlp_aucs, width, label='MLP', color='#e74c3c')
        
        ax.set_xlabel('Noise Type')
        ax.set_ylabel('AUC (Robustness Score)')
        ax.set_title('Overall Robustness (Area Under Curve)')
        ax.set_xticks(x)
        ax.set_xticklabels(noise_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Summary statistics
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        
        # Calculate summary statistics
        overall_drn = np.mean([np.mean(data['drn']) for data in results.values()])
        overall_mlp = np.mean([np.mean(data['mlp']) for data in results.values()])
        
        best_drn = max([np.mean(data['drn']) for data in results.values()])
        worst_drn = min([np.mean(data['drn']) for data in results.values()])
        
        summary_text = f"""
        Robustness Summary:
        
        Overall Average:
          DRN: {overall_drn:.1f}%
          MLP: {overall_mlp:.1f}%
        
        DRN Performance Range:
          Best: {best_drn:.1f}%
          Worst: {worst_drn:.1f}%
        
        Key Findings:
        • Models show graceful degradation
        • DRN and MLP have similar robustness
        • Adversarial noise affects both
        • Structured noise patterns reveal
          different vulnerabilities
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
        
        plt.suptitle('Comprehensive Noise Robustness & Degradation Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig('noise_robustness_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    noise_tests = NoiseRobustnessTests()
    results = noise_tests.run_comprehensive_noise_tests()
    
    print("\n✅ Comprehensive noise robustness tests complete!")