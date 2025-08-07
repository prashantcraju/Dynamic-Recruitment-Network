# drn_fixes_comprehensive_fixed.py
"""
Comprehensive fixes for DRN overfitting - FIXED VERSION
Corrects ModuleDict issue with Parameters
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import defaultdict
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


# ==================== OPTION 1: REGULARIZED DRN (FIXED) ====================

class RegularizedDRN(nn.Module):
    """DRN with strong regularization to prevent overfitting"""
    
    def __init__(self, input_size, num_classes, num_populations=4, 
                 neurons_per_pop=32, dropout_rate=0.2, weight_decay=0.001):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_populations = num_populations
        self.neurons_per_pop = neurons_per_pop
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.noise_scale = 0.05
        self.entropy_loss = 0
        
        # Layer normalization for inputs
        self.input_norm = nn.LayerNorm(input_size)
        
        # Population selector with dropout
        self.population_selector = nn.Sequential(
            nn.Linear(input_size, num_populations * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_populations * 4, num_populations)
        )
        
        # Create populations with regularization
        self.populations = nn.ModuleList()
        self.adaptation_params = nn.ParameterList()  # Store adaptation separately
        
        for _ in range(num_populations):
            # Create population modules
            pop = nn.ModuleDict({
                'input_transform': nn.Sequential(
                    nn.Linear(input_size, neurons_per_pop),
                    nn.LayerNorm(neurons_per_pop),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                'recurrent': nn.Sequential(
                    nn.Linear(neurons_per_pop, neurons_per_pop),
                    nn.LayerNorm(neurons_per_pop),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate * 0.5)
                ),
                'inhibition': nn.Sequential(
                    nn.Linear(neurons_per_pop, neurons_per_pop),
                    nn.Sigmoid(),
                    nn.Dropout(dropout_rate * 0.3)
                )
            })
            self.populations.append(pop)
            
            # Store adaptation as separate parameter
            self.adaptation_params.append(nn.Parameter(torch.ones(neurons_per_pop) * 0.5))
        
        # Output normalization
        self.output_norm = nn.LayerNorm(neurons_per_pop * num_populations)
        
        # Output classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(neurons_per_pop * num_populations, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Competition and collaboration with L2 constraint
        self.competition_matrix = nn.Parameter(torch.eye(num_populations) * 0.1)
        self.collaboration_matrix = nn.Parameter(torch.eye(num_populations) * 0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Add noise during training (biological realism)
        if self.training and self.noise_scale > 0:
            x = x + torch.randn_like(x) * self.noise_scale
        
        # Population selection with softmax
        selection_scores = self.population_selector(x)
        selection_probs = F.softmax(selection_scores, dim=-1)
        
        # Entropy regularization for diversity
        if self.training:
            entropy = -torch.sum(selection_probs * torch.log(selection_probs + 1e-8), dim=-1)
            self.entropy_loss = -entropy.mean() * 0.01
        
        # Process through populations
        population_outputs = []
        
        for i, (pop, adaptation) in enumerate(zip(self.populations, self.adaptation_params)):
            # Feedforward
            pop_out = pop['input_transform'](x)
            
            # Recurrent with residual
            recurrent_out = pop['recurrent'](pop_out)
            pop_out = pop_out + 0.1 * recurrent_out
            
            # Inhibition
            inhibition = pop['inhibition'](pop_out)
            pop_out = pop_out * (1 - 0.1 * inhibition)
            
            # Adaptation
            pop_out = pop_out * torch.sigmoid(adaptation)
            
            # Weight by selection probability
            pop_out = pop_out * selection_probs[:, i:i+1]
            
            population_outputs.append(pop_out)
        
        # Combine populations
        combined = torch.cat(population_outputs, dim=1)
        combined = self.output_norm(combined)
        
        # Classification
        output = self.classifier(combined)
        
        return output
    
    def get_regularization_loss(self):
        """Calculate L2 regularization loss"""
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        return self.weight_decay * reg_loss + self.entropy_loss


# ==================== OPTION 2: SIMPLIFIED DRN ====================

class SimplifiedDRN(nn.Module):
    """Minimal DRN with only effective components"""
    
    def __init__(self, input_size, num_classes, num_populations=4, neurons_per_pop=32):
        super().__init__()
        
        self.num_populations = num_populations
        self.neurons_per_pop = neurons_per_pop
        
        # Simple population selector
        self.population_selector = nn.Linear(input_size, num_populations)
        
        # Only feedforward transformations (what actually works)
        self.populations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, neurons_per_pop),
                nn.ReLU(),
                nn.BatchNorm1d(neurons_per_pop)
            ) for _ in range(num_populations)
        ])
        
        # Simple classifier
        self.classifier = nn.Linear(neurons_per_pop * num_populations, num_classes)
        
    def forward(self, x):
        # Population selection
        selection_probs = F.softmax(self.population_selector(x), dim=-1)
        
        # Process through populations (only feedforward)
        population_outputs = []
        for i, pop in enumerate(self.populations):
            pop_out = pop(x)
            # Sparse activation based on selection
            pop_out = pop_out * selection_probs[:, i:i+1]
            population_outputs.append(pop_out)
        
        # Combine and classify
        combined = torch.cat(population_outputs, dim=1)
        return self.classifier(combined)


# ==================== OPTION 3: IMPROVED TRAINING STRATEGY ====================

class ImprovedTrainer:
    """Advanced training strategies to prevent overfitting"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.best_model = None
        self.training_history = defaultdict(list)
        
    def augment_data(self, X, y, augmentation_factor=2):
        """Data augmentation for better generalization"""
        augmented_X = [X]
        augmented_y = [y]
        
        for _ in range(augmentation_factor - 1):
            # Add noise
            noise = torch.randn_like(X) * 0.1
            augmented_X.append(X + noise)
            augmented_y.append(y)
            
            # Random masking (dropout at data level)
            mask = torch.rand_like(X) > 0.1
            masked_X = X * mask.float()
            augmented_X.append(masked_X)
            augmented_y.append(y)
        
        return torch.cat(augmented_X), torch.cat(augmented_y)
    
    def train_with_early_stopping(self, model, X_train, y_train, X_val, y_val,
                                 epochs=100, patience=10, lr=0.001):
        """Train with early stopping and learning rate scheduling"""
        
        model = model.to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                    patience=5, verbose=False)
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                
                # Add regularization if available
                if hasattr(model, 'get_regularization_loss'):
                    loss = loss + model.get_regularization_loss()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss={train_loss:.3f}, Train Acc={train_acc:.1f}%, '
                      f'Val Loss={val_loss:.3f}, Val Acc={val_acc:.1f}%')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        model.load_state_dict(self.best_model)
        return model
    
    def evaluate(self, model, X_test, y_test):
        """Evaluate model performance"""
        model.eval()
        dataset = TensorDataset(X_test, y_test)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        return 100. * correct / total


# ==================== COMPREHENSIVE COMPARISON ====================

class ComprehensiveComparison:
    """Compare all three approaches"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def create_datasets(self, dataset_type='temporal'):
        """Create train/val/test datasets"""
        if dataset_type == 'temporal':
            # Temporal patterns
            num_samples = 3000
            X = torch.randn(num_samples, 64)
            
            for i in range(1, num_samples):
                X[i] = 0.3 * X[i-1] + 0.7 * torch.randn(64)
            
            y = torch.zeros(num_samples, dtype=torch.long)
            for i in range(num_samples):
                pattern = i % 4
                if pattern == 0:
                    y[i] = (X[i, :16].sum() > 0).long()
                elif pattern == 1:
                    y[i] = (X[i, 16:32].sum() > 0).long() + 1
                elif pattern == 2:
                    y[i] = (X[i, 32:48].sum() > 0).long() + 2
                else:
                    y[i] = 3
        else:
            # Simple patterns
            num_samples = 3000
            X = torch.randn(num_samples, 64)
            y = torch.zeros(num_samples, dtype=torch.long)
            
            for i in range(num_samples):
                feature_sum = X[i, :32].sum()
                if feature_sum > 1:
                    y[i] = 0
                elif feature_sum > 0:
                    y[i] = 1
                elif feature_sum > -1:
                    y[i] = 2
                else:
                    y[i] = 3
        
        # Split: 60% train, 20% val, 20% test
        indices = torch.randperm(num_samples)
        train_idx = indices[:int(0.6 * num_samples)]
        val_idx = indices[int(0.6 * num_samples):int(0.8 * num_samples)]
        test_idx = indices[int(0.8 * num_samples):]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def run_comparison(self):
        """Run comprehensive comparison of all approaches"""
        print("="*60)
        print("🔬 COMPREHENSIVE DRN FIX COMPARISON")
        print("="*60)
        
        trainer = ImprovedTrainer(self.device)
        
        for dataset_type in ['temporal', 'simple']:
            print(f"\n📊 Testing on {dataset_type.upper()} dataset")
            print("-"*40)
            
            # Create datasets
            X_train, y_train, X_val, y_val, X_test, y_test = self.create_datasets(dataset_type)
            
            # Augment training data
            X_train_aug, y_train_aug = trainer.augment_data(X_train, y_train, augmentation_factor=2)
            
            results = {}
            
            # Test Original DRN
            print("\n1️⃣ Original DRN (baseline):")
            try:
                from population_experiment import PopulationDRN
                original_model = PopulationDRN(input_size=64, num_classes=4)
                original_model = trainer.train_with_early_stopping(
                    original_model, X_train, y_train, X_val, y_val, epochs=50
                )
                test_acc = trainer.evaluate(original_model, X_test, y_test)
                results['original'] = {
                    'test_acc': test_acc,
                    'history': trainer.training_history.copy()
                }
                print(f"   Test Accuracy: {test_acc:.1f}%")
            except Exception as e:
                print(f"   Failed: {e}")
                results['original'] = {'test_acc': 0, 'history': defaultdict(list)}
            
            # Test Regularized DRN
            print("\n2️⃣ Regularized DRN (Option 1):")
            reg_model = RegularizedDRN(input_size=64, num_classes=4, dropout_rate=0.5)
            trainer.training_history = defaultdict(list)
            reg_model = trainer.train_with_early_stopping(
                reg_model, X_train_aug, y_train_aug, X_val, y_val, epochs=50
            )
            test_acc = trainer.evaluate(reg_model, X_test, y_test)
            results['regularized'] = {
                'test_acc': test_acc,
                'history': trainer.training_history.copy()
            }
            print(f"   Test Accuracy: {test_acc:.1f}%")
            
            # Test Simplified DRN
            print("\n3️⃣ Simplified DRN (Option 2):")
            simple_model = SimplifiedDRN(input_size=64, num_classes=4)
            trainer.training_history = defaultdict(list)
            simple_model = trainer.train_with_early_stopping(
                simple_model, X_train, y_train, X_val, y_val, epochs=50
            )
            test_acc = trainer.evaluate(simple_model, X_test, y_test)
            results['simplified'] = {
                'test_acc': test_acc,
                'history': trainer.training_history.copy()
            }
            print(f"   Test Accuracy: {test_acc:.1f}%")
            
            # Test Standard MLP for comparison
            print("\n4️⃣ Standard MLP (reference):")
            try:
                from drn.models.baseline_models import StandardMLP
                mlp_model = StandardMLP(input_size=64, hidden_sizes=[128, 64], output_size=4)
                trainer.training_history = defaultdict(list)
                mlp_model = trainer.train_with_early_stopping(
                    mlp_model, X_train, y_train, X_val, y_val, epochs=50
                )
                test_acc = trainer.evaluate(mlp_model, X_test, y_test)
                results['mlp'] = {
                    'test_acc': test_acc,
                    'history': trainer.training_history.copy()
                }
                print(f"   Test Accuracy: {test_acc:.1f}%")
            except Exception as e:
                print(f"   Failed: {e}")
                results['mlp'] = {'test_acc': 0, 'history': defaultdict(list)}
            
            self.results[dataset_type] = results
        
        # Visualize results
        self._visualize_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _visualize_results(self):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, dataset_type in enumerate(['temporal', 'simple']):
            results = self.results[dataset_type]
            
            # Training curves
            ax = axes[i, 0]
            for model_name, model_results in results.items():
                history = model_results['history']
                if history['train_acc']:
                    ax.plot(history['train_acc'], label=f'{model_name} train', linestyle='-', alpha=0.7)
                    ax.plot(history['val_acc'], label=f'{model_name} val', linestyle='--', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{dataset_type.capitalize()} - Training Curves')
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            
            # Loss curves
            ax = axes[i, 1]
            for model_name, model_results in results.items():
                history = model_results['history']
                if history['val_loss']:
                    ax.plot(history['val_loss'], label=model_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Loss')
            ax.set_title(f'{dataset_type.capitalize()} - Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Test accuracy comparison
            ax = axes[i, 2]
            models = list(results.keys())
            test_accs = [results[m]['test_acc'] for m in models]
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
            bars = ax.bar(models, test_accs, color=colors)
            ax.set_ylabel('Test Accuracy (%)')
            ax.set_title(f'{dataset_type.capitalize()} - Test Performance')
            ax.set_ylim([0, 100])
            
            # Add value labels on bars
            for bar, acc in zip(bars, test_accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.1f}%', ha='center', va='bottom')
            
            # Overfitting analysis
            ax = axes[i, 3]
            overfitting = []
            for model_name, model_results in results.items():
                history = model_results['history']
                if history['train_acc']:
                    final_train = history['train_acc'][-1]
                    overfit = final_train - model_results['test_acc']
                else:
                    overfit = 0
                overfitting.append(overfit)
            
            bars = ax.bar(models, overfitting, color=['red' if o > 10 else 'green' for o in overfitting])
            ax.set_ylabel('Train-Test Gap (%)')
            ax.set_title(f'{dataset_type.capitalize()} - Overfitting')
            ax.axhline(y=10, color='black', linestyle='--', alpha=0.3)
            
            for bar, gap in zip(bars, overfitting):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{gap:.1f}%', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive DRN Fix Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig('drn_fixes_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*60)
        print("📊 FINAL SUMMARY")
        print("="*60)
        
        print("\n🏆 BEST PERFORMERS BY DATASET:")
        for dataset_type in ['temporal', 'simple']:
            results = self.results[dataset_type]
            best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
            print(f"\n{dataset_type.capitalize()}:")
            print(f"  Best Model: {best_model[0]}")
            print(f"  Test Accuracy: {best_model[1]['test_acc']:.1f}%")
            
            # Calculate improvement over original
            original_acc = results['original']['test_acc']
            if original_acc > 0:
                improvement = best_model[1]['test_acc'] - original_acc
                print(f"  Improvement over original: {improvement:+.1f}%")
        
        print("\n💡 KEY FINDINGS:")
        
        # Compare approaches
        temp_results = self.results['temporal']
        simple_results = self.results['simple']
        
        # Check which approach works best
        approaches = ['regularized', 'simplified']
        for approach in approaches:
            if temp_results['original']['test_acc'] > 0:
                temp_imp = temp_results[approach]['test_acc'] - temp_results['original']['test_acc']
                simple_imp = simple_results[approach]['test_acc'] - simple_results['original']['test_acc']
                
                print(f"\n{approach.capitalize()} DRN:")
                print(f"  Temporal improvement: {temp_imp:+.1f}%")
                print(f"  Simple improvement: {simple_imp:+.1f}%")
                
                if temp_imp > 5 and simple_imp > 5:
                    print(f"  ✅ Effective for both dataset types")
                elif temp_imp > 5:
                    print(f"  ✅ Effective for temporal patterns")
                elif simple_imp > 5:
                    print(f"  ✅ Effective for simple patterns")
                else:
                    print(f"  ⚠️ Limited effectiveness")
        
        print("\n📝 RECOMMENDATIONS:")
        
        # Analyze results and provide recommendations
        if temp_results['regularized']['test_acc'] > temp_results['original']['test_acc'] + 10:
            print("• Regularization significantly reduces overfitting on complex patterns")
        
        if simple_results['simplified']['test_acc'] > simple_results['original']['test_acc']:
            print("• Simplification improves generalization on simple tasks")
        
        if 'mlp' in temp_results and temp_results['mlp']['test_acc'] > 0:
            if temp_results['mlp']['test_acc'] > max(temp_results[m]['test_acc'] for m in ['original', 'regularized', 'simplified'] if m in temp_results):
                print("• Consider that standard MLPs may be more suitable for some tasks")
            else:
                print("• DRN variants can outperform standard MLPs with proper regularization")
        
        print("\n✅ Analysis complete! Check 'drn_fixes_comparison.png' for visualizations")


if __name__ == "__main__":
    comparison = ComprehensiveComparison()
    results = comparison.run_comparison()
    
    print("\n" + "="*60)
    print("🎯 CONCLUSION")
    print("="*60)
    print("""
    The comprehensive fixes address DRN's overfitting through:
    
    1. REGULARIZATION: Dropout, noise, and L2 penalties reduce memorization
    2. SIMPLIFICATION: Removing non-essential mechanisms improves generalization  
    3. TRAINING STRATEGY: Early stopping and data augmentation prevent overfitting
    
    Choose the approach based on your specific needs:
    - Complex patterns → Regularized DRN
    - Simple patterns → Simplified DRN
    - Limited data → Focus on training strategy
    """)