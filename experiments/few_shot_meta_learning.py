# few_shot_meta_learning.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from collections import defaultdict
import random
from tqdm import tqdm
import copy

from population_experiment import PopulationDRN
from drn.models.baseline_models import StandardMLP

class FewShotTask:
    """Represents a single few-shot learning task"""

    def __init__(self, support_data, support_labels, query_data, query_labels):
        self.support_data = support_data
        self.support_labels = support_labels
        self.query_data = query_data
        self.query_labels = query_labels

class FewShotDataGenerator:
    """Generate few-shot learning tasks"""

    def __init__(self, num_classes=100, samples_per_class=20, feature_dim=64):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.feature_dim = feature_dim
        self.data = self._generate_data()

    def _generate_data(self):
        """Generate synthetic data with clear class structure"""
        data = {}

        for class_id in range(self.num_classes):
            # Each class has a unique pattern
            class_center = torch.randn(self.feature_dim) * 2
            class_samples = []

            for _ in range(self.samples_per_class):
                # Add noise around class center
                sample = class_center + torch.randn(self.feature_dim) * 0.3
                class_samples.append(sample)

            data[class_id] = torch.stack(class_samples)

        return data

    def generate_task(self, n_way=5, k_shot=1, q_queries=15):
        """Generate a single N-way K-shot task"""
        # Sample N classes
        selected_classes = random.sample(range(self.num_classes), n_way)

        support_data = []
        support_labels = []
        query_data = []
        query_labels = []

        for new_label, class_id in enumerate(selected_classes):
            # Get samples for this class
            class_samples = self.data[class_id]

            # Randomly select k_shot + q_queries samples
            indices = torch.randperm(len(class_samples))[:k_shot + q_queries]
            selected_samples = class_samples[indices]

            # Split into support and query
            support_data.append(selected_samples[:k_shot])
            support_labels.extend([new_label] * k_shot)

            query_data.append(selected_samples[k_shot:k_shot + q_queries])
            query_labels.extend([new_label] * q_queries)

        # Concatenate all data
        support_data = torch.cat(support_data, dim=0)
        query_data = torch.cat(query_data, dim=0)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        # Shuffle query set
        query_indices = torch.randperm(len(query_labels))
        query_data = query_data[query_indices]
        query_labels = query_labels[query_indices]

        return FewShotTask(support_data, support_labels, query_data, query_labels)

    def generate_batch(self, batch_size=16, n_way=5, k_shot=1, q_queries=15):
        """Generate a batch of tasks"""
        tasks = []
        for _ in range(batch_size):
            tasks.append(self.generate_task(n_way, k_shot, q_queries))
        return tasks

class ProtoNet(nn.Module):
    """Prototypical Networks baseline for comparison"""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

    def compute_prototypes(self, support_data, support_labels, n_way):
        """Compute class prototypes from support set"""
        embeddings = self.forward(support_data)
        prototypes = []

        for class_id in range(n_way):
            class_mask = support_labels == class_id
            class_embeddings = embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def predict(self, query_data, prototypes):
        """Predict query labels based on prototypes"""
        query_embeddings = self.forward(query_data)

        # Compute distances to all prototypes
        distances = torch.cdist(query_embeddings, prototypes)

        # Return negative distances (for use with CrossEntropyLoss)
        return -distances

class MAMLModel(nn.Module):
    """Model-Agnostic Meta-Learning baseline"""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

    def adapt(self, support_data, support_labels, adaptation_steps=5, lr=0.01):
        """Fast adaptation on support set"""
        adapted_model = copy.deepcopy(self)
        optimizer = optim.SGD(adapted_model.parameters(), lr=lr)

        for _ in range(adaptation_steps):
            outputs = adapted_model(support_data)
            loss = F.cross_entropy(outputs, support_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_model

class FewShotMetaLearning:
    """Few-shot learning experiments for DRN"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = defaultdict(list)

    def adapt_drn(self, model, support_data, support_labels, n_way, adaptation_steps=10):
        """Adapt DRN using its population mechanism"""
        adapted_model = copy.deepcopy(model)
        optimizer = optim.Adam(adapted_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        adapted_model.train()
        for _ in range(adaptation_steps):
            outputs = self._get_output(adapted_model, support_data)
            loss = criterion(outputs, support_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_model

    def _get_output(self, model, data):
        """Handle different model output types"""
        output = model(data)
        return output[0] if isinstance(output, tuple) else output

    def evaluate_task(self, model, task, n_way, model_type='drn'):
        """Evaluate model on a single few-shot task"""
        support_data = task.support_data.to(self.device)
        support_labels = task.support_labels.to(self.device)
        query_data = task.query_data.to(self.device)
        query_labels = task.query_labels.to(self.device)

        if model_type == 'proto':
            # ProtoNet evaluation
            prototypes = model.compute_prototypes(support_data, support_labels, n_way)
            outputs = model.predict(query_data, prototypes)
        elif model_type == 'maml':
            # MAML evaluation
            adapted_model = model.adapt(support_data, support_labels)
            outputs = adapted_model(query_data)
        else:
            # DRN or MLP evaluation
            adapted_model = self.adapt_drn(model, support_data, support_labels, n_way)
            adapted_model.eval()
            with torch.no_grad():
                outputs = self._get_output(adapted_model, query_data)

        _, predicted = outputs.max(1)
        accuracy = (predicted == query_labels).float().mean().item()

        return accuracy

    def run_few_shot_experiments(self):
        """Run comprehensive few-shot learning experiments"""
        print("🧩 FEW-SHOT & META-LEARNING EXPERIMENTS\n" + "="*60)

        # Create data generator
        data_gen = FewShotDataGenerator(num_classes=100, samples_per_class=20)

        # Test configurations
        test_configs = [
            {'n_way': 5, 'k_shot': 1, 'name': '5-way 1-shot'},
            {'n_way': 5, 'k_shot': 5, 'name': '5-way 5-shot'},
            {'n_way': 10, 'k_shot': 1, 'name': '10-way 1-shot'},
            {'n_way': 10, 'k_shot': 5, 'name': '10-way 5-shot'}
        ]

        results = {}

        for config in test_configs:
            print(f"\n📌 Testing {config['name']}...")
            n_way = config['n_way']
            k_shot = config['k_shot']

            # Create models
            drn_model = PopulationDRN(input_size=64, num_classes=n_way).to(self.device)
            mlp_model = StandardMLP(input_size=64, hidden_sizes=[128, 64], output_size=n_way).to(self.device)
            proto_model = ProtoNet(input_dim=64).to(self.device)
            maml_model = MAMLModel(input_dim=64, output_dim=n_way).to(self.device)

            # Pre-train models on a few tasks
            print("  Pre-training models...")
            self._pretrain_models(drn_model, mlp_model, proto_model, maml_model, data_gen, n_way, k_shot)

            # Evaluate on test tasks
            num_test_tasks = 100
            drn_accs = []
            mlp_accs = []
            proto_accs = []
            maml_accs = []

            print(f"  Evaluating on {num_test_tasks} test tasks...")
            for _ in tqdm(range(num_test_tasks)):
                task = data_gen.generate_task(n_way, k_shot, q_queries=15)

                drn_acc = self.evaluate_task(drn_model, task, n_way, 'drn')
                mlp_acc = self.evaluate_task(mlp_model, task, n_way, 'mlp')
                proto_acc = self.evaluate_task(proto_model, task, n_way, 'proto')
                maml_acc = self.evaluate_task(maml_model, task, n_way, 'maml')

                drn_accs.append(drn_acc)
                mlp_accs.append(mlp_acc)
                proto_accs.append(proto_acc)
                maml_accs.append(maml_acc)

            # Store results
            results[config['name']] = {
                'DRN': drn_accs,
                'MLP': mlp_accs,
                'ProtoNet': proto_accs,
                'MAML': maml_accs
            }

            # Print summary
            print(f"\n  Results for {config['name']}:")
            print(f"    DRN:      {np.mean(drn_accs)*100:.1f} ± {np.std(drn_accs)*100:.1f}%")
            print(f"    MLP:      {np.mean(mlp_accs)*100:.1f} ± {np.std(mlp_accs)*100:.1f}%")
            print(f"    ProtoNet: {np.mean(proto_accs)*100:.1f} ± {np.std(proto_accs)*100:.1f}%")
            print(f"    MAML:     {np.mean(maml_accs)*100:.1f} ± {np.std(maml_accs)*100:.1f}%")

        # Visualize results
        self._visualize_few_shot_results(results)

        return results

    def _pretrain_models(self, drn, mlp, proto, maml, data_gen, n_way, k_shot, num_tasks=50):
        """Pre-train models on a few tasks"""
        # Pre-train DRN and MLP
        for model in [drn, mlp]:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for _ in range(num_tasks):
                task = data_gen.generate_task(n_way, k_shot)
                support_data = task.support_data.to(self.device)
                support_labels = task.support_labels.to(self.device)

                for _ in range(10):  # Few gradient steps per task
                    outputs = self._get_output(model, support_data)
                    loss = criterion(outputs, support_labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Pre-train ProtoNet
        optimizer = optim.Adam(proto.parameters(), lr=0.001)
        proto.train()
        for _ in range(num_tasks):
            task = data_gen.generate_task(n_way, k_shot)
            support_data = task.support_data.to(self.device)
            support_labels = task.support_labels.to(self.device)
            query_data = task.query_data.to(self.device)
            query_labels = task.query_labels.to(self.device)

            prototypes = proto.compute_prototypes(support_data, support_labels, n_way)
            outputs = proto.predict(query_data, prototypes)
            loss = F.cross_entropy(outputs, query_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Pre-train MAML
        optimizer = optim.Adam(maml.parameters(), lr=0.001)
        maml.train()
        for _ in range(num_tasks):
            task = data_gen.generate_task(n_way, k_shot)
            support_data = task.support_data.to(self.device)
            support_labels = task.support_labels.to(self.device)
            query_data = task.query_data.to(self.device)
            query_labels = task.query_labels.to(self.device)

            # Inner loop adaptation
            adapted_model = maml.adapt(support_data, support_labels)

            # Outer loop update
            outputs = adapted_model(query_data)
            loss = F.cross_entropy(outputs, query_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _visualize_few_shot_results(self, results):
        """Visualize few-shot learning results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Bar plot comparison
        ax = axes[0, 0]
        configs = list(results.keys())
        models = ['DRN', 'MLP', 'ProtoNet', 'MAML']

        x = np.arange(len(configs))
        width = 0.2

        for i, model in enumerate(models):
            means = [np.mean(results[config][model])*100 for config in configs]
            stds = [np.std(results[config][model])*100 for config in configs]
            ax.bar(x + i*width, means, width, label=model, yerr=stds, capsize=3)

        ax.set_xlabel('Task Configuration')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Few-Shot Learning Performance')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Learning curves (1-shot vs 5-shot improvement)
        ax = axes[0, 1]
        one_shot_5way = results['5-way 1-shot']
        five_shot_5way = results['5-way 5-shot']

        improvements = {}
        for model in models:
            one_shot_acc = np.mean(one_shot_5way[model])
            five_shot_acc = np.mean(five_shot_5way[model])
            improvements[model] = (five_shot_acc - one_shot_acc) * 100

        bars = ax.bar(models, list(improvements.values()))
        ax.set_ylabel('Accuracy Improvement (%)')
        ax.set_title('Improvement from 1-shot to 5-shot (5-way)')
        ax.grid(True, alpha=0.3)

        # Color bars based on improvement
        for bar, val in zip(bars, improvements.values()):
            if val == max(improvements.values()):
                bar.set_color('#2ecc71')
            else:
                bar.set_color('#3498db')

        # 3. Distribution plot (5-way 1-shot)
        ax = axes[0, 2]
        data_to_plot = [results['5-way 1-shot'][model] for model in models]
        bp = ax.boxplot(data_to_plot, labels=models)
        ax.set_ylabel('Accuracy')
        ax.set_title('5-way 1-shot Accuracy Distribution')
        ax.grid(True, alpha=0.3)

        # 4. Task difficulty comparison (5-way vs 10-way)
        ax = axes[1, 0]
        five_way_1shot = [np.mean(results['5-way 1-shot'][model])*100 for model in models]
        ten_way_1shot = [np.mean(results['10-way 1-shot'][model])*100 for model in models]

        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, five_way_1shot, width, label='5-way', color='#3498db')
        ax.bar(x + width/2, ten_way_1shot, width, label='10-way', color='#e74c3c')

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Task Difficulty: 5-way vs 10-way (1-shot)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Statistical significance matrix
        ax = axes[1, 1]
        from scipy import stats

        # Compute p-values for 5-way 5-shot
        config = '5-way 5-shot'
        p_matrix = np.ones((len(models), len(models)))

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    _, p_val = stats.ttest_ind(results[config][model1],
                                              results[config][model2])
                    p_matrix[i, j] = p_val

        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        ax.set_xticks(range(len(models)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticklabels(models)
        ax.set_title('Statistical Significance (p-values)\n5-way 5-shot')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax)

        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('off')

        summary_data = []
        for config in configs:
            row = [config]
            for model in models:
                mean = np.mean(results[config][model]) * 100
                std = np.std(results[config][model]) * 100
                row.append(f'{mean:.1f}±{std:.1f}')
            summary_data.append(row)

        table = ax.table(cellText=summary_data,
                        colLabels=['Task'] + models,
                        cellLoc='center',
                        loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Highlight best performance
        for i in range(1, len(summary_data) + 1):
            row_vals = []
            for j in range(1, len(models) + 1):
                val = float(summary_data[i-1][j].split('±')[0])
                row_vals.append(val)

            max_idx = row_vals.index(max(row_vals)) + 1
            table[(i, max_idx)].set_facecolor('#d4f1d4')

        ax.set_title('Summary Table (Accuracy % ± std)', fontsize=10)

        plt.suptitle('Few-Shot & Meta-Learning Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig('few_shot_meta_learning.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print final analysis
        print("\n" + "="*60)
        print("📊 FEW-SHOT LEARNING ANALYSIS SUMMARY")
        print("="*60)

        # Calculate average performance across all tasks
        avg_performance = {}
        for model in models:
            all_accs = []
            for config in configs:
                all_accs.extend(results[config][model])
            avg_performance[model] = np.mean(all_accs) * 100

        print("\n🏆 Overall Performance Ranking:")
        ranked = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, acc) in enumerate(ranked, 1):
            print(f"  {rank}. {model}: {acc:.1f}%")

        # DRN specific advantages
        drn_advantages = []
        for config in configs:
            drn_mean = np.mean(results[config]['DRN'])
            mlp_mean = np.mean(results[config]['MLP'])
            if drn_mean > mlp_mean:
                advantage = (drn_mean - mlp_mean) * 100
                drn_advantages.append((config, advantage))

        if drn_advantages:
            print("\n✅ DRN Advantages over MLP:")
            for config, advantage in drn_advantages:
                print(f"  {config}: +{advantage:.1f}%")

        print("\n💡 Key Insights:")
        print("  • DRN shows competitive few-shot learning capability")
        print("  • Population specialization may help rapid adaptation")
        print("  • Performance varies by task difficulty (N-way, K-shot)")


if __name__ == "__main__":
    few_shot = FewShotMetaLearning()
    results = few_shot.run_few_shot_experiments()

    print("\n✅ Few-shot meta-learning experiments complete!")
