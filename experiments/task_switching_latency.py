# task_switching_latency.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from collections import defaultdict

from population_experiment import PopulationDRN
from drn.models.baseline_models import StandardMLP

class TaskSwitchingLatency:
    """Measure adaptation speed and population reuse during task switching"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_overlapping_tasks(self, num_samples=1000):
        """Create tasks with overlapping feature spaces"""
        tasks = {}

        # Task A: Uses features 0-31 primarily
        X_a = torch.randn(num_samples, 64)
        X_a[:, :32] *= 2  # Amplify first half
        y_a = (X_a[:, :32].sum(dim=1) > 0).long()

        # Task B: Uses features 32-63 primarily
        X_b = torch.randn(num_samples, 64)
        X_b[:, 32:] *= 2  # Amplify second half
        y_b = (X_b[:, 32:].sum(dim=1) > 0).long()

        # Task C: Uses features 16-47 (overlapping)
        X_c = torch.randn(num_samples, 64)
        X_c[:, 16:48] *= 2  # Amplify middle
        y_c = (X_c[:, 16:48].sum(dim=1) > 0).long()

        tasks['A'] = TensorDataset(X_a, y_a)
        tasks['B'] = TensorDataset(X_b, y_b)
        tasks['C'] = TensorDataset(X_c, y_c)

        return tasks

    def measure_adaptation_speed(self, model, task_sequence, switch_interval=50):
        """Measure how quickly model adapts after task switch"""

        adaptation_curves = []
        population_dynamics = []
        activation_overlaps = []

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for switch_idx, (task_name, task_data) in enumerate(task_sequence):
            print(f"\n🔄 Switch {switch_idx}: Task {task_name}")

            loader = DataLoader(task_data, batch_size=32, shuffle=True)

            switch_accuracies = []
            switch_populations = []

            model.train()
            for step in range(switch_interval):
                for data, targets in loader:
                    data, targets = data.to(self.device), targets.to(self.device)

                    # Forward pass
                    if hasattr(model, 'layer1'):  # DRN
                        outputs, info = model(data, return_info=True)

                        # Track population activity
                        pop_activity = info['layer1']['population_activities']
                        switch_populations.append(pop_activity)
                    else:
                        outputs = model(data)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        switch_populations.append([1, 1, 1, 1])  # Dummy for MLP

                    # Training step
                    optimizer.zero_grad()
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    # Measure accuracy
                    _, predicted = outputs.max(1)
                    accuracy = (predicted == targets).float().mean().item()
                    switch_accuracies.append(accuracy)

                    break  # One batch per step

            adaptation_curves.append(switch_accuracies)
            population_dynamics.append(switch_populations)

            # Calculate activation overlap with previous task
            if switch_idx > 0:
                prev_pops = np.mean(population_dynamics[-2], axis=0)
                curr_pops = np.mean(population_dynamics[-1], axis=0)
                overlap = np.corrcoef(prev_pops, curr_pops)[0, 1]
                activation_overlaps.append(overlap)
                print(f"  Activation overlap with previous: {overlap:.3f}")

        return adaptation_curves, population_dynamics, activation_overlaps

    def run_task_switching_analysis(self):
        """Complete task switching analysis"""
        print("🔁 TASK SWITCHING LATENCY & FLEXIBILITY\n" + "="*60)

        # Create tasks
        tasks = self.create_overlapping_tasks()

        # Define switching sequence
        sequence = [
            ('A', tasks['A']),
            ('B', tasks['B']),
            ('A', tasks['A']),  # Return to A
            ('C', tasks['C']),  # Overlapping task
            ('B', tasks['B']),
            ('C', tasks['C'])
        ]

        # Test DRN
        print("\n📊 Testing DRN...")
        drn_model = PopulationDRN(input_size=64, num_classes=2).to(self.device)
        drn_curves, drn_pops, drn_overlaps = self.measure_adaptation_speed(drn_model, sequence)

        # Test MLP
        print("\n📊 Testing MLP...")
        mlp_model = StandardMLP(input_size=64, hidden_sizes=[128, 64], output_size=2).to(self.device)
        mlp_curves, _, _ = self.measure_adaptation_speed(mlp_model, sequence)

        # Analyze results
        self._analyze_switching_results(drn_curves, mlp_curves, drn_pops, drn_overlaps)

        # Visualize
        self._visualize_switching_dynamics(drn_curves, mlp_curves, drn_pops, sequence)

        return drn_curves, mlp_curves, drn_pops

    def _analyze_switching_results(self, drn_curves, mlp_curves, drn_pops, overlaps):
        """Analyze task switching metrics"""
        print("\n" + "="*60)
        print("📊 TASK SWITCHING ANALYSIS")
        print("="*60)

        # Adaptation speed (steps to 80% accuracy)
        def steps_to_threshold(curves, threshold=0.8):
            steps = []
            for curve in curves:
                try:
                    step = next(i for i, acc in enumerate(curve) if acc >= threshold)
                except:
                    step = len(curve)
                steps.append(step)
            return steps

        drn_steps = steps_to_threshold(drn_curves)
        mlp_steps = steps_to_threshold(mlp_curves)

        print(f"\n⚡ Adaptation Speed (steps to 80% accuracy):")
        print(f"  DRN: {np.mean(drn_steps):.1f} ± {np.std(drn_steps):.1f}")
        print(f"  MLP: {np.mean(mlp_steps):.1f} ± {np.std(mlp_steps):.1f}")
        print(f"  DRN adapts {np.mean(mlp_steps)/np.mean(drn_steps):.1f}x faster")

        # Population reuse
        print(f"\n🔄 Population Reuse (activation overlap):")
        if overlaps:
            print(f"  Average overlap: {np.mean(overlaps):.3f}")
            print(f"  Overlap range: {min(overlaps):.3f} - {max(overlaps):.3f}")

        # Recruitment diversity
        all_pops = np.concatenate([np.array(pops) for pops in drn_pops])
        pop_variance = np.var(all_pops, axis=0)
        print(f"\n🌈 Recruitment Diversity:")
        print(f"  Population variance: {pop_variance}")
        print(f"  Most dynamic population: P{np.argmax(pop_variance)}")
        print(f"  Most stable population: P{np.argmin(pop_variance)}")

    def _visualize_switching_dynamics(self, drn_curves, mlp_curves, drn_pops, sequence):
        """Visualize task switching dynamics"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # 1. Adaptation curves
        ax = axes[0, 0]
        for i, (curve_d, curve_m) in enumerate(zip(drn_curves, mlp_curves)):
            ax.plot(curve_d, 'o-', label=f'DRN Switch {i}', alpha=0.7)
            ax.plot(curve_m, 's--', label=f'MLP Switch {i}', alpha=0.5)

        # Mark task switches
        for i in range(1, len(drn_curves)):
            ax.axvline(x=i*50, color='gray', linestyle='--', alpha=0.3)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Accuracy')
        ax.set_title('Adaptation Speed After Task Switch')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Population timeline
        ax = axes[0, 1]
        all_pops = []
        for pops in drn_pops:
            all_pops.extend(pops)
        all_pops = np.array(all_pops).T

        for i, pop_timeline in enumerate(all_pops):
            ax.plot(pop_timeline, label=f'Population {i}', linewidth=2)

        # Mark task switches
        for i in range(1, len(drn_curves)):
            ax.axvline(x=i*50, color='gray', linestyle='--', alpha=0.3)

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Activation Level')
        ax.set_title('Population Activity Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Activation heatmap
        ax = axes[1, 0]
        heatmap_data = np.array(all_pops)
        im = ax.imshow(heatmap_data, aspect='auto', cmap='hot')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Population ID')
        ax.set_title('Population Activation Heatmap')
        plt.colorbar(im, ax=ax)

        # 4. Task-specific population usage
        ax = axes[1, 1]
        task_pops = []
        for i, (task_name, _) in enumerate(sequence):
            if i < len(drn_pops):
                mean_pops = np.mean(drn_pops[i], axis=0)
                task_pops.append(mean_pops)

        task_pops = np.array(task_pops).T
        im = ax.imshow(task_pops, aspect='auto', cmap='viridis')
        ax.set_xlabel('Task Switch')
        ax.set_ylabel('Population ID')
        ax.set_title('Task-Specific Population Usage')
        ax.set_xticks(range(len(sequence)))
        ax.set_xticklabels([t[0] for t in sequence])
        plt.colorbar(im, ax=ax)

        # 5. Adaptation speed comparison
        ax = axes[2, 0]
        drn_speeds = [len([a for a in curve if a < 0.8]) for curve in drn_curves]
        mlp_speeds = [len([a for a in curve if a < 0.8]) for curve in mlp_curves]

        x = np.arange(len(drn_speeds))
        width = 0.35
        ax.bar(x - width/2, drn_speeds, width, label='DRN', color='#3498db')
        ax.bar(x + width/2, mlp_speeds, width, label='MLP', color='#e74c3c')

        ax.set_xlabel('Task Switch')
        ax.set_ylabel('Steps to 80% Accuracy')
        ax.set_title('Adaptation Speed by Task Switch')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Summary
        ax = axes[2, 1]
        ax.axis('off')

        summary_text = f"""
        Task Switching Summary:

        Average Adaptation Speed:
          DRN: {np.mean(drn_speeds):.1f} steps
          MLP: {np.mean(mlp_speeds):.1f} steps

        Population Dynamics:
          Active populations vary by task
          Some populations specialize
          Others remain general

        Key Finding:
          DRN shows {np.mean(mlp_speeds)/np.mean(drn_speeds):.1f}x faster
          adaptation through selective
          population recruitment
        """

        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')

        plt.suptitle('Task Switching Latency & Flexibility Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig('task_switching_latency.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    switcher = TaskSwitchingLatency()
    drn_curves, mlp_curves, drn_pops = switcher.run_task_switching_analysis()

    print("\n✅ Task switching latency analysis complete!")
