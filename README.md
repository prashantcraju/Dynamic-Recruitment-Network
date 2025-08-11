# Dynamic Recruitment Networks (DRN)

This repository contains the code, configurations, and evaluation framework:

> **When Biology Meets Deep Learning: An Empirical Analysis of Population-Based Neural Architectures**  
> [Anonymous Authors – under double-blind review]

---

## 🧠 Overview

Dynamic Recruitment Networks (DRNs) are a biologically-inspired neural architecture that integrates:
- **Population coding**
- **Sparse activation**
- **Recurrent processing**
- **Lateral inhibition**
- **Synaptic competition**
- **Neurotransmitter-based resource constraints**

DRNs dynamically recruit sub-populations of neurons based on input complexity, activating only ~25% of units per forward pass. This model is designed to test whether biological realism improves learning performance across a range of paradigms.

---

## 📌 Key Findings

- Biological mechanisms **engage adaptively** with task complexity (28–31% impact on complex tasks).
- Despite strong training impact, **generalization performance often worsens**, indicating overfitting.
- Removing all biological components:
  - **Improves accuracy** on all tasks  
  - **Reduces parameters** by 36%
- DRNs achieve **62% parameter efficiency**, but suffer:
  - **37% lower robustness**
  - **Catastrophic failure** on tasks with >16 output classes
  - **Poor few-shot learning** capability

These results suggest that **faithful biological replication can be harmful without the appropriate computational context**.

---

## 🧪 Evaluation Protocols

We assess DRNs using seven evaluation paradigms:

| Paradigm | Description |
|----------|-------------|
| Continual Learning | Tests population specialization for mitigating forgetting |
| Few-Shot Adaptation | N-way K-shot learning for rapid adaptation |
| Adversarial Robustness | Resistance to perturbations |
| Noise Resilience | Performance under Gaussian, salt-and-pepper, and occlusion noise |
| Computational Efficiency | Parameters, sparsity, and inference speed |
| Scaling Behavior | Sensitivity to model size and task complexity |
| Lesion Studies | Ablation of biological mechanisms |

---

## 🛠️ Getting Started

### Installation

```bash
git clone https://github.com/anonymous/drn-submission.git
cd drn-submission
pip install -r requirements.txt
