# Federated Learning for Intelligent Resource Allocation in Heterogeneous 5G/6G Networks

> Official implementation accompanying our research on Federated Learning for intelligent radio resource allocation in next-generation wireless networks.

---

## Overview

This repository presents a simulation framework for **Federated Learning (FL)**-based resource allocation in heterogeneous **5G/6G networks**. The framework enables multiple distributed gNodeBs (gNBs) to collaboratively train a global model without sharing raw user data, thereby preserving privacy while improving overall network performance.

The simulator evaluates the proposed FL framework against conventional centralized machine learning approaches using realistic wireless communication scenarios and multiple network performance metrics.

---

## Key Features

- Federated Learning using **FedAvg**
- Privacy-preserving distributed model training
- Adaptive client participation
- Dynamic resource allocation
- Heterogeneous 5G/6G network simulation
- Performance visualization
- Automated result generation
- Ablation study support

---

# Repository Structure

```
FL-PAPER/
│
├── figures/
│   ├── communication.png
│   ├── fl_convergence.png
│   ├── latency.png
│   ├── packetloss.png
│   ├── performance_summary.png
│   ├── prediction_accuracy.png
│   ├── spectral.png
│   └── throughput.png
│
├── output/
│   ├── ablation.csv
│   ├── communication.csv
│   ├── performance.csv
│   ├── prediction.csv
│   └── scalability.csv
│
├── fl_simulator.py
├── module1_network.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Simulation Workflow

The simulator performs the following steps:

1. Initialize heterogeneous 5G/6G network.
2. Deploy multiple gNB clients.
3. Generate synthetic traffic.
4. Perform local model training.
5. Aggregate local models using Federated Averaging (FedAvg).
6. Update the global model.
7. Evaluate network performance.
8. Generate plots and CSV reports.

---

# Performance Metrics

The framework evaluates:

- Throughput (Mbps)
- End-to-End Latency (ms)
- Packet Loss (%)
- Spectral Efficiency (bps/Hz)
- Prediction Accuracy
- Communication Overhead
- Federated Learning Convergence
- Scalability Analysis

---

# Output Files

After execution, the simulator automatically generates:

## Figures (`figures/`)

| File | Description |
|------|-------------|
| throughput.png | Throughput comparison |
| latency.png | Network latency |
| packetloss.png | Packet loss comparison |
| spectral.png | Spectral efficiency |
| communication.png | Communication overhead |
| fl_convergence.png | FL convergence curve |
| prediction_accuracy.png | Prediction accuracy |
| performance_summary.png | Overall performance summary |

---

## CSV Results (`output/`)

| File | Description |
|------|-------------|
| performance.csv | Overall performance metrics |
| communication.csv | Communication overhead |
| prediction.csv | Prediction accuracy |
| scalability.csv | Scalability evaluation |
| ablation.csv | Ablation study results |

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/FL-PAPER.git
```

Move into the project directory

```bash
cd FL-PAPER
```

Install the required Python packages

```bash
pip install -r requirements.txt
```

---

# Running the Simulation

Execute

```bash
python fl_simulator.py
```

The simulator will automatically:

- Generate network traffic
- Train the federated model
- Produce evaluation metrics
- Save CSV files
- Generate all figures

---

# Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Learning Paradigm | Federated Learning |
| Aggregation Method | FedAvg |
| Network Type | 5G/6G |
| Resource Allocation | Dynamic |
| Client Selection | Adaptive |
| Evaluation Metrics | Throughput, Latency, Packet Loss, Spectral Efficiency |

---

# Example Results

The proposed framework demonstrates improvements over centralized learning in terms of:

- Higher network throughput
- Lower communication latency
- Reduced packet loss
- Better spectral efficiency
- Faster convergence
- Lower communication overhead

Detailed numerical results can be found in the generated CSV files and corresponding plots.

---

# License

This project is released under the MIT License.

---

# Author

**Prajwal Gupta**

Department of Computer Science and Engineering

Research Area:
- Federated Learning
- Wireless Networks
- 5G/6G Communications
- Resource Allocation
- Edge Intelligence

---

## Acknowledgement

This repository accompanies our IEEE research work on Federated Learning for intelligent resource allocation in heterogeneous wireless networks. The implementation is intended to support reproducibility and further research in privacy-preserving edge intelligence.
