# VQE Multi-basis TSP-MaxCut

A quantum optimization framework implementing Variational Quantum Eigensolver (VQE) with multi-basis encoding to solve the Traveling Salesman Problem (TSP) via conversion to Maximum Cut (MaxCut).

## Overview

This implementation presents a novel approach combining:
- **Multi-basis encoding** (k=2) to reduce qubit requirements
- **TSP-to-MaxCut conversion** via QUBO intermediate representation  
- **Warm-starting** with Goemans-Williamson classical initialization
- **Hybrid post-processing** using bit-flip local search

## Results

- **64%** success rate with warm-starting vs **26%** standard approach
- **98.3%** approximation ratio at optimal parameters
- **6.4×** better win rate in head-to-head comparisons
- Consistent performance scaling with circuit depth (p=1→5)

## Files

| File | Description |
|------|-------------|
| `main.py` | Experiment runner with parameter sweeps |
| `qubo_problem.py` | Core VQE_Multibasis class implementation |
| `graph.py` | Graph generation and classical benchmarks |
| `plot_solutions.py` | Solution interpretation and visualization |
| `config.py` | IBM Quantum configuration |

## Algorithm Flow

1. **TSP Formulation**: Complete graph with N cities, variables x_{i}_{p}
2. **QUBO Conversion**: Penalty method with A=400 
3. **MaxCut Transformation**: Auxiliary node construction
4. **Multi-basis Encoding**: k=2 with Pauli operators {Z,X,Y}
5. **VQE Optimization**: TwoLocal ansatz with COBYLA
6. **Solution Extraction**: Expectation values → bitstring → TSP route

## Usage

```bash
# Basic execution
python main.py

# Install dependencies
pip install -r requirements.txt
```

## Parameters

| Parameter | Description | Optimal Value |
|-----------|-------------|---------------|
| `N` | Number of cities | 5 |
| `k` | Multi-basis encoding | 2 |
| `p` | Circuit layers | 5 |
| `epsilon` | Warm-start regularization | 0.2 |
| `A` | Penalty coefficient | 400 |

## Warm-Starting

The algorithm optionally uses Goemans-Williamson semidefinite programming to initialize the optimization:

```python
# Enable warm-starting
warm_starting=True, epsilon=0.2

# Weight factor enhancement
weight_factor = 1 + |c*_i - c*_j|
```

## Multi-basis Encoding

Reduces qubit count by encoding multiple binary variables per physical qubit:
- **Variables**: m = (N-1)² for TSP
- **Qubits**: n such that C(n,k) × |Pauli_ops| ≥ m
- **Efficiency**: Significant reduction for moderate N

## Dependencies

- `qiskit >= 1.0.0` - Quantum computing framework
- `qiskit-ibm-runtime >= 0.20.0` - IBM Quantum access
- `qiskit-algorithms >= 0.3.0` - VQE implementation
- `qiskit-optimization >= 0.6.0` - QUBO formulation
- `networkx >= 3.0` - Graph manipulation
- `numpy >= 1.24.0` - Numerical computing
- `matplotlib >= 3.7.0` - Visualization
- `cvxpy >= 1.3.0` - Convex optimization
- `pulp >= 2.7.0` - Linear programming

## Experimental Setup

- **Graph type**: Complete weighted graphs
- **Weight range**: [10, 100]
- **Seeds**: 10 independent runs per configuration
- **Backend**: AerSimulator with optional IBM hardware
- **Optimizer**: COBYLA with 1000 max iterations

- ## Performance Analysis

### Methodology Comparison

| Method | Description |
|--------|-------------|
| **Standard PCE** | Multi-basis VQE encoding (k=2)<br>Parameter: ε = 0.5<br>No classical initialization<br>Random parameter start |
| **Warm-PCE** | Multi-basis VQE encoding (k=2)<br>Parameter: ε = 0.2<br>Goemans-Williamson warm-start<br>Classical bias integration |

### Key Results

| Metric | PCE (ε=0.5) | Warm-PCE (ε=0.2) | Improvement |
|--------|-------------|-------------------|-------------|
| **Approximation Ratio (p=5)** | 0.967 | **0.983** | +1.7% |
| **Success Rate at p=5** | 26% | **64%** | +146% |
| **Best-of-10 Win Rate** | 5/50 (10%) | **32/50 (64%)** | 6.4× better |
| **Optimal ε Parameter** | - | **ε = 0.2** | E/E_nc ≈ 0.92 |

### Key Observations

- **Circuit depth scaling** (p=1→5) shows consistent performance improvement for both methods, with warm-starting providing exponential gains in success rates.

- **Regularization parameter ε** exhibits clear optimum at 0.2, indicating balanced classical-quantum integration in the warm-starting approach.

- **Multi-basis encoding** (k=2) successfully reduces qubit requirements while maintaining solution quality for TSP instances up to N=5 cities.

- **Goemans-Williamson initialization** provides consistent advantage across all circuit depths, with most dramatic improvements at higher layer counts.

---

**Experimental Setup:** N=5 cities, k=2 multi-basis encoding, p∈{1,2,3,4,5} circuit layers, weight range (10,100), 10 seeds per configuration, COBYLA optimizer.

## Citation

```bibtex
@article{vqe_multibasis_tsp,
  title={VQE Multi-basis Encoding for TSP-MaxCut Problems},
  author={Your Name},
  year={2025}
}
```
