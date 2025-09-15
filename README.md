```markdown
# Warm-Starting PCE for Traveling Salesman Problem

A quantum optimization framework implementing **Pauli Correlation Encoding (PCE)** with Goemans-Williamson warm-starting to solve the Traveling Salesman Problem (TSP) via QUBO-to-MaxCut conversion.

## Overview

This implementation presents a novel approach combining:
- **Pauli Correlation Encoding (PCE)** with k=2 correlations to reduce qubit requirements
- **Warm-starting strategy** using Goemans-Williamson randomized rounding bias
- **TSP-to-MaxCut conversion** via QUBO intermediate representation  
- **Hybrid post-processing** using classical bit-swap local search

## Key Innovation

**Warm-PCE** extends standard PCE by incorporating a bias term derived from the Goemans-Williamson (GW) algorithm solution, steering the optimization toward improved approximation ratios while maintaining the qubit efficiency of PCE.

## Results

- **28-64%** success rate with Warm-PCE vs **4-26%** standard PCE
- **2×-15×** improvement in finding optimal solutions depending on circuit depth
- **36-41/50 graphs** won by Warm-PCE in best-of-10 comparisons at p=3,4
- Consistent performance improvement with circuit depth (p=1→5)

## Algorithm Flow

1. **TSP QUBO Formulation**: Binary variables x_{i,t} with constraints
2. **QUBO-to-MaxCut Transformation**: Standard auxiliary node method
3. **PCE Encoding**: Map binary variables to Pauli string correlations
4. **Warm-Start Bias**: Incorporate GW solution with regularization ε
5. **VQE Optimization**: TwoLocal ansatz with COBYLA optimizer
6. **Solution Extraction**: Pauli expectations → binary assignment → TSP route

## Warm-PCE Objective Function

```
L_Warm-PCE(θ) = Σ_{(i,j)∈E} W_ij [1 + |c̃*_i - c̃*_j|] s_i(θ) s_j(θ) + L^(reg)
```

Where:
- `s_i(θ) = tanh(α⟨Π_i⟩)`: smoothed Pauli correlation
- `c̃*_i`: regularized GW bit for variable i
- `ε ∈ (0,0.5)`: warm-start strength parameter

## Usage

```bash
# Run experiments with Warm-PCE
python main.py --warm_starting=True --epsilon=0.2

# Run standard PCE for comparison  
python main.py --warm_starting=False

# Install dependencies
pip install -r requirements.txt
```

## Parameters

| Parameter | Description | Paper Value |
|-----------|-------------|-------------|
| `N` | Number of cities | 5 |
| `k` | PCE correlation order | 2 |
| `p` | Circuit layers | 1-5 |
| `epsilon` | GW bias regularization | 0.2 |
| `alpha` | PCE smoothing parameter | - |

## PCE Encoding Efficiency

For k=2 correlations with n qubits:
- **Variables encoded**: m = ½n(n-1)(n-2)
- **TSP variables needed**: (N-1)² = 16 for N=5
- **Qubits required**: 4 (vs 16 in one-hot encoding)
- **Compression ratio**: 4× reduction

## Goemans-Williamson Integration

The warm-start bias up-weights edges whose endpoints are assigned to different subsets by the GW solution:

```python
# GW bit regularization
c̃*_i = {
    ε,      if c*_i < ε
    c*_i,   if ε ≤ c*_i ≤ 1-ε  
    1-ε,    if c*_i > 1-ε
}
```

## Experimental Results

### Methodology Comparison

| Method | Description |
|--------|-------------|
| **PCE** | Standard Pauli Correlation Encoding<br>No classical initialization<br>ε = 0.5 (neutral bias) |
| **Warm-PCE** | PCE with GW warm-starting<br>ε = 0.2 (optimal bias)<br>Classical solution guidance |

### Performance Metrics

| Metric | PCE | Warm-PCE | Improvement |
|--------|-----|----------|-------------|
| **Success Rate (p=1)** | 4% | **28%** | 7× better |
| **Success Rate (p=5)** | 26% | **64%** | 2.5× better |
| **Mean Approximation (p=5)** | ~0.96 | **~0.98** | +2% points |
| **Best-of-10 Wins (p=4)** | 4/50 | **41/50** | 10× better |

### Key Findings

- **Monotonic improvement**: Warm-PCE approximation ratio increases with circuit depth p, while PCE remains flat
- **Optimal regularization**: ε = 0.2 provides best balance between GW guidance and search flexibility
- **Scalability**: Method shows promise for larger instances due to PCE's polynomial qubit reduction

## Dependencies

```
qiskit >= 1.0.0
qiskit-algorithms >= 0.3.0
networkx >= 3.0
numpy >= 1.24.0
matplotlib >= 3.7.0
cvxpy >= 1.3.0        # For Goemans-Williamson SDP
```

## Citation

```bibtex
@article{carmo2025warm,
  title={Warm-Starting PCE for Traveling Salesman Problem},
  author={Rafael Simões do Carmo and Renato Gomes dos Reis and Samuel Fernando and Luiz Gustavo Esmenard Arruda and Felipe F. Fanchini},
  journal={arXiv preprint},
  year={2025}
}
```

## Files Structure

```
├── main.py                 # Experimental runner
├── qubo_problem.py         # PCE implementation (VQE_Multibasis class)  
├── graph.py               # TSP instance generation
├── plot_solutions.py      # Results visualization
├── config.py              # Quantum backend configuration
└── requirements.txt       # Dependencies
```

## Acknowledgments

This work extends the Pauli Correlation Encoding framework introduced by Sciorilli et al. (2025) by incorporating warm-starting strategies for improved performance on combinatorial optimization problems.
```
