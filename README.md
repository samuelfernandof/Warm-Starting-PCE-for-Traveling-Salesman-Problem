# Warm-Starting PCE for Traveling Salesman Problem

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-purple)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-red.svg)](https://arxiv.org/abs/xxxx.xxxxx)

> A quantum optimization framework implementing Pauli Correlation Encoding (PCE) with Goemans-Williamson warm-starting for solving the Traveling Salesman Problem.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/samuelfernandof/Warm-Starting-PCE-for-Traveling-Salesman-Problem.git
cd Warm-Starting-PCE-for-Traveling-Salesman-Problem

# Install dependencies
pip install -r requirements.txt

# Run experiments
python main.py
```

## 📊 Key Results

| Method | Success Rate (p=5) | Optimal at p=4 | Improvement |
|--------|-------------------|----------------|-------------|
| Standard PCE | 26% | 4% | - |
| **Warm-PCE** | **64%** | **60%** | **15× better** |

## 🎯 What This Does

This implementation introduces **Warm-PCE**, an enhanced version of Pauli Correlation Encoding that:

- 🔄 **Reduces qubit requirements** from 16 to 4 qubits for 5-city TSP
- 🎯 **Improves success rates** by 2-15× using classical warm-starting
- 📈 **Scales with circuit depth** unlike standard PCE
- 🧮 **Solves real optimization problems** with current quantum hardware

## 🔬 Algorithm Overview

1. **TSP → QUBO**: Formulate traveling salesman as binary optimization
2. **QUBO → MaxCut**: Transform using auxiliary variables  
3. **PCE Encoding**: Map variables to Pauli string correlations (k=2)
4. **Warm-Start**: Bias optimization using Goemans-Williamson solution
5. **VQE**: Optimize using variational quantum circuits
6. **Extract Solution**: Convert quantum state to TSP route

## 📁 Project Structure

```
├── main.py                 # Main experiment runner
├── qubo_problem.py         # PCE implementation 
├── graph.py               # TSP instance generation
├── plot_solutions.py      # Visualization tools
├── config.py              # Quantum backend setup
└── requirements.txt       # Dependencies
```

## ⚙️ Configuration

Key parameters in `main.py`:

```python
# Problem setup
N = 5                    # Number of cities
k = 2                    # PCE correlation order
p = 5                    # Circuit depth

# Warm-starting
warm_starting = True     # Enable Warm-PCE
epsilon = 0.2           # GW bias strength

# Optimization
optimizer = 'COBYLA'     # Classical optimizer
max_iter = 1000         # Max iterations
```

## 🧪 Running Experiments

### Basic Usage
```bash
python main.py
```

### Parameter Sweep
```python
# Edit main.py to sweep parameters
for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for p in range(1, 6):
        # Run experiment
```

### Comparison Mode
```python
# Compare Warm-PCE vs Standard PCE
results_warm = run_experiment(warm_starting=True, epsilon=0.2)
results_standard = run_experiment(warm_starting=False, epsilon=0.5)
```

## 📈 Performance Analysis

### Success Rate by Circuit Depth
- **p=1**: 28% vs 4% (7× improvement)
- **p=2**: 32% vs 8% (4× improvement) 
- **p=3**: 54% vs 16% (3.4× improvement)
- **p=4**: 60% vs 4% (15× improvement)
- **p=5**: 64% vs 26% (2.5× improvement)

### Approximation Quality
Warm-PCE shows monotonic improvement with circuit depth, while standard PCE plateaus around 96% approximation ratio.

## 🛠️ Installation

### Requirements
- Python 3.8+
- Qiskit 1.0+
- NumPy, Matplotlib, NetworkX
- CVXPY (for Goemans-Williamson)

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Test installation
python -c "import qiskit; print('✅ Ready to go!')"
```

## 🔧 Advanced Usage

### Custom TSP Instances
```python
from graph import create_random_graph

# Generate custom instance
G = create_random_graph(n_cities=6, weight_range=(1, 50), seed=42)
```

### Hardware Execution
```python
# Configure IBM Quantum backend
from config import setup_backend
backend = setup_backend('ibm_quantum_device')
```

### Result Analysis
```python
from plot_solutions import analyze_results

# Generate comprehensive plots
analyze_results(results_dict, save_path='./results/')
```

## 📚 Method Details

### Pauli Correlation Encoding (PCE)
Maps binary variables to quantum correlations:
```
x_i = sgn(⟨Π_i⟩)  where Π_i is a Pauli string
```

### Warm-Starting Objective
```
L_Warm-PCE = Σ W_ij [1 + |c*_i - c*_j|] s_i(θ) s_j(θ) + L_reg
```
Where `c*_i` are Goemans-Williamson solution bits.

### Qubit Efficiency
- **Standard encoding**: 16 qubits for 5-city TSP
- **PCE encoding**: 4 qubits for 5-city TSP
- **Reduction**: 75% fewer qubits required

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{carmo2025warm,
  title={Warm-Starting PCE for Traveling Salesman Problem},
  author={Rafael Sim{\~o}es do Carmo and Renato Gomes dos Reis and Samuel Fernando and Luiz Gustavo Esmenard Arruda and Felipe F. Fanchini},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the PCE framework by [Sciorilli et al. (2025)](https://arxiv.org/abs/xxxx.xxxxx)
- Inspired by Goemans-Williamson approximation algorithm
- Quantum circuits implemented using [Qiskit](https://qiskit.org/)

## 📞 Contact

- **Samuel Fernando** - [@samuelfernandof](https://github.com/samuelfernandof)
- **Project Link** - [https://github.com/samuelfernandof/Warm-Starting-PCE-for-Traveling-Salesman-Problem](https://github.com/samuelfernandof/Warm-Starting-PCE-for-Traveling-Salesman-Problem)

---

⭐ **Star this repo** if you find it useful!
