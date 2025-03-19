# Spectral Optimal Control of Fokker–Planck Equations

This repository contains code and supplementary materials related to our research on a spectral optimal control framework for Fokker–Planck equations via a ground state transformation. The approach transforms the Fokker–Planck operator into a Schrödinger operator, enabling effective control strategies to target slow-decaying modes and improve convergence.

## Repository Structure

```
.
├── codes_notebooks
│   ├── fp-solver.py                   # Solver for Fokker–Planck equation on bounded domains using Legendre polynomials
│   ├── schrodinger_operator.py        # Schrödinger optimal control solver for the real line and plane
│   ├── optimal_control_problem.ipynb  # Notebook: Optimal control plots and studies (1D & 2D)
│   └── schrodinger_operator.ipynb       # Notebook: Detailed study of the Schrödinger operator
├── images                           # Figures and plots from the experiments
└── notes                            # Research notes, derivations, and additional documentation
```

## Getting Started

1. **Environment Setup:**  
   Ensure you have Python 3 installed along with necessary packages (e.g., NumPy, SciPy, matplotlib, Jupyter). 
   Consider installing the libraries through `enviroment.yml` file.

2. **Running the Code:**  
   - **Bounded Domain Solver:** Run `fp-solver.py` to solve the Fokker-Planck equation on a bounded domain.
   - **Schrodinger Operator Solver:** Run `schrodinger_operator.py` for the optimal control solver on unbounded domains (real line and plane).

3. **Exploring Notebooks:**  
   Launch Jupyter Notebook in the `codes_notebooks` folder and open:
   - `optimal_control_problem.ipynb` for optimal control experiments.
   - `schrodinger_operator.ipynb` for an in-depth study of the operator.

## About

This project implements a spectral optimal control framework for Fokker–Planck equations, emphasizing control strategies that accelerate convergence toward the steady state. The transformation to a Schrödinger operator enables efficient eigenfunction-based methods.

If you have any questions or feedback, feel free to open an issue or contact the corresponding author.