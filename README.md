
# A Spectral Approach to Optimal Control of the Fokker–Planck Equation

This repository accompanies the paper:

**A Spectral Approach to Optimal Control of the Fokker–Planck Equation**  
by Dante Kalise, Lucas M. Moschen, Grigorios A. Pavliotis, and Urbain Vaes  
Accepted in *IEEE Control Systems Letters (L-CSS)*, 2025: [https://ieeexplore.ieee.org/document/11015582/](https://ieeexplore.ieee.org/document/11015582/)

The ArXiV version is located [https://arxiv.org/abs/2503.15125](https://ieeexplore.ieee.org/document/11015582/).

To reproduce the examples, check the release with tag lcss2025-v1. 

---

## Overview

This code implements a spectral method for the optimal control of the Fokker–Planck equation based on a Schrödinger operator formulation. 
The framework supports:
- Eigenfunction-based discretization in 1D and 2D
- Analytic, finite-difference, and PDE-based solvers
- Forward–backward optimal control via reduced-order modeling
- Visualization and benchmark scripts for controlled convergence

---

## Repository Structure

```text
.
├── schrodinger_operator.py        # Main class-based solver library
├── example_control.py             # Script: example run with 1–4 control fields
├── main_examples.ipynb            # Reproduces the main results from the paper
├── explore_schrodinger_operators.ipynb  # Experimental notebook for spectral operators
├── experimental_fp_solver.py      # Prototype for Fokker–Planck dynamics (unused)
├── experimental_LQ_solver.py      # Experimental LQR control script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

---

## Setup

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

To run notebooks, you may also use:

```bash
jupyter notebook
```

**Note:** Some solvers require access to [Wolfram Engine](https://www.wolfram.com/engine/) via the `wolframclient` Python interface.

---

## How to Run

To reproduce the key control results (Figure 1 in the paper):

```bash
python example_control.py
```

This generates a plot of the $L^2$ distance between the evolving state and target, under 0 to 4 control terms.

To explore the examples used in the paper:

```bash
jupyter notebook main_examples.ipynb
```

---

## Citation

If this code contributes to your research, please cite:

```bibtex
@article{kalise2025spectralfp,
  author={Kalise, Dante and Moschen, Lucas M. and Pavliotis, Grigorios A. and Vaes, Urbain},
  journal={IEEE Control Systems Letters}, 
  title={A Spectral Approach to Optimal Control of the Fokker–Planck Equation}, 
  year={2025},
  doi={10.1109/LCSYS.2025.3573604}}
```

---

## Notes

- `experimental_fp_solver.py` and `experimental_LQ_solver.py` are prototypes not used in the final results.
- `explore_schrodinger_operators.ipynb` is an exploratory notebook for testing spectral properties and interfaces.
- All computational routines are written in Python using NumPy, SciPy, and optionally Wolfram Language for PDE-based solvers.

