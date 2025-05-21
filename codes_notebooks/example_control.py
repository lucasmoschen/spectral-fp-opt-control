"""
Example: Optimal Control of a 2D Fokker–Planck Equation using a Spectral Method

This script demonstrates the use of the SchrodingerControlSolver with an analytic approximator
to compute the effect of increasing control complexity on the trajectory of a probability density.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib import rcParams

from schorodinger_operator import SchrodingerControlSolver, AnalyticSchrodingerApproximator

# Plot configuration
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Define potential and parameters
a_value = 1
b_value = 0.05

def potential(X, Y):
    return 0.5 * (a_value*X**2 + b_value*Y**2)

approximator = AnalyticSchrodingerApproximator(
    L=(3.0, 15.0),
    N=(500, 500),
    sigma=1.0,
    a=a_value,
    b=b_value,
)

eigeninfo = approximator.solve_eigen(num_eigen=50, derivative=True)

# Define initial condition
def integrand(r):
    return r * np.exp(-1 / (1 - r**2 / 4))

integral_value, _ = quad(integrand, 0, 2)
C = 1 / (2 * np.pi * integral_value)

def rho_0(X, Y):
    r2 = (X + 0.2)**2 + (Y - 0.5)**2
    out = np.zeros_like(r2)
    mask = r2 < 4
    out[mask] = C * np.exp(-1 / (1 - r2[mask] / 4))
    return out

nabla_V = lambda X, Y: (a_value*X, b_value*Y)

# Instantiate solvers for different control configurations
solvers = [SchrodingerControlSolver(
    approximator=approximator,
    num_eigen=50,
    potential=potential,
    rho_0=rho_0,
    kappa=5.0,
    nu=1e-4,
    nabla_V=nabla_V,
    correct_lambda0=True,
    eigeninfo=eigeninfo,
    compute_alpha=alpha_idx
) for alpha_idx in range(1, 5)]

T = 5.0
result_controls = [solver.solve(
    T=T,
    max_iter=500,
    tol=1e-4,
    time_eval=T * np.linspace(0, 1, 1001)**2,
    verbose=True,
    learning_rate_kwargs={'method': 'bb', 'gamma': 0.01},
    inicialization=True
) for solver in solvers]

# Baseline (uncontrolled) run
result = solvers[-1].solve(
    T=T,
    max_iter=200,
    tol=1e-4,
    time_eval=T * np.linspace(0, 1, 501)**4,
    verbose=True,
    control_funcs=[lambda t: np.zeros_like(t)],
    optimise=False
)

# Plotting
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
diff_nocontrol = np.sqrt(np.sum((result['a_vals'] - solvers[-1].a_dag) ** 2, axis=1))
ax.plot(result['time'], diff_nocontrol, linestyle='-', linewidth=3, color=colors[0], label='Uncontrolled')

for i, (res, solver) in enumerate(zip(result_controls, solvers)):
    diff_control = np.sqrt(np.sum((res['a_vals'] - solver.a_dag) ** 2, axis=1))
    ax.plot(res['time'], diff_control, linestyle=linestyles[i % len(linestyles)],
            linewidth=3, color=colors[i+1], label=f'{i+1} control{"s" if i > 0 else ""}')

ax.set_xlabel('Time', fontsize=13)
ax.set_ylabel(r'$||\psi(t) - \psi^\dagger||$', fontsize=13)
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.75, color='0.7', alpha=0.8)
ax.legend(frameon=True, fontsize=12, loc='lower left')
fig.tight_layout()
fig.savefig("example_control_plot.pdf", bbox_inches='tight')
plt.show()
