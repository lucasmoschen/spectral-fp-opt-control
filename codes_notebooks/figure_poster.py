import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import scienceplots
import matplotlib as mpl
import matplotlib.font_manager as fm

from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from matplotlib.gridspec import GridSpec

from scipy.stats import gaussian_kde

plt.style.use('science')

poster_font_path = "/Users/lmm122/Documents/codes/Fonts/ImperialSansText-Regular.ttf"
fm.fontManager.addfont(poster_font_path)
font_prop = fm.FontProperties(fname=poster_font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# Import required classes from your module
from schorodinger_operator import WolframNDEigensystemApproximator, SchrodingerControlSolver, AnalyticSchrodingerApproximator

def big_figure():

    # ---------------------------
    # 1. Set Up the Approximator
    # ---------------------------
    approximator = WolframNDEigensystemApproximator(
        potential_expr="4*x^6 - 12*x^4 + 3*x^2 + 3",
        L=4.0,
        N=5000,       
        sigma=1.0,
        options={"MaxCellMeasure": 0.05}
    )

    def potential(x):
        return (x**2 - 1.5)**2

    # ---------------------------
    # 2. Define the Initial Distribution rho_0
    # ---------------------------
    def _bump_scalar(x):
        """Scalar bump function for integration."""
        return np.exp(-1/(1 - x**2)) if np.abs(x) < 1 else 0.0

    norm, _ = quad(_bump_scalar, -1, 1)
    C = 1 / norm

    def rho_0(x):
        y = np.zeros_like(x, dtype=float)
        mask = np.abs(x-1) < 1
        y[mask] = C * np.exp(-1/(1 - (x[mask]-1)**2))
        return y

    # ---------------------------
    # 3. Define nabla_alpha functions
    # ---------------------------
    nabla_alpha_list = [
        lambda x: np.sin(0.5*x)/(1 + x**4),    # similar to e_1
        lambda x: (1 - x**2) * np.exp(-x**2 / 2) # similar to e_2
    ]

    # ---------------------------
    # 4. Instantiate the Solver
    # ---------------------------
    solver1 = SchrodingerControlSolver(
        approximator=approximator,
        num_eigen=20,
        potential=potential,
        rho_0=rho_0,
        rho_dag=None,
        kappa=1.0,
        nu=1e-4,
        const=1.0,
        nabla_alpha_list=nabla_alpha_list,
        nabla_V=lambda x: 4*(x**2 - 1.5)*x,
        correct_lambda0=True
    )

    # ---------------------------
    # 5. Solve the Control Problem
    # ---------------------------
    T = 1.0
    time_eval = np.linspace(0, T, 1001)
    result_control1 = solver1.solve(
        T=T,
        max_iter=50,
        tol=1e-4,
        time_eval=time_eval,
        verbose=True,
        learning_rate_kwargs={'method': 'bb', 'gamma': 10},
        inicialization=True
    )

    result = solver1.solve(T=T, 
                        max_iter=200, 
                        tol=1e-4, 
                        time_eval=time_eval, 
                        verbose=True, 
                        control_funcs=[lambda t: np.zeros_like(t)], 
                        optimise=False)

    # ---------------------------
    # 6. Generate the Four Figures
    # ---------------------------

    # (A) Initial Distribution
    # Use the defined rho_0 on the domain of interest.
    # Compute the initial distribution using your rho_0 function
    x_vals = solver1.x
    x_target = solver1.x
    mu_initial = rho_0(solver1.x)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('white')

    # Customize axes: show x- and y-axes with minimal style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=25)
    ax.tick_params(axis='y', colors='black', labelsize=25)

    # Optionally, add axis lines at y=0 and x=0 for reference.
    ax.axhline(0, color='gray', linewidth=5, ls='--')
    ax.axvline(x_target.min(), color='gray', linewidth=5, ls='--')

    # Fill the area under the curve with a pastel color (mistyrose)
    ax.fill_between(x_vals, mu_initial, color='#0072B2', alpha=0.4, zorder=1)
    # Plot the outline with a stronger, deeper color (firebrick)
    ax.plot(x_vals, mu_initial, color='#0072B2', linewidth=5, zorder=2, alpha=0.9)

    # Remove axes, ticks, and gridlines for a clean schematic look
    plt.tight_layout(pad=0)
    plt.savefig("plot_initial_distribution.pdf", bbox_inches="tight", pad_inches=0)
    plt.close()

    # (B) Spectral Discretization
    # Plot a bar chart of the first 10 spectral coefficients at t=0.
    # Create the background colormap from 'aliceblue' to 'lightcyan'
    bg_cmap = LinearSegmentedColormap.from_list('bg_cmap_final', ['aliceblue', 'lightcyan'])

    # Extract the spectral coefficients from the result (only first 10 at t=0)
    a_vals = result_control1['a_vals']
    num_modes_to_plot = 6
    indices = np.arange(num_modes_to_plot)
    coeff_initial = a_vals[0, :num_modes_to_plot]  # first 10 coefficients at t=0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('white')
    # Plot the bar chart on top of the gradient background
    ax.bar(indices, coeff_initial, color="#0072B2", edgecolor="black", zorder=1)

    # Set axis labels and ticks with custom tick labels e_0, e_1, ...
    ax.set_xticks(indices)
    ax.set_xticklabels([f"$a_{{{i}}}$" for i in indices], fontsize=25)
    #ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=2)

    # Customize axes: show x- and y-axes with minimal style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=25)
    ax.tick_params(axis='y', colors='black', labelsize=25)

    # Optionally, add axis lines at y=0 and at the left edge (x = -0.5) for reference
    ax.axhline(0, color='gray', linewidth=2, ls='--')
    ax.axvline(-0.5, color='gray', linewidth=2, ls='--')

    plt.tight_layout()
    plt.savefig("plot_spectral_discretization.pdf", bbox_inches="tight")
    plt.close()

    # (C) Optimal Control Intermediary: Grouped Bar Plot
    # Compare the first 10 coefficients at t=0 (without control) and at t=T (with control)
    x_indices = np.arange(num_modes_to_plot)
    coeff_initial = result['a_vals'][-1, :num_modes_to_plot]  # Coefficients at t=T (without control)
    coeff_final   = a_vals[-1, :num_modes_to_plot]   # Coefficients at t=T (with control)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('white')

    # Plot the grouped bars
    ax.bar(x_indices - bar_width/2, coeff_initial, width=bar_width,
        color="#0072B2", edgecolor="black", label="Without Control", zorder=1)
    ax.bar(x_indices + bar_width/2, coeff_final, width=bar_width,
        color="#D55E00", edgecolor="black", label="With Control", zorder=1)

    # Customize axes: show x- and y-axes with minimal style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=25)
    ax.tick_params(axis='y', colors='black', labelsize=25)

    # Optionally, add axis lines at y=0 and at the left edge (x = -0.5) for reference
    ax.axhline(0, color='gray', linewidth=2, ls='--')
    ax.axvline(-0.5, color='gray', linewidth=2, ls='--')

    # Customize the axis labels and ticks
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"$a_{{{i}}}$" for i in x_indices], fontsize=25)
    ax.legend(fontsize=24, frameon=True)
    #ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=2)

    plt.tight_layout()
    plt.savefig("plot_optimal_control_intermediary.pdf", bbox_inches="tight")
    plt.close()

    # (D) Final Distribution: Target vs. Final Controlled
    # Obtain the final controlled distribution and the target distribution.
    # result_control1['psi'][:,-1] is the final controlled state.
    # solver1.rho_infty(solver1.x) is the target distribution.
    # Assume solver1.x is defined and provides the spatial grid.
    # Compute the final controlled distribution and target distribution.
    mu_final = result_control1['psi'][-1, :] * np.sqrt(solver1.rho_infty(x_target))
    mu_target = solver1.rho_infty(x_target)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('white')

    # Customize axes: show x- and y-axes with minimal style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black', labelsize=25)
    ax.tick_params(axis='y', colors='black', labelsize=25)

    # Optionally, add axis lines at y=0 and x=0 for reference.
    ax.axhline(0, color='gray', linewidth=5, ls='--')
    ax.axvline(x_target.min(), color='gray', linewidth=5, ls='--')

    # Fill the area under the curve with a pastel color (mistyrose)
    ax.fill_between(x_vals, mu_final, color='mistyrose', alpha=0.8, zorder=1)

    # Plot the target distribution with a pastel blue outline and fill
    ax.plot(x_target, mu_target, color="#0072B2", linewidth=5, zorder=2, label="Target")
    # Plot the final controlled distribution with a pastel red/orange outline
    ax.plot(x_target, mu_final, color="#D55E00", linewidth=5, zorder=3, label="Final Controlled")

    # Add labels and legend
    ax.legend(fontsize=24, loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig("plot_final_distribution.pdf", bbox_inches="tight")
    plt.close()

def small_figure():

    a_value = 1
    b_value = 0.1

    def potential(X, Y):
        return 0.5 * (a_value*X*X + b_value*Y*Y)

    approximator_normal = AnalyticSchrodingerApproximator(
        L=(3.0, 15.0),
        N=(500, 500),
        sigma=1.0,
        a=a_value,
        b=b_value,
    )

    eigeninfo_normal = approximator_normal.solve_eigen(num_eigen=50, derivative=True)

    def integrand(r):
        return r * np.exp(-1 / (1 - r**2 / 4))

    integral_value, _ = quad(integrand, 0, 2)
    C = 1 / (2 * np.pi * integral_value)

    def rho_0(X, Y):
        r2 = (X + 0.2)**2 + (Y - 0.5)**2  # Shift the center to (0.5, 0.5)
        out = np.zeros_like(r2)
        mask = r2 < 4  # Ensure the support remains within r < 2
        out[mask] = C * np.exp(-1 / (1 - r2[mask] / 4))
        return out

    nabla_V = lambda X, Y: (a_value*X, b_value*Y)

    solver = SchrodingerControlSolver(
                                    approximator=approximator_normal,
                                    num_eigen=50,
                                    potential=potential,
                                    rho_0=rho_0,
                                    rho_dag=None,
                                    kappa=5.0,
                                    nu=1e-4,
                                    nabla_alpha_list=None,
                                    nabla_V=nabla_V,
                                    correct_lambda0=True,
                                    eigeninfo=eigeninfo_normal,
                                    compute_alpha=4
                                    )

    T = 5.0
    result_control = solver.solve(
            T=T, 
            max_iter=100, 
            tol=1e-4, 
            time_eval=T * np.linspace(0, 1, 1001)**2, 
            verbose=True, 
            learning_rate_kwargs={'method': 'bb', 'gamma': 0.01}, 
            inicialization=True)

    result = solver.solve(
                        T=T, 
                        time_eval=T * np.linspace(0, 1, 501)**4, 
                        control_funcs=[lambda t: np.zeros_like(t)], 
                        optimise=False
                        )

    # Global font and tick adjustments for large format
    mpl.rcParams.update({
        'font.size': 22,             # Base font size
        'axes.labelsize': 26,        # Axis label size
        'axes.titlesize': 28,        # Title size
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'lines.linewidth': 3,        # Thicker lines
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'legend.frameon': True
    })

    # Colorblind-friendly palette
    colors = ['#0072B2', '#D55E00']  # Blue and reddish-orange

    fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure

    # Uncontrolled
    diff_nocontrol = np.sqrt(np.sum((result['a_vals'] - solver.a_dag) ** 2, axis=1))
    ax.plot(result['time'], diff_nocontrol, linestyle='-', color=colors[0], label='Uncontrolled', lw=5)

    # Controlled
    diff_control = np.sqrt(np.sum((result_control['a_vals'] - solver.a_dag) ** 2, axis=1),)
    ax.plot(result_control['time'], diff_control, linestyle='--', color=colors[1], label='Controlled', lw=5)

    # Axis labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Distance to equilibrium in $L^2$')

    # Log scale and grid
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Legend
    ax.legend(loc='lower left', fontsize=25, frameon=True)

    # Tight layout
    fig.tight_layout()
    plt.savefig('poster_plot_1.pdf')
    plt.show()

def mean_field_figure():

    # Define potential and gradient
    def V(x, y):
        return 0.25 * ((x**2 - 1)**2 + 2 * y**2)

    def grad_V(x, y):
        dVdx = x * (x**2 - 1)
        dVdy = y
        return np.array([dVdx, dVdy])

    # Sampling initial particles uniformly
    np.random.seed(42)
    n_particles = 30000
    n_visible = 50
    xlim, ylim = [-2, 2], [-2, 2]
    particles = np.random.normal(loc=[1, 1], scale=0.2, size=(n_particles, 2))

    # SDE simulation using Euler-Maruyama
    def simulate(particles, dt, steps, sigma=0.5):
        traj = [particles.copy()]
        for _ in range(steps):
            grad = np.array([grad_V(p[0], p[1]) for p in particles])
            noise = np.random.normal(scale=np.sqrt(dt), size=particles.shape)
            particles = particles - grad * dt + np.sqrt(2 * sigma) * noise
            traj.append(particles.copy())
        return traj

    dt = 0.01
    steps1 = int(5 / dt)

    traj_t1 = simulate(particles, dt, steps1, sigma=0.2)
    #traj_t10 = simulate(traj_t1[-1], dt, steps2 - steps1, sigma=2)

    # Sample from target density e^{-V}/C
    def unnormalized_density(x, y, sigma=1.0):
        return np.exp(-V(x, y) / sigma)

    # Compute normalization constant C
    norm_const, _ = dblquad(unnormalized_density, -5, 5, lambda x: -5, lambda x: 5)
    def target_density(x, y, sigma=1.0):
        return np.exp(-V(x, y) / sigma) / norm_const

    # Sample from target using rejection sampling
    def sample_from_target(n_samples, sigma=1.0):
        samples = []
        count = 0
        while len(samples) < n_samples and count < 100000:
            x_try = np.random.uniform(-2, 2)
            y_try = np.random.uniform(-2, 2)
            u = np.random.uniform(0, 1)
            if u < target_density(x_try, y_try, sigma=sigma) / (1 / norm_const):  # upper bound approx
                samples.append([x_try, y_try])
            count += 1
        return np.array(samples)

    target_samples = sample_from_target(n_particles, sigma=0.2)

    # Plotting
    fig = plt.figure(figsize=(4.5, 6.5))
    gs = GridSpec(3, 2, figure=fig)

    times = [0, 5, 20]
    trajs = [particles, traj_t1[-1], target_samples]
    densities = [particles[:, 0], traj_t1[-1][:, 0], target_samples[:, 0]]

    # Left column: particle plots
    for i, (t, pts) in enumerate(zip(times, trajs)):
        ax = fig.add_subplot(gs[i, 0])
        x = np.linspace(xlim[0], xlim[1], 300)
        y = np.linspace(ylim[0], ylim[1], 300)
        X, Y = np.meshgrid(x, y)
        Z = V(X, Y)
        ax.contourf(X, Y, Z, levels=50, cmap='YlGnBu', alpha=0.6)
        ax.scatter(pts[:n_visible, 0], pts[:n_visible, 1], color='grey', alpha=0.6, s=30)
        mean = pts[:n_visible, :].mean(axis=0)
        for p in pts[:n_visible]:
            g = -grad_V(p[0], p[1])
            ax.add_patch(FancyArrowPatch(posA=p, posB=p + 0.5 * g / np.linalg.norm(g), color='black', arrowstyle='->', mutation_scale=10, alpha=0.4))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel(f'Time t={t}', fontsize=14)
        ax.set_aspect('equal')
        ax.scatter(mean[0], mean[1], color='red', s=30, label='Average')

    # Right column: x-marginals
    for i, d in enumerate(densities):
        ax = fig.add_subplot(gs[i, 1])
        kde = gaussian_kde(d)
        x_vals = np.linspace(-2, 2, 200)
        y_vals = kde(x_vals)
        ax.fill_between(x_vals, 0, y_vals, color='#0072B2', alpha=0.4)
        ax.set_xlim(-2, 2)
        ax.set_ylabel('Marginal Density in $x$', fontsize=14)

    plt.tight_layout(h_pad=0.1)
    plt.show()

if __name__ == '__main__':

    #big_figure()
    #small_figure()
    mean_field_figure()
