#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from abc import ABC, abstractmethod

from wolframclient.language import wlexpr
from wolframclient.evaluation import WolframLanguageSession

class BaseOperatorApproximator(ABC):
    def __init__(self, L=10.0, N=256, sigma=1.0,
                 potential_func=None, potential_expr=None, options=None):
        """
        Parameters:
          L             : half-length of the domain (domain: [-L, L])
          N             : number of spatial grid points
          sigma         : parameter in H = -sigma * Δ + W(x)
          potential_func: callable returning W(x) for an array of x values.
          potential_expr: a string representing W(x) (e.g. for Wolfram Language).
          options       : additional options.
          
        At least one of potential_func or potential_expr must be provided.
        """
        self.L = L
        self.N = N
        self.sigma = sigma
        self.options = options if options is not None else {}
        self.x = np.linspace(-L, L, N)
        self.dx = self.x[1] - self.x[0]
        if potential_func is None and potential_expr is None:
            raise ValueError("Provide at least potential_func or potential_expr.")
        self.potential_func = potential_func
        self.potential_expr = potential_expr

    @abstractmethod
    def solve_eigen(self, num_eigen=10, derivative=False):
        """Compute the lowest num_eigen eigenpairs."""
        pass

class WolframNDEigensystemApproximator(BaseOperatorApproximator):
    def __init__(self, kernel_path='/Applications/Wolfram.app/Contents/MacOS/WolframKernel', L=10.0, N=256, sigma=1.0,
                 potential_expr=None, potential_func=None, options=None):
        """
        Parameters:
          kernel_path   : path to the WolframKernel executable.
          potential_expr: a string representing W(x) in Wolfram Language.
          (potential_func is accepted for uniformity but not used here.)
        """
        super().__init__(L=L, N=N, sigma=sigma,
                         potential_func=potential_func, potential_expr=potential_expr,
                         options=options)
        self.kernel_path = kernel_path
        if self.potential_expr is None:
            raise ValueError("WolframNDEigensystemApproximator requires potential_expr.")

    def solve_eigen(self, num_eigen=10, derivative=False):
        
        session = WolframLanguageSession(kernel=self.kernel_path)
        MaxCellMeasure = self.options.get("MaxCellMeasure", 0.5)
        
        wolfram_script = f"""
        Module[{{eigenvalues, eigenfunctions, pts}},
          Clear[u];
          (* pts is a flat list of evaluation points *)
          pts = Subdivide[-{self.L+self.dx}, {self.L+self.dx}, {self.N} + 1];
          
          (* Compute eigenvalues and eigenfunctions *)
          {{eigenvalues, eigenfunctions}} = 
            NDEigensystem[
              {{-{self.sigma}*D[u[x], {{x, 2}}] + ({self.potential_expr})*u[x], u[-{self.L+self.dx}] == 0, u[{self.L+self.dx}] == 0}},
              u[x],
              {{x, -{self.L+self.dx}, {self.L+self.dx}}},
              {num_eigen},
              Method -> {{"Eigensystem" -> "Direct",
                          "PDEDiscretization" -> {{"FiniteElement", {{"MeshOptions" -> {{"MaxCellMeasure" -> {MaxCellMeasure}}}
                                                                       }}
                                                    }}
                        }}
            ];
        
          (* Evaluate each eigenfunction at the points in pts *)
          (* This produces a list of two lists, one per eigenfunction *)
        
          values = Table[N[eigenfunctions[[i]][x]], {{i, Length[eigenfunctions]}}, {{x, pts}}];

          (* Return both eigenvalues and the evaluated eigenfunction values *)
          {{eigenvalues, eigenfunctions, values}}
        ]
        """
        
        result = session.evaluate(wlexpr(wolfram_script))
        eigenvalues = np.array(result[0], dtype=float)
        eigfunc_matrix = np.array([[value.head for value in eig] for eig in result[2]]).T

        session.terminate()

        if derivative:
            eigfunc_diffs = (eigfunc_matrix[2:, :] - eigfunc_matrix[:-2, :]) / (2*self.dx)
            return eigenvalues, eigfunc_matrix[1:-1], eigfunc_diffs
        
        return eigenvalues, eigfunc_matrix[1:-1]

class FiniteDifferenceApproximator(BaseOperatorApproximator):
    def build_operator(self):
        # This approximator relies on a callable potential.
        if self.potential_func is None:
            raise ValueError("FiniteDifferenceApproximator requires potential_func.")
        W = self.potential_func(self.x)
        main_diag = -2.0 * np.ones(self.N) / self.dx**2
        off_diag = np.ones(self.N - 1) / self.dx**2
        laplacian = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1]).toarray()
        H = -self.sigma * laplacian + np.diag(W)
        return H

    def solve_eigen(self, num_eigen=10, derivative=False):
        H = self.build_operator()
        eigvals, eigvecs = eigh(H)
        return eigvals[:num_eigen], eigvecs[:, :num_eigen]

class SchrodingerSolver:
    def __init__(self, approximator, psi0, num_eigen=10):
        """
        Parameters:
          approximator: an instance of a subclass of BaseOperatorApproximator.
          psi0: initial condition. It should be either a callable (function of x) or an array of initial values f(x,0).
          num_eigen: number of eigenfunctions to use in the spectral expansion.
        """
        self.approximator = approximator
        self.x = approximator.x
        self.dx = approximator.dx
        self.eigvals, self.eigfuncs = self.approximator.solve_eigen(num_eigen)
        self.set_initial_condition(psi0)

    def set_initial_condition(self, f0):
        if callable(f0):
            f0_arr = f0(self.x)
        else:
            f0_arr = np.asarray(f0)
            if f0_arr.shape != self.x.shape:
                raise ValueError("The initial condition array must have the same shape as the spatial grid.")

        self.psi0 = f0_arr
        self.a0 = np.dot(np.conjugate(self.eigfuncs).T, f0_arr) * self.dx

    def evolve(self, t):
        """
        Returns the solution psi(x,t) at time t using the spectral expansion.
        """
        a_t = self.a0 * np.exp(-self.eigvals * t)
        psi_t = np.dot(self.eigfuncs, a_t)
        return psi_t

class FokkerPlanckSolver(SchrodingerSolver):
    
    def __init__(self, potential, approximator, rho0, num_eigen=10):
        """
        Parameters:
          potential: function of x V. 
                     Notice that the appromimator should be computed accordanly using W(x) = 1/(4sigma) |V'(x)|^2 - 1/2 V''(x)
          approximator: an instance of a subclass of BaseOperatorApproximator.
          rho0: initial condition. It should be either a callable (function of x).
          num_eigen: number of eigenfunctions to use in the spectral expansion.
        """
        eval_points = np.linspace(-10*approximator.L, 10*approximator.L, int(10000*approximator.L))
        constant = np.trapezoid(np.exp(-potential(eval_points) / approximator.sigma), eval_points) 
        self.rho_infinity = lambda x: np.exp(-potential(x) / approximator.sigma) / constant
        psi0 = lambda x: rho0(x) / np.sqrt(self.rho_infinity(x))

        super().__init__(approximator=approximator, psi0=psi0, num_eigen=num_eigen)

    def solve(self, t):
        """
        Returns the solution rho(x,t) at time t using the spectral expansion.
        """
        return self.evolve(t) * np.sqrt(self.rho_infinity(self.x))
    
class SchrodingerControlSolver:
    """
    Solve the optimal control problem for
        d/dt psi = -H psi + sum_i u_i(t) b_i(x) psi,
    with final-time cost and control regularization.

    Inputs:
    -------
    - approximator :  an instance of a subclass of BaseOperatorApproximator.
    - num_eigen    : number of eigenvalues 
    - potential    : function V(x)
    - rho_0        : function initial density
    - rho_dag      : function rho^\dagger(x)
    - nu           : float, regularization parameter
    - nabla_alpha_list   : list of functions [alpha_1'(x), ..., alpha_m'(x)]
                           for the control terms
    - b_list       : optional list of arrays b_i(x). If None, we compute
                     b_i(x) = (1/(2*sigma)) * alpha_i'(x)*V'(x).

    Methods:
    --------
    - solve(max_iter, tol): performs a forward-backward iteration using
      scipy ODE solvers, returning (c_opt, d_opt, u_list), the final
      state, adjoint, and controls over time.
    """

    def __init__(self, potential, rho_0, nu, nabla_alpha_list, approximator, num_eigen, nabla_V=None,
                 correct_lambda0=False, rho_dag=None, kappa=0.0):

        self.approximator = approximator
        self.num_eigen = num_eigen
        self.potential = potential
        self.nu = nu
        self.kappa = kappa
        self.sigma = approximator.sigma
        self.rho_0 = rho_0

        self.x = approximator.x
        self.dx = approximator.dx
        self.L = approximator.L
        self.N = approximator.N
        self.eigvals, self.eigfuncs, self.nabla_eigfuncs = self.approximator.solve_eigen(num_eigen, derivative=True)
        
        eval_points = np.linspace(-10*approximator.L, 10*approximator.L, int(10000*approximator.L))
        constant = np.trapezoid(np.exp(-potential(eval_points) / approximator.sigma), eval_points)
        self.rho_infty = lambda x: np.exp(-potential(x) / approximator.sigma) / constant

        if rho_dag is None:
            self.rho_dag = self.rho_infty
        else:
            self.rho_dag = rho_dag
            
        self.nabla_alpha_list = [nabla_alpha_i(approximator.x) for nabla_alpha_i in nabla_alpha_list]
        if nabla_V is None:
            nabla_V = np.gradient(potential(approximator.x), approximator.x)
        else:
            nabla_V = nabla_V(approximator.x)
        self.b_list = [-(nabla_alpha_i * nabla_V)/(2.0*self.sigma) for nabla_alpha_i in self.nabla_alpha_list]

        if correct_lambda0:
            self.eigvals[0] = 0.0
            self.eigfuncs[:,0] = np.sqrt(self.rho_infty(self.x))
            self.nabla_eigfuncs[:,0] = -self.eigfuncs[:,0] * nabla_V * 0.5 * self.sigma

        B_mats = np.array([self._build_operator_matrix_B(b_i) for b_i in self.b_list])
        A_mats = np.array([self._build_operator_matrix_A(nabla_alpha_i) for nabla_alpha_i in self.nabla_alpha_list])
        self.Delta = B_mats - A_mats
        self.Delta[:, 0, :] = 0.0
        

        psi0 = self.rho_0(self.x) / np.sqrt(self.rho_infty(self.x))
        self.a0 = self._project_to_basis(psi0)

        psi_dag = self.rho_dag(self.x) / np.sqrt(self.rho_infty(self.x))
        self.a_dag = self._project_to_basis(psi_dag)
        self.a_dag = np.zeros_like(self.a_dag)
        self.a_dag[0] = 1.0

        self.m = len(nabla_alpha_list)  # number of controls

    def _project_to_basis(self, fvals):
        """
        Return coefficients a_k = <f, varphi_k> in the truncated basis, i.e.,
            a_k = ∫ f(x) * varphi_k(x) dx,
        approximated using the trapezoidal rule.
        """
        a = np.trapezoid(fvals[:, None] * self.eigfuncs, x=self.x, axis=0)
        return a

    def _build_operator_matrix_A(self, alpha_vals):
        """
        Build the matrix A of shape (num_eigen, num_eigen) such that
        A_{j,k} = ∫ alpha'(x)*varphi_j'(x)*varphi_k(x) dx,
        approximated using the trapezoidal rule on the grid self.x.
        """
        weights = np.empty_like(alpha_vals)
        weights[0] = (self.x[1] - self.x[0]) / 2.0
        weights[-1] = (self.x[-1] - self.x[-2]) / 2.0
        weights[1:-1] = (self.x[2:] - self.x[:-2]) / 2.0
        weights *= alpha_vals
        Amat = (self.nabla_eigfuncs.T * weights) @ self.eigfuncs
        return Amat

    def _build_operator_matrix_B(self, bvals):
        """
        Build the matrix B of shape (num_eigen, num_eigen) for the multiplication operator b(x),
        i.e., B_{j,k} = ∫ b(x)*varphi_j(x)*varphi_k(x) dx,
        approximated using the trapezoidal rule on the grid self.x.
        """
        weights = np.empty_like(bvals)
        weights[0] = (self.x[1] - self.x[0]) / 2.0
        weights[-1] = (self.x[-1] - self.x[-2]) / 2.0
        weights[1:-1] = (self.x[2:] - self.x[:-2]) / 2.0
        weights *= bvals
        Bmat = (self.eigfuncs.T * weights) @ self.eigfuncs
        return Bmat

    def _solve_forward(self, u_list, T, time_eval):
        """Solve the forward ODE: dot{a}(t) = -Lambda*a(t) + sum_i u_i(t)*(B^i - A^i)*a(t)."""
        sol_fwd = solve_ivp(
            fun=lambda t, y: self._forward_ode(t, y, u_list),
            t_span=(0, T),
            y0=self.a0,
            t_eval=time_eval,
            rtol=1e-7, atol=1e-9
        )
        return sol_fwd.y.T  # shape: (len(time_eval), num_eigen)

    def _forward_ode(self, t, a_vec, u_list):
        control_terms = [u(t) * Delta.dot(a_vec) for u, Delta in zip(u_list, self.Delta)]
        return -self.eigvals * a_vec + np.sum(control_terms, axis=0)

    def _solve_backward(self, u_list, T, time_eval, a_vals):
        """Solve the backward ODE for the adjoint d(t) (with reversed time)."""
        a_interp = interp1d(time_eval, a_vals, axis=0, fill_value="extrapolate", kind='linear')
        sol_bwd = solve_ivp(
            fun=lambda t, y: self._backward_ode(t, y, u_list, T, a_interp(T-t)),
            t_span=(0, T),
            y0=self.a_dag - a_vals[-1, :],
            t_eval=(T - time_eval)[::-1],
            rtol=1e-7, atol=1e-9
        )
        return sol_bwd.y.T[::-1]  # reverse to proper time order

    def _backward_ode(self, t, p_vec, u_list, T, a_t):
        # Using u(T-t) to account for time reversal.
        control_terms = [u(T - t) * Delta.T.dot(p_vec) for u, Delta in zip(u_list, self.Delta)]
        return -self.eigvals * p_vec + np.sum(control_terms, axis=0) - self.kappa * (a_t - self.a_dag)

    def _backtracking_line_search(self, inner_prod, u_old, grad, learning_rate_kwargs):
        """
        Perform a backtracking line search using an inlined objective evaluation.
        """
        # Flatten arrays for the line search.
        u_old_flat = u_old.ravel()
        inner_prod_flat = inner_prod.ravel()
        grad_flat = grad.ravel()
        # Compute objective
        f_old = self._reduced_cost_functional(u_old_flat, inner_prod_flat)
        
        gamma = learning_rate_kwargs['gamma_init']
        grad_norm_sq = np.dot(grad_flat, grad_flat)
        u_new_flat = u_old_flat - gamma * grad_flat
        f_new = self._reduced_cost_functional(u_new_flat, inner_prod_flat)
        while f_new > f_old - learning_rate_kwargs['alpha'] * gamma * grad_norm_sq:
            gamma *= learning_rate_kwargs['beta']
            u_new_flat = u_old_flat - gamma * grad_flat
            f_new = self._reduced_cost_functional(u_new_flat, inner_prod_flat)
        u_new = u_new_flat.reshape(u_old.shape)
        return gamma, u_new
    
    def _reduced_cost_functional(self, u, inner_prod):
        return 0.5 * self.nu * np.dot(u, u) - np.dot(inner_prod, u)

    def _constant_learning_rate(self, inner_prod, u_old, grad, learning_rate_kwargs):
        u_new = u_old - learning_rate_kwargs['gamma'] * grad
        return learning_rate_kwargs['gamma'], u_new

    def _update_control(self, a_vals, p_vals, u_old, learning_rate_kwargs, method):
        """
        Compute new control array U with shape (Nt, m):
          U[n, i] = (1/nu)*<a(t_n), (B^i - A^i) p(t_n)>
        using vectorized operations.
        """
        inner_prod = np.einsum('ni,mij,nj->nm', a_vals, self.Delta, p_vals)
        grad = self.nu * u_old - inner_prod
        grad_norm = np.linalg.norm(grad)

        gamma, u_new = method(inner_prod, u_old, grad, learning_rate_kwargs)
        return u_new, grad_norm, gamma

    def solve(self, T, max_iter=20, tol=1e-6, time_eval=None, verbose=True, control_funcs=None,
              learning_rate_kwargs={'gamma': 1.0, 'gamma_init': 1.0, 'alpha': 0.5, 'beta': 0.8}):
        """
        Perform a forward-backward iteration to determine the optimal control functions.
        If control_funcs is provided (a list of functions of time), they are used instead
        of iterating for an optimal control.
        """
        if time_eval is None:
            time_eval = np.linspace(0, T, 101)
        # Use user-set controls if provided; otherwise, initialize as zero.
        if control_funcs is not None:
            u_list = control_funcs
        else:
            exp_decay = lambda t: np.exp(-5 * t)
            u_list = [exp_decay] * self.m
        old_u_discrete = np.column_stack([u(time_eval) for u in u_list])

        if 'method' in learning_rate_kwargs:
            if learning_rate_kwargs['method'] == 'constant':
                lr_method = self._constant_learning_rate
            else:
                lr_method = self._backtracking_line_search
        else:
            lr_method = self._constant_learning_rate
        
        # Only iterate for optimal control if control_funcs is not provided.
        if control_funcs is None:
            it = 0
            while it < max_iter:
                a_vals = self._solve_forward(u_list, T, time_eval)  # Solve forward ODE.
                p_vals = self._solve_backward(u_list, T, time_eval, a_vals)  # Solve backward ODE.
                # Update control.
                new_u_discrete, grad_norm, gamma = self._update_control(a_vals, p_vals, old_u_discrete, learning_rate_kwargs, lr_method)
                if verbose:
                    print(f"Iteration {it+1}: ||grad|| = {grad_norm:.3e}, gamma = {gamma}")
                if grad_norm < tol:
                    break
                old_u_discrete = new_u_discrete.copy()
                # Update u_list as piecewise linear interpolants.
                u_list = [
                    (lambda arr, te: lambda t: np.interp(t, te, arr, left=arr[0], right=arr[-1]))(new_u_discrete[:, i], time_eval)
                    for i in range(self.m)
                ]
                it += 1
            if it == max_iter: print("WARNING - Maximum number of iterations attained")
        else:
            # If user controls are provided, solve forward and backward once.
            a_vals = self._solve_forward(u_list, T, time_eval)
            p_vals = self._solve_backward(u_list, T, time_eval, a_vals)
            new_u_discrete = np.array([u(time_eval) for u in control_funcs]).T
        
        # Reconstruct psi(x,t) and varphi(x,t) on the spatial grid.
        psi_vals = a_vals @ self.eigfuncs.T    # shape: (len(time_eval), len(self.x))
        varphi_vals = p_vals @ self.eigfuncs.T
        
        return {
            "time": time_eval,
            "a_vals": a_vals,
            "p_vals": p_vals,
            "u_vals": new_u_discrete,
            "psi": psi_vals,
            "varphi": varphi_vals,
        }