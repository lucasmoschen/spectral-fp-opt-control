#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.linalg import solve_continuous_are

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
                 correct_lambda0=False, rho_dag=None, rho_hat=None, kappa=0.0):

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
        if rho_hat is None:
            self.rho_hat = self.rho_infty
        else:
            self.rho_hat = rho_hat
            
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

        psi_hat = self.rho_hat(self.x) / np.sqrt(self.rho_infty(self.x))
        self.a_hat = self._project_to_basis(psi_hat)

        if rho_dag is None:
            # Numerical correction if rho_dag is not provided.
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
        return -self.eigvals * p_vec + np.sum(control_terms, axis=0) - self.kappa * (a_t - self.a_hat)

    def _reduced_cost_functional(self, u, inner_prod):
        return 0.5 * self.nu * np.dot(u, u) - np.dot(inner_prod, u)

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

    def _constant_learning_rate(self, inner_prod, u_old, grad, learning_rate_kwargs):
        u_new = u_old - learning_rate_kwargs['gamma'] * grad
        return learning_rate_kwargs['gamma'], u_new

    def _barzilai_borwein(self, u_old, grad, prev_u, prev_grad, learning_rate_kwargs):
        """
        Compute a BB step-size update.
        prev_u and prev_grad are the previous iterate and gradient (or None if not available).
        Returns:
        gamma: computed step-size,
        u_new: updated control,
        new_prev_u, new_prev_grad: updated previous iterate and gradient (to be used in the next iteration).
        """
        # If no previous iterate is available, use the initial gamma.
        if prev_u is None or prev_grad is None:
            gamma = learning_rate_kwargs.get('gamma', 1.0)
        else:
            s = (u_old - prev_u).ravel()
            y = (grad - prev_grad).ravel()
            denominator = np.dot(y, y)
            if np.abs(denominator) < 1e-12:
                gamma = learning_rate_kwargs.get('gamma', 1.0)
            else:
                gamma = np.dot(s, y) / denominator
        u_new = u_old - gamma * grad
        return gamma, u_new, u_old.copy(), grad.copy()

    def _update_control(self, a_vals, p_vals, u_old, learning_rate_kwargs, method, bb_data):
        """
        Compute new control array U with shape (Nt, m) using vectorized operations.
        Returns u_new, grad_norm, gamma.
        If using BB (i.e. method == _barzilai_borwein), bb_data is a tuple (prev_u, prev_grad).
        """
        inner_prod = np.einsum('ni,mij,nj->nm', a_vals, self.Delta, p_vals)
        grad = self.nu * u_old - inner_prod
        grad_norm = np.linalg.norm(grad)
        
        if method == self._barzilai_borwein:
            prev_u, prev_grad = bb_data if bb_data is not None else (None, None)
            gamma, u_new, new_prev_u, new_prev_grad = method(u_old, grad, prev_u, prev_grad, learning_rate_kwargs)
            bb_data = (new_prev_u, new_prev_grad)
        else:
            gamma, u_new = method(inner_prod, u_old, grad, learning_rate_kwargs)
        return u_new, grad_norm, gamma, bb_data

    def _optimize_control(self, T, time_eval, control_funcs, learning_rate_kwargs, max_iter, tol, verbose, inicialization):
        """
        Optimize the control on [0, T] using forward-backward iterations.
        Returns:
        u_list: list of control interpolants on [0, T]
        u_discrete: discrete control (array shape (len(time_eval), self.m))
        a_vals: state expansion coefficients on [0, T]
        p_vals: adjoint variables on [0, T]
        """
        # Initialize control functions on [0, T]
        if control_funcs is not None:
            u_list = control_funcs
        elif inicialization:
            print("WARNING - Using LQR inicialization")
            u_list, self.u_initial, self.v_initial, self.K = self._lrq_inicialization(T, time_eval)
        else:
            exp_decay = lambda t: np.exp(-5 * t)
            u_list = [exp_decay] * self.m
        u_discrete = np.column_stack([u(time_eval) for u in u_list])
        
        # Select learning rate method.
        method_choice = learning_rate_kwargs.get('method', 'constant')
        bb_data = (None, None)
        if method_choice == 'constant':
            lr_method = self._constant_learning_rate
        elif method_choice == 'bb':
            lr_method = self._barzilai_borwein
        else:
            lr_method = self._backtracking_line_search
        
        it = 0
        while it < max_iter:
            a_vals = self._solve_forward(u_list, T, time_eval)
            p_vals = self._solve_backward(u_list, T, time_eval, a_vals)
            new_u_discrete, grad_norm, gamma, bb_data = self._update_control(a_vals, p_vals, u_discrete,
                                                                            learning_rate_kwargs, lr_method, bb_data)
            if verbose:
                print(f"Iteration {it+1}: ||grad|| = {grad_norm:.3e}, gamma = {gamma}")
            if grad_norm < tol:
                break
            u_discrete = new_u_discrete.copy()
            # Update control interpolants on [0, T]
            u_list = [lambda t, arr=u_discrete[:, i], te=time_eval: np.interp(t, te, arr, left=arr[0], right=arr[-1])
                    for i in range(self.m)]
            it += 1
        if it == max_iter:
            print("WARNING - Maximum number of iterations attained")
        return u_list, u_discrete, a_vals, p_vals

    def _simulate_free_dynamics(self, t0, tf, time_eval, a_vals_until_t0, p_vals_until_t0):
        """
        Run the forward simulation over [0, T_total] using the control functions u_list_full.
        """
        sol_fwd = solve_ivp(
            fun=lambda t, y: self._forward_ode(t, y, [lambda t: 0.0]*self.m),
            t_span=(t0, tf),
            y0=a_vals_until_t0[-1,:],
            t_eval=time_eval[time_eval > t0],
            rtol=1e-7, atol=1e-9
        )
        a_vals_after_t0 = sol_fwd.y.T

        a_interp = interp1d(time_eval[time_eval>t0], a_vals_after_t0, axis=0, fill_value="extrapolate", kind='linear')
        sol_bwd = solve_ivp(
            fun=lambda t, y: -self._backward_ode(t, y, [lambda t: 0.0]*self.m, tf, a_interp(t)),
            t_span=(t0, tf),
            y0=p_vals_until_t0[-1,:],
            t_eval=time_eval[time_eval > t0],
            rtol=1e-7, atol=1e-9
        )
        p_vals_after_t0 = sol_bwd.y.T

        a_vals = np.vstack( [a_vals_until_t0, a_vals_after_t0])
        p_vals = np.vstack( [p_vals_until_t0, p_vals_after_t0])
        return a_vals, p_vals

    def _solve_lqr_steady(self, A, B):
        """
        Compute the steady-state optimal feedback control law for the LQR problem
        without the terminal cost ||v(T)||^2.

        Parameters:
            A (ndarray): Diagonal system matrix (n x n)
            B (ndarray): Control matrix (n x m)
            nu (float): Control weight (> 0)
            kappa (float): State tracking weight

        Returns:
            K (ndarray): Constant optimal feedback gain matrix (m x n)
        """
        Q = self.kappa * np.eye(self.num_eigen-1)
        R = self.nu * np.eye(self.m)
        P = solve_continuous_are(A, B, Q, R)
        K = B.T @ P / self.nu
        return K

    def _lrq_inicialization(self, T, time_eval):
        """
        Compute the steady-state optimal control u*(t) = -K v(t).

        Parameters:
            v_func (function): Function v(t) returning the state at time t
            K (ndarray): Constant feedback matrix

        Returns:
            u_func (function): Function returning u(t) at any time t
        """
        A = -np.diag(self.eigvals[1:])
        B = self.Delta[:, 1:, 0].T
        K = self._solve_lqr_steady(A, B)

        sol = solve_ivp(fun=lambda t, v: A @ v + B @ (-K@v), 
                        t_span=(0, T),
                        y0=self.a0[1:] - self.a_dag[1:], 
                        t_eval=time_eval,
                        rtol=1e-7, atol=1e-9)
        v_vals = sol.y.T
        u_vals = (-K @ v_vals.T).T
        u_list = [lambda t, arr=u_vals[:, i], te=time_eval: np.interp(t, te, arr, left=arr[0], right=arr[-1])
                  for i in range(self.m)]
        return u_list, u_vals, v_vals, K

    def compute_cost_functional(self, u_list, T, time_eval=None):
        """
        Compute the cost functional
        J = 0.5 * ||a(T) - a_dag||^2 
            + 0.5 * nu * int_0^T ||u(t)||^2 dt 
            + 0.5 * kappa * int_0^T ||a(t) - a_hat||^2 dt,
        where a(t) is obtained by solving the forward ODE with control functions u_list.
        
        Parameters:
        u_list : list of callables
            Control functions defined on [0, T].
        T : float
            Final time for the controlled phase.
            
        Returns:
        cost : float
            The computed cost functional.
        """
        if time_eval is None:
            time_eval = np.linspace(0, T, 101)
        a_vals = self._solve_forward(u_list, T, time_eval)
        
        terminal_cost = 0.5 * np.linalg.norm(a_vals[-1, :] - self.a_dag)**2
        u_vals = np.column_stack([u(time_eval) for u in u_list])
        control_cost = 0.5 * self.nu * np.trapz(np.sum(u_vals**2, axis=1), time_eval)
        tracking_cost = 0.5 * self.kappa * np.trapz(np.sum((a_vals - self.a_hat)**2, axis=1), time_eval)
        cost = terminal_cost + control_cost + tracking_cost
        return cost

    def solve(self, T, t_free=0.0, max_iter=20, tol=1e-6, time_eval=None, verbose=True, 
            control_funcs=None, optimise=True, inicialization=False,
            learning_rate_kwargs={'gamma': 1.0, 'gamma_init': 1.0, 'alpha': 0.5, 'beta': 0.8}):
        """
        Solve the control problem in two stages:
        1. Optimisation stage: Optimize the control on [0, T].
        2. Free-dynamics stage: Run the free (uncontrolled) dynamics on [T, T+t_free], if t_free > 0.
        Returns a dictionary with the following keys:
        - time: full time grid [0, T+t_free]
        - a_vals: state coefficients on the full time grid
        - p_vals: adjoint coefficients on [0, T]
        - u_vals: control values on the full time grid
        - psi: reconstructed state on the spatial grid
        - varphi: adjoint variables (not yet implemented)
        """
        T_total = T + t_free
        if time_eval is None:
            time_eval = np.linspace(0, T_total, 101)
        time_eval_partial = time_eval[time_eval <= T]
        if optimise:
            u_list, u_discrete, a_vals, p_vals = self._optimize_control(T, time_eval_partial, control_funcs,
                                                                        learning_rate_kwargs, max_iter, tol, verbose, 
                                                                        inicialization)
        else:
            # If not optimizing, use the provided control functions (or default) on [0, T].
            if control_funcs is not None:
                u_list = control_funcs
            else:
                exp_decay = lambda t: np.exp(-5 * t)
                u_list = [exp_decay] * self.m
            a_vals = self._solve_forward(u_list, T, time_eval_partial)
            p_vals = self._solve_backward(u_list, T, time_eval_partial, a_vals)
            u_discrete = np.column_stack([u(time_eval_partial) for u in u_list])

        if t_free > 0:
            u_discrete = np.vstack([u_discrete, np.zeros((time_eval[time_eval > T].shape[0], self.m))])
            a_vals, p_vals = self._simulate_free_dynamics(T, T_total, time_eval, a_vals, p_vals)
        
        psi_vals = a_vals @ self.eigfuncs.T
        varphi_vals = p_vals @ self.eigfuncs.T
        
        return {
            "time": time_eval,
            "a_vals": a_vals,
            "p_vals": p_vals,
            "u_vals": u_discrete,
            "psi": psi_vals,
            "varphi": varphi_vals,
        }
