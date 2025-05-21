import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
try:
    from slycot import sb02md
    _HAS_SLYCOT = True
except ImportError:
    _HAS_SLYCOT = False
from schorodinger_operator import SchrodingerControlSolver

class RiccatiSolver:
    """
    Solve the time-varying Riccati differential equation using Slycot if available,
    otherwise fallback to SciPy ODE integration (BDF).

    dP/dt = -(A_eff^T P + P A_eff - P B_eff R^{-1} B_eff^T P + Q)
    backward on [0, T] with P(T) = P_T.
    """
    def __init__(self, A, B, N, Q, R, T, t_grid, P_T=None):
        self.A = A
        self.B = B
        self.N = N
        self.Q = Q
        self.R = R
        self.T = T
        self.t_grid = t_grid
        self.R_inv = np.linalg.inv(R)
        self.P_T = P_T if P_T is not None else np.zeros_like(A)

    def solve(self, x_traj, u_traj):
        """
        Returns P_traj on self.t_grid of shape (len(t_grid), n, n).
        """
        n = self.A.shape[0]
        # Interpolators for x and u
        x_interp = interp1d(self.t_grid, x_traj, axis=0, kind='cubic', fill_value='extrapolate')
        u_interp = interp1d(self.t_grid, u_traj, axis=0, kind='cubic', fill_value='extrapolate')

        if _HAS_SLYCOT:
            # Prepare time-varying coefficients arrays
            AT = np.zeros((n, n, len(self.t_grid)))
            BT = np.zeros((n, u_traj.shape[1] if u_traj.ndim>1 else 1, len(self.t_grid)))
            for idx, t in enumerate(self.t_grid):
                x = x_interp(t)
                u = u_interp(t)
                A_eff = self.A + 0.5 * u * self.N
                B_eff = self.B + 0.5 * self.N @ x[:, None]
                AT[:, :, idx] = A_eff.T
                BT[:, :, idx] = B_eff
            # time grid in ascending order
            times = self.t_grid
            # call Slycot SB02MD: continuous-time Riccati
            X, _, _ = sb02md('D', n, BT.shape[1], len(times), AT, BT, self.Q, self.R, times)
            # X has shape (n, n, len(times)) with P(t) forward; need backward
            P_traj = np.moveaxis(X, 2, 0)
            return P_traj
        else:
            # Fallback: SciPy ODE with BDF
            def riccati_ode(t, P_flat):
                P = P_flat.reshape(n, n)
                x = x_interp(t)
                u = u_interp(t)
                A_eff = self.A + 0.5 * u * self.N
                B_eff = self.B + 0.5 * self.N @ x[:, None]
                term = P @ B_eff @ self.R_inv @ B_eff.T @ P
                P_dot = -(A_eff.T @ P + P @ A_eff - term + self.Q)
                return P_dot.flatten()

            sol = solve_ivp(
                fun=riccati_ode,
                t_span=(self.T, 0),
                y0=self.P_T.flatten(),
                t_eval=self.t_grid[::-1],
                method='BDF',
                atol=1e-8,
                rtol=1e-6
            )
            P_traj = sol.y.T[::-1].reshape(-1, n, n)
            return P_traj

class LOperator:
    def __init__(self, A, B, N, Q, R, x0, T, t_grid):
        self.x0 = x0
        self.riccati = RiccatiSolver(A, B, N, Q, R, T, t_grid)
        self.A = A; self.B = B; self.N = N
        self.R_inv = np.linalg.inv(R)
        self.t_grid = t_grid

    def apply(self, x_traj, u_traj):
        P_traj = self.riccati.solve(x_traj, u_traj)
        n = self.x0.size
        y = np.zeros((len(self.t_grid), n))
        v = np.zeros((len(self.t_grid),))
        y[0] = self.x0
        for i in range(len(self.t_grid)-1):
            dt = self.t_grid[i+1] - self.t_grid[i]
            xi, ui, Pi = y[i], u_traj[i], P_traj[i]
            A_eff = self.A + 0.5 * ui * self.N
            B_eff = self.B + 0.5 * self.N @ x_traj[i][:, None]
            K = self.R_inv @ B_eff.T @ Pi
            vi = -K @ xi
            v[i] = vi
            y[i+1] = xi + dt * (A_eff @ xi + (B_eff * vi).flatten())
        v[-1] = v[-2]
        return y, v

class FixedPointSolver:
    def __init__(self, A, B, N, Q, R, x0, T, t_grid, tol=1e-6, max_iter=50):
        self.operator = LOperator(A, B, N, Q, R, x0, T, t_grid)
        self.tol = tol
        self.max_iter = max_iter
        self.t_grid = t_grid

    def solve(self, x_init, u_init):
        x, u = x_init.copy(), u_init.copy()
        for k in range(self.max_iter):
            y, v = self.operator.apply(x, u)
            err = np.linalg.norm(np.vstack((y-x, (v-u)[:,None])))
            x, u = y, v
            if err < self.tol:
                print(f"Converged in {k+1} iterations")
                break
        return x, u

if __name__ == "__main__":

    # Example usage
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    N = np.array([[0, 0], [0, 0]])
    Q = np.eye(2)
    R = np.array([[1]])
    x0 = np.array([1, 0])
    T = 10
    t_grid = np.linspace(0, T, 100)

    solver = FixedPointSolver(A, B, N, Q, R, x0, T, t_grid)
    x_init = np.zeros((len(t_grid), 2))
    u_init = np.zeros((len(t_grid), 1))
    x_sol, u_sol = solver.solve(x_init, u_init)
    print("x trajectory:", x_sol)
    print("u trajectory:", u_sol)