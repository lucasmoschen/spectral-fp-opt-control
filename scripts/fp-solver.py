#!usr/bin/env/python

#import numpy as np
import autograd.numpy as np 
from autograd import elementwise_grad as egrad, grad
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

class FokkerPlanckEquation:
    """
    The Fokker Planck equation is a Partial Differential Equation for the density of a random variable whose dynamics is 
    controlled by an Stochastic Differential Equation with drift and diffusion coefficients.
    This class aim to solve this equation considering initial and boundary conditions.
    """
    def __init__(self, G_func, alpha_func, control, parameters) -> None:
        """
        Consider the equation 
        p_t = v * delta p + nabla . (p * nabla (G + alpha * u))  

        Parameters
        ----
        G_func: The function G with is the potential function without the control. 
                It is a function whose parameter is a vector x and returns a single-value.
        alpha_func: The function alpha, which indicates the shape control in the space. 
                It is a function whose parameter is a vector x and returns a single-value.
        control: The function u, which measures the used control in the model.
                It is a function whose parameter is a positive real t and returns a single value.
        parameters: Dictionary with additional fixed parameters: v (rate), T (final time), p_0 (initial function) and interval.
        """
        self.G = G_func
        self.alpha = alpha_func
        self.u = control
        self.v = parameters['v']
        self.T = parameters['T']
        self.p0 = parameters['p_0']
        self.lb = float(parameters['interval'][0])
        self.ub = float(parameters['interval'][1])
        self.X = self.ub - self.lb

    def _pre_calculations_1d_forward_finite_difference(self, N_x, N_t):
        """
        Perform the matrix calculations for the 1d solving problem - Forward Difference.
        """
        h_x = self.X/N_x
        h_t = self.T/N_t 

        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)
        V_x = lambda x,t: G_x(x) + alpha_x(x) * self.u(t)
        G_xx = egrad(G_x)
        alpha_xx = egrad(alpha_x)
        V_xx = lambda x,t: G_xx(x) + alpha_xx(x) * self.u(t)

        x_var = np.arange(self.lb, self.ub+0.99*h_x, h_x)
        t_var = np.arange(0, self.T, h_t)
        X_var, T_var = np.meshgrid(x_var[1:-1], t_var)
        V_x_matrix = h_t/(2*h_x) * V_x(X_var, T_var)
        V_xx_matrix = h_t * V_xx(X_var, T_var)
        if self.v != 0:
            V_x_border = h_x/self.v * V_x(*np.meshgrid([self.lb, self.ub-h_x], t_var+h_t))
        else:
            V_x_border = None
        coef1 = self.v*h_t/h_x**2
        coef2 = 1-2*self.v*h_t/h_x**2

        A = coef1 + V_x_matrix
        B = coef2 + V_xx_matrix
        C = coef1 - V_x_matrix
        rho_0 = self.p0(x_var)
        
        return A, B, C, rho_0, V_x_border

    def _pre_calculations_1d_ck_finite_difference(self, N_x, N_t):
        """
        Perform the matrix calculations for the 1d solving problem - Crack-Nicolson.
        """
        h_x = self.X/N_x
        h_t = self.T/N_t 

        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)
        V_x = lambda x,t: G_x(x) + alpha_x(x) * self.u(t)
        G_xx = egrad(G_x)
        alpha_xx = egrad(alpha_x)
        V_xx = lambda x,t: G_xx(x) + alpha_xx(x) * self.u(t)

        x_var = np.arange(self.lb, self.ub+0.9*h_x, h_x)
        t_var = np.arange(0, self.T+0.9*h_t, h_t)
        X_var, T_var = np.meshgrid(x_var[1:-1], t_var)
        V_x_matrix = V_x(X_var, T_var)
        V_xx_matrix = V_xx(X_var, T_var)
        if self.v != 0:
            V_x_border = [V_x(*np.meshgrid([self.lb, self.ub-h_x], t_var[1:])) - self.v/h_x, self.v/h_x]
        else:
            V_x_border = None

        A1 = 0.5/h_x * (0.5 * V_x_matrix[1:, :] - self.v/h_x)
        A2 = 1/h_t + self.v/h_x**2 - 0.5 * V_xx_matrix[1:, :]
        A3 = -0.5/h_x * (0.5 * V_x_matrix[1:, :] + self.v/h_x)
        B1 = 0.5/h_x * (self.v/h_x - 0.5 * V_x_matrix[:-1, :])
        B2 = 1/h_t - self.v/h_x**2 + 0.5 * V_xx_matrix[:-1, :]
        B3 = 0.5/h_x * (0.5 * V_x_matrix[:-1, :] + self.v/h_x)
        rho_0 = self.p0(x_var)

        return A1, A2, A3, B1, B2, B3, rho_0, V_x_border

    def _pre_calculations_1d_galerkin_finite_elements(self, n_f, N_t):
        """
        Perform the matrix calculations for the 1d solving problem - Galerkin Finite Elements.
        """
        h_t = self.T/N_t

        family = self._building_family()
        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)
        V_x = lambda x,t: G_x(x) + alpha_x(x) * self.u(t)
        diff = lambda n: grad(lambda x: family(x,n))
        
        A = np.zeros((n_f, n_f))
        B = np.zeros((n_f, n_f))
        b = np.zeros(n_f)
        c = np.zeros(n_f)

        for i in tqdm(range(n_f)):
            b[i] = self.v * diff(i)(self.lb) + family(self.lb, i) * V_x(self.lb, 0)
            c[i] = self.v * diff(i)(self.ub) + family(self.ub, i) * V_x(self.ub, 0)
            for j in range(i,n_f):
                A[i,j] = quad(func=lambda x: family(x,i) * family(x,j), a=self.lb, b=self.ub)[0]
                A[j,i] = A[i,j]
                B[i,j] = -quad(func=lambda x: (self.v * diff(j)(x) + V_x(x,0) * family(x,j)) * family(x,i), a=self.lb, b=self.ub)[0]
                B[j,i] = -quad(func=lambda x: (self.v * diff(i)(x) + V_x(x,0) * family(x,i)) * family(x,j), a=self.lb, b=self.ub)[0]

        next_matrix = A - 0.5*h_t*B
        previous_matrix = A + 0.5*h_t*B
        previous_matrix[[0,-1],:] = 0.0
        next_matrix[0,:] = b
        next_matrix[-1,:] = c

        rho_0 = np.zeros(n_f)
        rho_0[0] = 1.0

        return previous_matrix, next_matrix, rho_0, family

    def _building_family(self):
        """
        This function builds a list of functions who will serve as test functions for the Galerkin method.
        """
        def family(x,n):
            if n == 0:
                return self.p0(x)
            elif n % 2 == 1:
                return np.sin(2*n*np.pi*x)
            else:
                return np.cos(2*n*np.pi*x)
        return family

    def _solve1d_forward_finite_difference(self, N_x, N_t):
        A, B, C, rho_0, V_x_border = self._pre_calculations_1d_forward_finite_difference(N_x, N_t)
        rho = np.zeros((N_t+1, N_x+1))
        rho[0,:] = rho_0
        for i in tqdm(range(N_t)):
            rho[i+1,1:N_x] = C[i,:]*rho[i,0:N_x-1] + B[i,:]*rho[i,1:N_x] + A[i,:]*rho[i,2:N_x+1]
            rho[i+1,0] = rho[i+1,1]/(1 - V_x_border[i][0])
            rho[i+1,N_x] = rho[i+1,N_x-1]*(1 - V_x_border[i][1])
        return rho

    def _solve1d_ck_finite_difference(self, N_x, N_t):
        A1, A2, A3, B1, B2, B3, rho_0, V_x_border = self._pre_calculations_1d_ck_finite_difference(N_x, N_t)
        A = np.zeros((N_x+1, N_x+1))
        rho = np.zeros((N_t+1, N_x+1))
        rho[0,:] = rho_0
        for i in tqdm(range(N_t)):
            rho[i+1,1:N_x] = B1[i,:]*rho[i,0:N_x-1] + B2[i,:]*rho[i,1:N_x] + B3[i,:]*rho[i,2:N_x+1]
            A[0,0] = V_x_border[0][i,0]
            A[0,1] = V_x_border[1]
            A[N_x, N_x-1] = V_x_border[0][i,1]
            A[N_x, N_x] = V_x_border[1]
            A[range(1,N_x), range(N_x-1)] = A1[i,:]
            A[range(1,N_x), range(1,N_x)] = A2[i,:]
            A[range(1,N_x), range(2,N_x+1)] = A3[i,:]
            rho[i+1,:] = np.linalg.solve(A, rho[i+1,:])
        return rho

    def _solve1d_galerkin_finite_elements(self, n_f, N_x, N_t):
        
        previous_matrix, next_matrix, rho_0, family = self._pre_calculations_1d_galerkin_finite_elements(n_f, N_t)
        h_t = self.T/N_t
        h_x = self.X/N_x

        rho_vec = np.zeros((N_t+1, n_f))
        rho_vec[0,:] = rho_0

        for i in tqdm(range(N_t)):
            rho_vec[i+1,:] = previous_matrix @ rho_vec[i,:]
            rho_vec[i+1,:] = np.linalg.solve(next_matrix, rho_vec[i+1,:])
        phi_matrix = np.zeros((n_f, N_x+1))
        x = np.arange(self.lb, self.ub+0.9*h_x, h_x)
        for k in range(n_f):
            phi_matrix[k,:] = family(x, k)
        rho = rho_vec @ phi_matrix

        return rho

    def solve1d(self, N_x=None, N_t=None, n_f=None, type='forward'):
        """
        Solves the Fokker-Planck equation in 1 dimension, including initial and boundary conditions.
        """
        if type == 'forward':
            rho = self._solve1d_forward_finite_difference(N_x, N_t)
        elif type == 'ck':
            rho = self._solve1d_ck_finite_difference(N_x, N_t)
        elif type == 'galerkin_fem':
            rho = self._solve1d_galerkin_finite_elements(n_f, N_x, N_t)
        return rho

if __name__ == '__main__':

    G_func = lambda x: x*x
    alpha_func = lambda x: x
    control = lambda t: 0.0
    p_0 = lambda x: np.exp(-x*x)/(np.sqrt(np.pi)*(norm.cdf(np.sqrt(2)) - 0.5))
    interval = [0.0,1.0]
    parameters = {'v': 1.0, 'T': 10.0, 'p_0': p_0, 'interval': interval}

    FP_equation = FokkerPlanckEquation(G_func, alpha_func, control, parameters)

    solving1 = FP_equation.solve1d(N_x=25, N_t=100000, type='forward')
    solving2 = FP_equation.solve1d(N_x=25, N_t=100000, type='ck')
    solving3 = FP_equation.solve1d(n_f=21, N_x=25, N_t=100000, type='galerkin_fem')

    x = np.linspace(0, 1, 26)
    t = np.linspace(0, 10, 100001)
    X, T = np.meshgrid(x,t)

    # Plotting the 3d figure
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, T, solving1)
    ax.plot_surface(X, T, solving2)
    ax.plot_surface(X, T, solving3)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('rho')
    plt.show() 

    # Plotting the integral of x-axis for each time
    #plt.plot(t, solving1.sum(axis=1)/100)
    #plt.plot(t, solving2.sum(axis=1)/100)
    #plt.ylim(0, 2)
    #plt.show()
        