#!usr/bin/env/python

import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

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

    def _pre_calculations_1d(self, N_x, N_t):
        """
        Perform the matrix calculations for the 1d solving problem.
        """
        h_x = self.X/N_x
        h_t = self.T/N_t 

        G_x = grad(self.G)
        alpha_x = grad(self.alpha)
        V_x = lambda x,t: G_x(x) + alpha_x(x) * self.u(t)
        G_xx = grad(G_x)
        alpha_xx = grad(alpha_x)
        V_xx = lambda x,t: G_xx(x) + alpha_xx(x) * self.u(t)
        
        V_x_matrix = [[h_t/(2*h_x) * V_x(self.lb + j*h_x, i*h_t) for j in range(1,N_x)] for i in range(N_t)]
        V_xx_matrix = [[h_t * V_xx(self.lb + j*h_x, i*h_t) for j in range(1,N_x)] for i in range(N_t)]     
        if self.v != 0:
            V_x_border = [[h_x/self.v * V_x(self.lb, i*h_t), h_x/self.v * V_x(self.ub, i*h_t)] for i in range(1,N_t+1)]
        else:
            V_x_border = None
        coef1 = self.v*h_t/h_x**2
        coef2 = 1-2*self.v*h_t/h_x**2

        A = np.array([[coef1 + V_x_matrix[i][j] for j in range(N_x-1)] for i in range(N_t)])
        B = np.array([[coef2 + V_xx_matrix[i][j] for j in range(N_x-1)] for i in range(N_t)])
        C = np.array([[coef1 - V_x_matrix[i][j] for j in range(N_x-1)] for i in range(N_t)])

        rho_0 = [self.p0(self.lb + j*h_x) for j in range(N_x+1)]

        return A, B, C, rho_0, V_x_border

    def solve1d(self, N_x, N_t):
        """
        Solves the Fokker-Planck equation in 1 dimension, including initial and boundary conditions.
        """
        A, B, C, rho_0, V_x_border = self._pre_calculations_1d(N_x, N_t)
        rho = np.zeros((N_t+1, N_x+1))
        rho[0,:] = rho_0
        for i in range(N_t):
            rho[i+1,1:N_x] = C[i,:]*rho[i,0:N_x-1] + B[i,:]*rho[i,1:N_x] + A[i,:]*rho[i,2:N_x+1]
            rho[i+1,0] = rho[i+1,1]/(1 - V_x_border[i][0])
            rho[i+1,N_x] = rho[i+1,N_x-1]*(1 - V_x_border[i][1])
        return rho

if __name__ == '__main__':

    G_func = lambda x: x*x
    alpha_func = lambda x: 1.0
    control = lambda t: 0.0
    p_0 = lambda x: x*(1-x)
    interval = [0,1]
    parameters = {'v': 0.0, 'T': 1, 'p_0': p_0, 'interval': [0,1]}

    FP_equation = FokkerPlanckEquation(G_func, alpha_func, control, parameters)

    solving = FP_equation.solve1d(N_x=100, N_t=50)

    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 51)
    X, T = np.meshgrid(x,t)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, T, solving)
    ax.plot_surface(X, T, solving2)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('rho')
    ax.set_zlim(0,2)
    plt.show() 

        