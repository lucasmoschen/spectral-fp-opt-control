#!usr/bin/env/python

#import numpy as np
import autograd.numpy as np 
from autograd import elementwise_grad as egrad, grad

from scipy.integrate import quad, solve_ivp
from scipy.stats import norm, truncnorm
from scipy.linalg import solve_banded, solve_continuous_are
from scipy.special import legendre
from scipy.interpolate import lagrange

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

    def _pre_calculations_1d_spectral_galerkin(self, n_f):
        """
        Perform the matrix calculations for the 1d solving problem - Spectral Legendre-Galerkin method.
        """
        legendre_family = [legendre(k) for k in range(n_f+1)]
        legendre_family_diff = [np.polyder(poly, 1) for poly in legendre_family]
        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)
        
        A = np.diag( self.X / (2*np.linspace(0,n_f,n_f+1) + 1))
        B = self.v*np.array([[min(i,j)*(min(i,j)+1)*(1+(-1)**(i+j)) for j in range(n_f+1)] for i in range(n_f+1)]) / self.X
        C = np.zeros((n_f+1, n_f+1))
        D = np.zeros((n_f+1, n_f+1))
        rho_0 = np.zeros(n_f+1)
        for i in tqdm(range(n_f+1)):
            for j in range(n_f+1):
                C[i,j] = quad(func=lambda x: G_x(0.5*(self.X*x + self.lb+self.ub))*legendre_family[j](x)*legendre_family_diff[i](x), 
                              a=-1, b=1)[0]
                D[i,j] = quad(func=lambda x: alpha_x(0.5*(self.X*x + self.lb+self.ub))*legendre_family[j](x)*legendre_family_diff[i](x), 
                              a=-1, b=1)[0]
            rho_0[i] = (i + 0.5)*quad(func=lambda x: self.p0(0.5*(self.X*x + self.lb+self.ub))*legendre_family[i](x), 
                                      a=-1, b=1)[0]

        b1 = np.array([(-1)**(i-1) * (self.v * i * (i+1) / self.X - G_x(self.lb)) for i in range(n_f+1)])
        c1 = np.array([self.v * i * (i+1) / self.X + G_x(self.ub) for i in range(n_f+1)])

        b2 = np.array([(-1)**i * alpha_x(self.lb) for i in range(n_f+1)])
        c2 = np.array([alpha_x(self.ub) for i in range(n_f+1)])

        boundary_conditions = [b1, c1, b2, c2]
        
        return A, B, C, D, rho_0, legendre_family, boundary_conditions

    def _pre_calculations_1d_galerkin_finite_elements(self, n_f, N_t):
        """
        Perform the matrix calculations for the 1d solving problem - Galerkin Finite Elements. 
        We consider the basis of functions linear by parts.
        """
        h_f = self.X/(n_f-1)
        h_t = self.T/N_t
        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)
        V_x = lambda x,t: G_x(x) + alpha_x(x) * self.u(t)
        x_values = np.arange(self.lb, self.ub+0.9*h_f, h_f)
        G_values = self.G(x_values)
        alpha_values = self.alpha(x_values)

        # A = np.zeros((n_f, n_f))
        # A[range(0,n_f-1), range(1,n_f)] = h_f/6
        # A[range(1,n_f-1), range(1,n_f-1)] = 2*h_f/3
        # A[range(1,n_f), range(n_f-1)] = h_f/6
        # A[[0,-1],[0,-1]] = h_f/3

        A = np.zeros((3, n_f))
        A[0,1:] = h_f/6
        A[1,1:-1] = 2*h_f/3
        A[1,[0,-1]] = h_f/3
        A[2,:-1] = h_f/6

        # B = np.zeros((n_f, n_f))
        # B[range(0,n_f-1), range(1,n_f)] = self.v/h_f
        # B[range(1,n_f-1), range(1,n_f-1)] = -2*self.v/h_f
        # B[range(1,n_f), range(n_f-1)] = self.v/h_f
        # B[[0,-1],[0,-1]] = -self.v/h_f

        B = np.zeros((3, n_f))
        B[0, 1:] = self.v/h_f
        B[1, 1:-1] = -2*self.v/h_f
        B[1,[0,-1]] = -self.v/h_f
        B[2, :-1] = self.v/h_f

        # C = np.zeros((n_f, n_f))
        # g_x = np.zeros(n_f)
        # for i in range(n_f-1):
        #     g_x[i] = quad(lambda x: G_x(x+self.lb)*x, a=i*h_f, b=(i+1)*h_f)[0]/h_f
        # for i in range(1, n_f-1):
        #     C[i,i-1] = g_x[i-1] - i*(G_values[i] - G_values[i-1])
        #     C[i,i] = (i-1) * (G_values[i]-G_values[i-1]) + (i+1) * (G_values[i+1]-G_values[i]) - g_x[i-1] - g_x[i]
        #     C[i,i+1] = g_x[i] - i*(G_values[i+1]-G_values[i])
        # C /= h_f

        C = np.zeros((3,n_f))
        g_x = np.zeros(n_f)
        for i in range(n_f-1):
            g_x[i] = quad(lambda x: G_x(x+self.lb)*x, a=i*h_f, b=(i+1)*h_f)[0]/h_f
        for i in range(1, n_f-1):
            C[0,i+1] =  g_x[i] - i*(G_values[i+1]-G_values[i])
            C[1,i] = (i-1) * (G_values[i]-G_values[i-1]) + (i+1) * (G_values[i+1]-G_values[i]) - g_x[i-1] - g_x[i]
            C[2,i-1] = g_x[i-1] - i*(G_values[i] - G_values[i-1])
        C /= h_f
        
        # D = np.zeros((n_f, n_f))
        # a_x = np.zeros(n_f)
        # for i in range(n_f-1):
        #     a_x[i] = quad(lambda x: alpha_x(x+self.lb)*x, a=i*h_f, b=(i+1)*h_f)[0]/h_f
        # for i in range(1, n_f-1):
        #     D[i,i-1] = a_x[i-1] - i*(alpha_values[i] - alpha_values[i-1])
        #     D[i,i] = (i-1) * (alpha_values[i]-alpha_values[i-1]) + (i+1) * (alpha_values[i+1]-alpha_values[i]) - a_x[i-1] - a_x[i]
        #     D[i,i+1] = a_x[i] - i*(alpha_values[i+1]-alpha_values[i])
        # D /= h_f

        D = np.zeros((3,n_f))
        a_x = np.zeros(n_f)
        for i in range(n_f-1):
            a_x[i] = quad(lambda x: alpha_x(x+self.lb)*x, a=i*h_f, b=(i+1)*h_f)[0]/h_f
        for i in range(1, n_f-1):
            D[0,i+1] =  a_x[i] - i*(alpha_values[i+1]-alpha_values[i])
            D[1,i] = (i-1) * (alpha_values[i]-alpha_values[i-1]) + (i+1) * (alpha_values[i+1]-alpha_values[i]) - a_x[i-1] - a_x[i]
            D[2,i-1] = a_x[i-1] - i*(alpha_values[i] - alpha_values[i-1])
        D /= h_f

        a = np.zeros(n_f)
        a[0] = quad(lambda x: self.p0(x+self.lb)*(1-x/h_f), a=0.0, b=h_f)[0]
        a[-1] = quad(lambda x: self.p0(x+self.lb)*(x/h_f-n_f+1), a=(n_f-1)*h_f, b=n_f*h_f)[0]
        for i in range(1,n_f-1):
            a[i] = quad(lambda x: self.p0(x+self.lb)*(x/h_f-i+1), a=(i-1)*h_f, b=i*h_f)[0]
            a[i] += quad(lambda x: self.p0(x+self.lb)*(i+1-x/h_f), a=i*h_f, b=(i+1)*h_f)[0]

        #rho_0 = np.linalg.solve(A, a)
        rho_0 = solve_banded(l_and_u=(1,1), ab=A, b=a)
        #A[[0,-1], :] = 0.0
        A[[0,1,1,2],[1,0,-1,-2]] = 0.0

        # b = np.zeros((N_t, n_f))
        b = np.zeros((N_t, 2))        
        b[:,0] = V_x(*np.meshgrid([self.lb], np.arange(0, self.T, h_t) + h_t))[:,0] - self.v/h_f
        b[:,1] = self.v/h_f

        #c = np.zeros((N_t, n_f))
        c = np.zeros((N_t, 2))
        #c[:,-1] = V_x(*np.meshgrid([self.ub], np.arange(0, self.T, h_t) + h_t))[:,0] + self.v/h_f
        #c[:,-2] = -self.v/h_f
        c[:,1] = V_x(*np.meshgrid([self.ub], np.arange(0, self.T, h_t) + h_t))[:,0] + self.v/h_f
        c[:,0] = -self.v/h_f

        return A, B, C, D, b, c, rho_0, h_f

    def _pre_calculations_1d_chang_finite_difference(self, N_x, N_t):
        """
        Perform the matrix calculations for the 1d solving problem - Finite Differences by Chang.
        """

        h_x = self.X/N_x
        h_t = self.T/N_t

        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)
        V_x = lambda x,t: G_x(x) + alpha_x(x) * self.u(t)
        V = lambda x,t: self.G(x) + self.alpha(x) * self.u(t)

        x_var = np.arange(self.lb, self.ub+0.9*h_x, h_x)
        rho_0 = self.p0(x_var)

        t_var = np.arange(0, self.T, h_t)
        B = V_x(*np.meshgrid(x_var[:-1]+0.5*h_x, t_var))
        rho_bar = np.exp(-V(*np.meshgrid(x_var+0.5*h_x, t_var+h_t)))
        delta = self.v/h_x * (1/B) - 1/(rho_bar[:,:-1]/rho_bar[:,1:] - 1)

        return delta, B, rho_0, h_x, h_t

    def _pre_calculation_1d_spectral_collocation(self, n_p):
        """
        Perform the matrix calculations for the 1d solving problem - Collocation method and spectral.
        """

        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)
        G_xx = egrad(G_x)
        alpha_xx = egrad(alpha_x)

        # 1. Find the collocation points 
        poly = legendre(n_p+1) - legendre(n_p-1)
        collocation_points = np.sort(np.roots(poly))
        collocation_points = 0.5*(self.ub-self.lb)*collocation_points + 0.5*(self.ub+self.lb)
        collocation_points[0] = self.lb
        collocation_points[-1] = self.ub

        # 2. Calculate the M matrix
        Gx_values = G_x(collocation_points)
        alphax_values = alpha_x(collocation_points)
        Gxx_values = G_xx(collocation_points)
        alphaxx_values = alpha_xx(collocation_points)

        phi_x = np.zeros((n_p+1, n_p+1))
        phi_xx = np.zeros((n_p+1, n_p+1))

        poly_objs = []

        w_vec = np.zeros(n_p+1)
        for j in range(n_p+1):
            w_vec[j] = 1.0
            poly = lagrange(x=collocation_points, w=w_vec)
            w_vec[j] = 0.0
            poly_objs.append(poly)
            phi_x[:, j] = np.polyder(poly, 1)(collocation_points)
            phi_xx[:, j] = np.polyder(poly, 2)(collocation_points)
            
        M1 = self.v * phi_xx + (phi_x.T*Gx_values).T + np.diag(Gxx_values)
        M2 = (phi_x.T*alphax_values).T + np.diag(alphaxx_values)

        # 3. Calculate the boundary conditions
        boundaries = [self.v * phi_x[[0,-1],:],  Gx_values[[0,-1]], alphax_values[[0,-1]]]

        return M1, M2, poly_objs, collocation_points, boundaries

    def _solve1d_spectral_collocation(self, n_p, N_x, N_t):

        h_t = self.T/N_t
        h_x = self.X/N_x
        identity = np.eye(n_p+1)

        M1, M2, poly_objs, collocation, boundaries = self._pre_calculation_1d_spectral_collocation(n_p)

        rho_vec = np.zeros((N_t+1, n_p+1))
        rho_vec[0,:] = self.p0(collocation)
        for i in tqdm(range(N_t)):
            previous_matrix = identity + 0.5*h_t*(M1 + self.u(h_t*i) * M2)
            next_matrix = identity - 0.5*h_t*(M1 + self.u(h_t*(i+1)) * M2)
            previous_matrix[[0,-1],:] = 0.0
            rho_vec[i+1,:] = previous_matrix @ rho_vec[i,:]
            next_matrix[[0,-1],:] = boundaries[0]
            next_matrix[[0,-1],[0,-1]] += boundaries[1] + boundaries[2] * self.u(h_t*(i+1))
            rho_vec[i+1,:] = np.linalg.solve(next_matrix, rho_vec[i+1,:])
            
        phi_matrix = np.zeros((n_p+1, N_x+1))
        x = np.arange(0.0, self.X+0.9*h_x, h_x)
        for k in range(n_p+1):
            phi_matrix[k,:] = poly_objs[k](x)
        rho = rho_vec @ phi_matrix
        return rho

    def _solve1d_chang_finite_difference(self, N_x, N_t):
        delta, B, rho_0, h_x, h_t = self._pre_calculations_1d_chang_finite_difference(N_x, N_t)
        A = np.zeros((N_x+1,N_x+1))
        rho = np.zeros((N_t+1, N_x+1))
        rho[0,:] = rho_0
        for i in tqdm(range(N_t)):
            A[range(1,N_x), range(0,N_x-1)] = self.v/h_x - delta[i,:-1] * B[i,:-1]
            A[range(1,N_x), range(1,N_x)] = -2*self.v/h_x - (1-delta[i,:-1]) * B[i,:-1] + delta[i,1:] * B[i,1:] 
            A[range(1,N_x), range(2,N_x+1)] = self.v/h_x + (1-delta[i,1:]) * B[i,1:]
            A *= h_t/h_x
            A = np.eye(N_x+1) - A
            A[0, [0,1]] = [delta[i,0] * B[i,0] - self.v/h_x, (1 - delta[i,0]) * B[i,0] + self.v/h_x]
            A[-1, [-2,-1]] = [delta[i,-1] * B[i,-1] - self.v/h_x, (1 - delta[i,-1]) * B[i,-1] + self.v/h_x]
            rho_save = np.copy(rho[i,[0,-1]])
            rho[i,[0,-1]] = 0.0
            rho[i+1,:] = np.linalg.solve(A, rho[i,:])
            rho[i,[0,-1]] = rho_save
        return rho

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

    def _solve1d_spectral_galerkin(self, n_f, N_x, N_t):
        
        A, B, C, D, rho_0, legendre_family, boundary_conditions = self._pre_calculations_1d_spectral_galerkin(n_f)
        h_t = self.T/N_t
        h_x = self.X/N_x

        rho_vec = np.zeros((N_t+1, n_f+1))
        rho_vec[0,:] = rho_0

        for i in tqdm(range(N_t)):
            previous_matrix = (A - 0.5*h_t*(B+C+self.u(h_t*i)*D)) 
            previous_matrix[[-2,-1], :] = 0.0
            rho_vec[i+1,:] = previous_matrix @ rho_vec[i,:]
            next_matrix = A + 0.5*h_t*(B+C+self.u(h_t*(i+1))*D)
            next_matrix[-2,:] = boundary_conditions[0] + self.u((i+1)*h_t) * boundary_conditions[2]
            next_matrix[-1,:] = boundary_conditions[1] + self.u((i+1)*h_t) * boundary_conditions[3]
            rho_vec[i+1,:] = np.linalg.solve(next_matrix, rho_vec[i+1,:])
        phi_matrix = np.zeros((n_f+1, N_x+1))
        x = np.arange(-1.0, 1.0+1.8/N_x, 2/N_x)
        for k in range(n_f+1):
            phi_matrix[k,:] = legendre_family[k](x)
        rho = rho_vec @ phi_matrix

        return rho

    def _solve1d_galerkin_finite_elements(self, n_f, N_x, N_t):
        
        n_f += 1
        A, B, C, D, b, c, rho_0, h_f = self._pre_calculations_1d_galerkin_finite_elements(n_f, N_t)
        h_t = self.T/N_t
        h_x = self.X/N_x

        rho_vec = np.zeros((N_t+1, n_f))
        rho_vec[0,:] = rho_0

        for i in tqdm(range(N_t)):

            previous_matrix = A + 0.5*h_t*(B + C + self.u((i+1)*h_t) * D)
            next_matrix = A - 0.5*h_t*(B + C + self.u((i+1)*h_t) * D)
            #next_matrix[0,:] = b[i,:]
            #next_matrix[-1,:] = c[i,:]
            next_matrix[[1,0], [0,1]] = b[i,:]
            next_matrix[[2,1],[-2,-1]] = c[i,:]
            #rho_vec[i+1,:] = previous_matrix @ rho_vec[i,:]
            #rho_vec[i+1,:] = np.linalg.solve(next_matrix, rho_vec[i+1,:])
            rho_vec[i+1,:] = np.roll(previous_matrix[0,:]*rho_vec[i,:], -1)
            rho_vec[i+1,:] += previous_matrix[1,:]*rho_vec[i,:] 
            rho_vec[i+1,:] += np.roll(previous_matrix[2,:]*rho_vec[i,:], 1)
            rho_vec[i+1,:] = solve_banded(l_and_u=(1,1), ab=next_matrix, b=rho_vec[i+1,:])

        phi_matrix = np.zeros((n_f, N_x+1))
        x = np.arange(0.0, self.X+0.9*h_x, h_x)
        for k in range(n_f):
            phi_matrix[k,:] = (x/h_f - (k-1)) * (x >= (k-1)*h_f) * (x < k*h_f) 
            phi_matrix[k,:] += (k+1 - x/h_f) * (x >= k*h_f) * (x <= (k+1)*h_f)
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
        elif type == 'chang_cooper':
            rho = self._solve1d_chang_finite_difference(N_x, N_t)
        elif type == 'spectral_galerkin':
            rho = self._solve1d_spectral_galerkin(n_f, N_x, N_t)
        elif type == 'galerkin_fem':
            rho = self._solve1d_galerkin_finite_elements(n_f, N_x, N_t)
        elif type == 'spectral_collocation':
            rho = self._solve1d_spectral_collocation(n_f, N_x, N_t)
        return rho

class ControlFPequation:

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
        self.p_infty = self._calculus_steady_state()

    def _calculus_steady_state(self):
        """
        The steady state is p(x) = c e^(-G(x)/v) where c is chosen so as to p(x) has integral equal to 1.
        """
        c = 1/quad(lambda x: np.exp(-self.G(x)/self.v), a=self.lb, b=self.ub)[0]
        p_infty = lambda x: c*np.exp(-self.G(x)/self.v)
        return p_infty

    def _solve_system_coefs_spectral_legendre(self, n_f, Gxl, Gxu):

        coef = np.zeros((n_f-1,2))
        for k in range(n_f-1):
            M = np.array([
                [self.v*(k+1)*(k+2) - 2*Gxl, -self.v*(k+2)*(k+3) + 2*Gxl],
                [self.v*(k+1)*(k+2) + 2*Gxu, self.v*(k+2)*(k+3) + 2*Gxu],
            ])
            m = np.array([self.v*k*(k+1) - 2*Gxl, -self.v*k*(k+1) - 2*Gxu])
            coef[k] = np.linalg.solve(M, m)
        return coef

    def _update_with_boundary_conditions(self, i, j, M, coef):
        aux = M[i,j] + coef[i,0]*M[i+1,j] + coef[i,1]*M[i+2,j] 
        aux += coef[j,0]*M[i,j+1] + coef[i,0]*coef[j,0]*M[i+1,j+1] + coef[i,1]*coef[j,0]*M[i+2,j+1]
        aux += coef[j,1]*M[i,j+2] + coef[i,0]*coef[j,1]*M[i+1,j+2] + coef[i,1]*coef[j,1]*M[i+2,j+2]
        return aux

    def _pre_calculations_1d_spectral_legendre(self, n_f):
        """
        Perform the matrix calculations for the 1d solving problem - Spectral Legendre-Galerkin method.
        """
        legendre_family = [legendre(k) for k in range(n_f+1)]
        legendre_family_diff = [np.polyder(poly, 1) for poly in legendre_family]
        G_x = egrad(self.G)
        alpha_x = egrad(self.alpha)

        # Calculate boundary conditions restrictions
        coef = self._solve_system_coefs_spectral_legendre(n_f, G_x(self.lb), G_x(self.ub))

        y0 = lambda x: self.p0(x) - self.p_infty(x)

        # Calculate matrix preparation matrices
        L = np.diag(2/(2*np.linspace(0,n_f,n_f+1)+1))
        L_dot = np.zeros((n_f+1, n_f+1))
        G_dot_L = np.zeros((n_f+1, n_f+1))
        alpha_dot_L = np.zeros((n_f+1, n_f+1))
        steady_L = np.zeros(n_f+1)
        y0_L = np.zeros(n_f+1)
        for i in tqdm(range(n_f+1)):
            for j in range(n_f+1):
                L_dot[i,j] = min(i,j)*(min(i,j)+1)*((i+j)%2==0)
                G_dot_L[i,j] = quad(func=lambda x: G_x(0.5*(self.X*x + self.lb+self.ub))*legendre_family[j](x)*legendre_family_diff[i](x),
                                    a=-1, b=1)[0]
                alpha_dot_L[i,j] = quad(func=lambda x: alpha_x(0.5*(self.X*x + self.lb+self.ub))*legendre_family[j](x)*legendre_family_diff[i](x),
                                        a=-1, b=1)[0]
            steady_L[i] = quad(func=lambda x: alpha_x(0.5*(self.X*x + self.lb+self.ub))*self.p_infty(0.5*(self.X*x + self.lb+self.ub))*legendre_family_diff[i](x), 
                               a=-1, b=1)[0]
            y0_L[i] = quad(func=lambda x: y0(0.5*(self.X*x + self.lb+self.ub))*legendre_family[i](x), 
                              a=-1, b=1)[0]
        
        # Update matrices with boundary conditions
        Phi = np.zeros((n_f-1, n_f-1))
        Lambda = np.zeros((n_f-1, n_f-1))
        Theta1 = np.zeros((n_f-1, n_f-1))
        Theta2 = np.zeros((n_f-1, n_f-1))
        v = np.zeros(n_f-1)
        b = np.zeros(n_f-1)
        for i in range(n_f-1):
            for j in range(n_f-1):
                Phi[i,j] = self._update_with_boundary_conditions(i, j, L, coef)
                Lambda[i,j] = self.v*self._update_with_boundary_conditions(i, j, L_dot, coef)
                Theta1[i,j] = self._update_with_boundary_conditions(i, j, G_dot_L, coef)
                Theta2[i,j] = self._update_with_boundary_conditions(i, j, alpha_dot_L, coef)
            v[i] = steady_L[i] + coef[i,0]*steady_L[i+1] + coef[i,1]*steady_L[i+2]
            b[i] = y0_L[i] + coef[i,0]*y0_L[i+1] + coef[i,1]*y0_L[i+2]

        # Initial condition
        y0 = np.linalg.solve(Phi, b)
        
        return Phi, Lambda, Theta1, Theta2, v, y0, legendre_family, coef

    def _solve_ricatti(self, A, B, M):
        Pi = solve_continuous_are(a=A, b=B, q=M, r=1)
        return Pi

    def _solve1d_spectral_legendre(self, n_f, N_x, N_t, controlled=True):
        
        Phi, Lambda, Theta1, Theta2, v, y0, legendre_family, coef = self._pre_calculations_1d_spectral_legendre(n_f)
        Phi_inv = np.linalg.inv(Phi)
        A = -4*(Lambda + Theta1)/self.X**2
        B = -4*v.reshape((-1,1))/self.X**2
        # Supposes M is the identity transform.
        Pi = self._solve_ricatti(A, B, Phi)

        h_t = self.T/N_t
        h_x = self.X/N_x

        if controlled:
            sol = solve_ivp(fun=lambda t,y: Phi_inv@(A-B@B.T@Pi+Theta2*(B.T@Pi@y))@y,
                            t_span=(0,self.T), 
                            t_eval=np.linspace(0,self.T, N_t+1),
                            y0=y0)
        else:
            sol = solve_ivp(fun=lambda t,y: Phi_inv@A@y,
                            t_span=(0,self.T), 
                            t_eval=np.linspace(0,self.T, N_t+1),
                            y0=y0)
        y_vec = sol.y.T
        phi_matrix = np.zeros((n_f-1, N_x+1))
        x = np.arange(-1.0, 1.0+1.8/N_x, 2/N_x)
        for k in range(n_f-1):
            phi_matrix[k,:] = legendre_family[k](x) + coef[k,0]*legendre_family[k+1](x) + + coef[k,1]*legendre_family[k+2](x)
        y = y_vec @ phi_matrix

        return y      

if __name__ == '__main__':

    G_func = lambda x: x*x
    alpha_func = lambda x: x**2*(1/2-x/3)
    control = lambda t: 1.0
    #p_0 = lambda x: np.exp(-x*x)/(np.sqrt(np.pi)*(norm.cdf(np.sqrt(2)) - 0.5))
    #p_0 = lambda x: 140 * x**3 * (1-x)**3
    #p_0 = lambda x: truncnorm(a=-0.5/1e-2, b=0.5/1e-2, loc=0.5, scale=1e-2).pdf(x)
    def p_0(x):
        l = 0.01
        h = 1/l
        return h/l * (abs(x-0.5) < l) * (l + (x-0.5)*(x<=0.5) - (x-0.5)*(x>0.5))

    interval = [0.0, 1.0]
    parameters = {'v': 0.1, 'T': 1.0, 'p_0': p_0, 'interval': interval}

    #FP_equation = FokkerPlanckEquation(G_func, alpha_func, control, parameters)

    # solving1 = FP_equation.solve1d(N_x=100, N_t=60000, type='forward')
    # solving2 = FP_equation.solve1d(N_x=200, N_t=3000, type='ck')
    # solving3 = FP_equation.solve1d(n_f=15, N_x=200, N_t=15000, type='spectral_galerkin')
    # solving4 = FP_equation.solve1d(n_f=500, N_x=500, N_t=30000, type='galerkin_fem')
    # solving5 = FP_equation.solve1d(N_x=200, N_t=15000, type='chang_cooper')
    # solving6 = FP_equation.solve1d(n_f=20, N_x=200, N_t=15000, type='spectral_collocation')
    
    FP_equation = ControlFPequation(G_func, alpha_func, control, parameters)
    solving1 = FP_equation._solve1d_spectral_legendre(n_f=20, N_x=1000, N_t=200)
    solving2 = FP_equation._solve1d_spectral_legendre(n_f=20, N_x=1000, N_t=200, controlled=False)

    x = np.linspace(0, 1, 1001)
    t = np.linspace(0, 1, 201)
    X, T = np.meshgrid(x,t)

    # Plotting the 3d figure
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, T, solving1)
    # ax.plot_surface(X, T, solving2)
    # ax.plot_surface(X, T, solving3)
    # ax.plot_surface(X, T, solving4)
    # ax.plot_surface(X, T, solving5)
    # ax.plot_surface(X, T, solving6)
    # ax.set_xlabel('x')
    # ax.set_ylabel('t')
    # ax.set_zlabel('y')
    # plt.show() 

    # Plotting the integral of x-axis for each time
    plt.plot(t, np.mean(solving1**2, axis=1), label='controlled')
    plt.plot(t, np.mean(solving2**2, axis=1), label='uncontrolled')
    plt.yscale('log')
    plt.legend()
    plt.show()   

    # Plotting the integral of x-axis for each time
    # plt.plot(t, solving1.sum(axis=1)/200, label='forward')
    # plt.plot(t, solving2.sum(axis=1)/200, label='ck')
    # plt.plot(t, solving3.sum(axis=1)/200, label='legendre galerkin')
    # plt.plot(t, solving4.sum(axis=1)/500, label='galerkin')
    # plt.plot(t, solving5.sum(axis=1)/200, label='chang')
    # plt.plot(t, solving6.sum(axis=1)/200, label='spectral')
    # plt.legend()
    # plt.show() 
        