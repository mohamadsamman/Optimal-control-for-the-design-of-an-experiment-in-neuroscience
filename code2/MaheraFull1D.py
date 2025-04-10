import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import expm
import scipy as sp
import sympy as sy



np.random.seed(10)

def constructTime(t0=0, tmax=1, npoints=100):
    """Construct time points for simulation."""
    return np.linspace(t0, tmax, npoints)


def experiment(A, B, C, s2, x0, tpoints, u):
    """
    Simulates the system:
        x_dot = A * x + g(p) * u
        y = C * x 
    where v ~ N(M, s2) is white Gaussian noise.
    
    Parameters:
    - B: a function of p that defines g.
    - p: a parameter that influences g.

    returns
    y 
    ymes

    """
    np.random.seed(10)
    dt = tpoints[1] - tpoints[0]  # Time step
    x = np.zeros_like(tpoints)  
    y = np.zeros_like(tpoints)  
    ymes = np.zeros_like(tpoints)  # Measurement trajectory
    
    x[0] = x0  # Set initial condition

    for i in range(1, len(tpoints)):
        # Euler integration of x_dot = f * x + g(p) * u
        x[i] = x[i-1] + dt * (A * x[i-1] + B * u[i-1])
        # Measurement with noise
        y[i] = C * x[i] 
        ymes[i] = C * x[i] + np.random.normal(0, np.sqrt(s2))

    return x, y, ymes

def getypred(tpoints,u,x0,Ahat,Bhat,C):
    dt = tpoints[1] - tpoints[0]
    x = np.zeros_like(tpoints)
    ypred = np.zeros_like(tpoints)  
    x[0] = x0
    for i in range(1, len(tpoints)):
        # Euler integration of x_dot = f * x + g(p) * u
        x[i] = x[i-1] + dt * (Ahat * x[i-1] + Bhat * u[i-1])
        # Measurement with noise
        ypred[i] = C * x[i]

    return ypred

def cf(Ahat,Bhat,A, B, C, M, s2, x0, tpoints, u ,p):
    x, y, ymes = experiment(A, B, C, s2, x0, tpoints, u)
    ypred = getypred(tpoints,u,x0,Ahat,Bhat,C)
    cost = np.sum((ymes-ypred)*(ymes-ypred))
    return cost 

def Mvalues(s2hat):
    # Calculate the upper bound for M
    upper_bound = 1 / s2hat

    # Create a list of possible values for M within the range
    M_values = np.linspace(-upper_bound, upper_bound, 1000) 

    # Check if the condition abs(M * s2hat) < 1 holds
    Mvalues = [M for M in M_values if abs(M * s2hat) < 1]
    return Mvalues

def findM(s2, Mvalues, p):
    # Définir l'EDO
    def edo(t,S, M, s2):
        return - S * S * (1./s2) - 2. * S - 1. / M
    S0 = 0.


    # Stocker les résultats
    M_list = []
    lower_left_list = []

    # Intervalle de temps
    t_span = (0, 1.)
    t_eval = np.linspace(t_span[0], t_span[1], 1000000)

    for M in Mvalues:
        # Résolution de l'EDO
        sol = solve_ivp(edo, t_span, [S0], args=(M, s2), t_eval=t_eval)

        # Trouver le moment où S(t) descend en dessous de -60
        t_crit = None
        for i in range(1, len(sol.t)):
            if np.abs(sol.y[0][i]) > 60:
                t_crit = sol.t[i]
                break
  
        T = t_crit
        print('tcrit:',T)
        # Définir la matrice A
        A = np.array([[-1, -1/M],
                    [1/s2, 1]])

        # Calcul de exp(AT)
        exp_AT = expm(T*A)
        # print("La matrice exp(AT) est :")
        # print(exp_AT)
        # print("lambda0 ",np.sqrt(M*M/t_crit))

        # Extraire l'élément en bas à gauche (indice [1,0])
        lower_left_value = exp_AT[1, 1]

        # Stocker les valeurs
        M_list.append(M)
        lower_left_list.append(lower_left_value)

    # Tracer le graphique
    plt.figure(figsize=(8, 6))
    plt.plot(M_list, np.abs(lower_left_list), marker='o', linestyle='-')
    plt.xlabel('M')
    plt.ylabel('Élément inférieur gauche de exp(AT)')
    plt.title('Relation entre M et l\'élément inférieur gauche de exp(AT)')
    plt.grid(True)
    plt.show()
    return M





# Parameters 
n=1
itern=1.
tol = 2.


# misc
x0 = 0.
lambda0 = 1.
s0 = 0.

M = 155.
s2 = 1.e-4
s2hat = 1.e-4
t_span = (0, 1.) 
t_eval = np.linspace(t_span[0], t_span[1], 100000)


Ahat = -1.
Bhat = 1.
# t0 = 0.
# tmax = 1.
# npoints = 100000
A = -1.
B = 1.
C = 1.

tpoints = t_eval
u = np.ones_like(tpoints)
p = 1.e-4

# Function initialization
ypred=np.zeros(n)
ymes=np.zeros(n)

cost = 100.
iter = 0.
while( cost > tol and iter < itern):
    cost   = cf(Ahat, Bhat, A, B, C, M, s2, x0, tpoints, u,p)
    print(cost)
    iter = iter + 1
    # Mvalues = Mvalues(s2hat)
    #M = findM(s2hat, Mvalues, p)
    
    initial_guess=1.e-4
    # Call the minimize function using the Nelder-Mead method
    result = minimize(cf, initial_guess,method='Nelder-Mead',args=(Ahat, Bhat, A, B, C, M, s2, x0, tpoints, u), )

    # Output the results
    print("Optimal parameters:", result.x)
    print("Minimum value:", result.fun)
    print("Optimization success:", result.success)
    print("Exit message:", result.message)
    