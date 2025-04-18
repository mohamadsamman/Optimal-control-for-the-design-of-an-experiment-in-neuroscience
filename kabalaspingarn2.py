import numpy as np
from scipy.integrate import solve_ivp

# Constants from the problem setup
a = 0.78
b = 0.208
c = 4.34
d = 2.92
E = 0.0489402  # Energy constraint
T = 1.02  # Terminal time (in hours)
q_init = 0.02  # Initial guess for Lagrange multiplier q

# Sensitivity ODE system (state and costate)
def sensitivities_ODE(t, Z, S, C):
    """
    Computes the time derivative of sensitivities (∂Z/∂Ci).
    Inputs:
        Z: current state vector [x1, x2, xb1, xb2, q, p1, p2, p3, p4, p5]
        S: sensitivity matrix (flattened form)
        C: current guess vector [p1(0), ..., p4(0), q(0)]
    Returns:
        dSdt: flattened derivative of sensitivity matrix
    """
    # Unpack state and sensitivity variables
    x1, x2, xb1, xb2, q, p1, p2, p3, p4, p5 = Z[:10]
    S = S.reshape((10, 5))  # Convert to 2D matrix
    
    dSdt = np.zeros((10, 5))  # Prepare the derivative matrix
    d_beta_dCi = np.zeros((10, 5))  # φ_ci (beta_ci)
    
    for i in range(5):
        # ∂/∂Ci terms only contribute to p2_dot via β (nonzero at row 6)
        p2_ci = S[6, i]   # ∂p2/∂Ci
        q_ci = S[4, i]    # ∂q/∂Ci
        beta_ci = -(1 / q**2) * p2_ci + (2 / q**3) * q_ci * p2
        d_beta_dCi[6, i] = beta_ci

    for i in range(5):
        dZ_dCi = S[:, i]

        dx1, dx2, dxb1, dxb2, dq, dp1, dp2, dp3, dp4, dp5 = dZ_dCi
        
        # Compute ∂y/∂Ci
        if q != 0:
            dy_dCi = - (p2_ci * q - p2 * q_ci) / q**2
        else:
            dy_dCi = 0.0

        # Derivatives of state equations
        dSdt[0, i] = -a * dx1 - b * dx2 + dy_dCi
        dSdt[1, i] = c * dx1 - d * dx2
        dSdt[2, i] = dSdt[0, i]  # dxb1' = x1'
        dSdt[3, i] = dSdt[1, i]  # dxb2' = x2'
        dSdt[4, i] = 0  # dq' = 0 always

        # Derivatives of costate equations
        dSdt[5, i] = a * dp3 - c * dp4
        dSdt[6, i] = b * dp3 + d * dp4 + d_beta_dCi[6, i]
        dSdt[7, i] = -dxb1  # ∂p3/∂t = -xb1
        dSdt[8, i] = -dxb2  # ∂p4/∂t = -xb2
        dSdt[9, i] = 0  # p5 always constant

    return dSdt.flatten()

# Define the system of ODEs
def system_of_ODEs(t, Z, C):
    """
    Defines the augmented state vector and its derivatives (state + costate).
    """
    x1, x2, xb1, xb2, q, p1, p2, p3, p4, p5 = Z#[:10]
    # State dynamics
    dx1 = -a * x1 - b * x2 + xb2
    dx2 = c * x1 - d * x2 + xb1
    
    # Costate dynamics
    dp1 = -a * p3 + c * p4
    dp2 = -b * p3 - d * p4 + (1 / q**2) * p2
    dp3 = -x1
    dp4 = -x2
    dp5 = 0

    return [dx1, dx2, xb1, xb2, 0, dp1, dp2, dp3, dp4, dp5]

# Runge-Kutta method to solve ODEs numerically
def runge_kutta_integration(system, t_span, Z0, dt):
    """
    Solves the ODE system using a 4th-order Runge-Kutta method.
    """
    times = np.arange(t_span[0], t_span[1], dt)
    Z = np.zeros((len(times), len(Z0)))  # Solution array
    Z[0, :] = Z0
    
    for i in range(1, len(times)):
        t = times[i-1]
        z = Z[i-1, :]
        
        k1 = np.array(system(t, z, C=None))
        k2 = np.array(system(t + dt / 2, z + dt / 2 * k1, C=None))
        k3 = np.array(system(t + dt / 2, z + dt / 2 * k2, C=None))
        k4 = np.array(system(t + dt, z + dt * k3, C=None))
        
        Z[i, :] = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return times, Z

# Newton-Raphson optimization loop
def newton_raphson_optimization():
    """
    Performs the Newton-Raphson method to optimize the initial conditions.
    """
    # Initial guesses for the parameters
    C_init = np.array([0.0, 0.0, 0.0, 0.0, q_init])  # Initial guesses for [p1(0), ..., p4(0), q(0)]
    Z_init = np.array([0.1, 0.1, 0.1, 0.1, q_init, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state guess
    
    # Time span for integration
    t_span = [0, T]
    dt = 1 / 50  # Time step for numerical integration
    
    # Runge-Kutta integration to solve the ODE system
    times, Z = runge_kutta_integration(system_of_ODEs, t_span, Z_init, dt)
    
    # Newton-Raphson loop (iteration)
    for iteration in range(10):
        # Compute the sensitivities at each time step
        S = np.zeros((len(times), 5))  # Sensitivity matrix [10 x 5] flattened
        for i in range(5):
            S[:, i] = np.array(sensitivities_ODE(times, Z, S, C_init))
        
        # Update the initial conditions using the Newton-Raphson step
        dC = np.linalg.solve(S.T @ S, S.T @ (Z[-1, :] - Z_init))  # Linear solve for C
        C_init += dC
    
    return C_init, Z

# Main function to run the optimization and simulation
if __name__ == "__main__":
    # Run the Newton-Raphson optimization
    C_final, Z_final = newton_raphson_optimization()
    print("Optimized initial conditions: ", C_final)
    print("Final state vector: ", Z_final[-1, :])

