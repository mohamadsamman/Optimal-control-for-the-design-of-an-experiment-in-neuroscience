import numpy as np
from scipy.integrate import solve_ivp

# Constants and Parameters (Bolie)
a = 0.78
b = 0.208
c = 4.34
d = 2.92
T = 1.0  # hours
E = 0.0489402
dt = 1/50  # time step in hours
time_grid = np.arange(0, T + dt, dt)

# Initial state conditions
x1_0 = 0.0
x2_0 = 0.1
xb1_0 = 0.0
xb2_0 = 0.0

# Newton-Raphson setup
max_iters = 10
tol = 1e-6
C = np.array([0.0, 0.0, 0.0, 0.0, 0.02])  # [p1(0), ..., p4(0), q(0)]

def system_ODE(t, Z, C):
    """
    Defines the system of state + costate ODEs.
    Z = [x1, x2, xb1, xb2, q, p1, p2, p3, p4, p5]
    """
    x1, x2, xb1, xb2, q, p1, p2, p3, p4, p5 = Z
    y = -p2 / q  # Optimal control law

    # State equations
    dx1 = -a * x1 - b * x2 + y
    dx2 = c * x1 - d * x2
    dxb1 = dx1
    dxb2 = dx2

    dq = 0.0

    # Costate equations (negative adjoint)
    dp1 = -p3 * (-a) - p4 * c
    dp2 = -p3 * (-b) - p4 * (-d) + (1 / q**2) * p2**2
    dp3 = -xb1
    dp4 = -xb2
    dp5 = 0.0  # Dummy for q-energy constraint

    return [dx1, dx2, dxb1, dxb2, dq, dp1, dp2, dp3, dp4, dp5]

def sensitivities_ODE(t, S, Z, C):
    """
    Computes the time derivative of sensitivities (∂Z/∂Ci).
    Inputs:
        S: 10x5 sensitivity matrix ∂Z/∂Ci
        Z: current state vector [x1, x2, xb1, xb2, q, p1, p2, p3, p4, p5]
        C: current guess vector [p1(0), ..., p4(0), q(0)]
    Returns:
        dSdt: flattened derivative of sensitivity matrix
    """
    x1, x2, xb1, xb2, q, p1, p2, p3, p4, p5 = Z
    S = S.reshape((10, 5))  # Convert to 2D matrix

    # Compute partial derivatives of the system (Jacobian A)
    # Define dH/dZ for the costate p2 equation (Eq. 29 + 49)
    d_beta_dCi = np.zeros((10, 5))  # φ_ci (beta_ci)
    
    for i in range(5):
        # ∂/∂Ci terms only contribute to p2_dot via β (nonzero at row 6)
        # dβ/dCi = - (1/q^2) * p2_ci + (2/q^3) * q_ci * p2
        p2_ci = S[6, i]   # ∂p2/∂Ci
        q_ci = S[4, i]    # ∂q/∂Ci
        beta_ci = -(1 / q**2) * p2_ci + (2 / q**3) * q_ci * p2
        d_beta_dCi[6, i] = beta_ci

    # Linearized system dS/dt = A(t) @ S + d_beta_dCi
    dSdt = np.zeros((10, 5))

    for i in range(5):
        # Unpack the ith column (∂Z/∂Ci)
        dZ_dCi = S[:, i]

        # Extract components of the derivative vector
        dx1, dx2, dxb1, dxb2, dq, dp1, dp2, dp3, dp4, dp5 = dZ_dCi

        # Compute ∂y/∂Ci = - (p2_ci * q - p2 * q_ci) / q^2
        if q != 0:
            dy_dCi = - (p2_ci * q - p2 * q_ci) / q**2
        else:
            dy_dCi = 0.0  # To avoid division by zero; safeguard

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


def integrate_full_system(C):
    """
    Integrate the system and its sensitivities for a given C.
    Returns terminal p values and sensitivity matrix J.
    """
    Z0 = [x1_0, x2_0, xb1_0, xb2_0, C[4], C[0], C[1], C[2], C[3], 0.0]
    S0 = np.zeros((10, 5))
    for i in range(5):
        S0[i + 5, i] = 1  # Sensitivity of pi(0) wrt Ci
        if i == 4:
            S0[4, i] = 1  # dq/dC5 = 1
    ZS0 = np.concatenate([Z0, S0.flatten()])

    def combined_ODE(t, ZS):
        Z = ZS[:10]
        S = ZS[10:].reshape((10, 5))
        dZ = system_ODE(t, Z, C)
        dS = sensitivities_ODE(t, S, Z, C)
        return np.concatenate([dZ, dS])

    sol = solve_ivp(combined_ODE, [0, T], ZS0, t_eval=[T], method='RK45')
    ZT = sol.y[:10, -1]
    ST = sol.y[10:, -1].reshape((10, 5))
    pT = ZT[5:10]

    # Residual vector for Newton-Raphson
    residual = np.array([pT[0], pT[1], pT[2], pT[3], pT[4] + E / 2])
    J = ST[5:10, :]  # Only the costate sensitivities at final time

    return residual, J

# Newton-Raphson Loop
for k in range(max_iters):
    res, J = integrate_full_system(C)
    if np.linalg.norm(res) < tol:
        print(f"Converged in {k} iterations!")
        break
    dC = np.linalg.solve(J, -res)
    C += dC

q_final = C[4]
print("Final q:", q_final)

