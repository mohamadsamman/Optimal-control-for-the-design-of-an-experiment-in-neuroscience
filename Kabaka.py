import numpy as np
import matplotlib.pyplot as plt

# Constants and Parameters (Bolie)
a = 0.78
b = 0.208
c = 4.34
d = 2.92
T = 1.0  # hours
E = 0.0489402
s=0.01
s2=s**2
q = 0.015  # Initial guess for Lagrange multiplier q
dt = 1./50.  # time step in hours
tt_array = np.arange(0, T + dt, dt)
tt=[]

for i in range(len(tt_array)):
    tt.append(tt_array[i])



Z0 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0.1,0, 0, 0, 0, 0],  # Replace x2_0 with your initial value for x2(0)
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [q, 0, 0, 0, 0, 1],    # Replace C5 with your value
    [0, 1, 0, 0, 0, 0],    # Replace C1 with your value
    [0, 0, 1, 0, 0, 0],    # Replace C2 with your value
    [0, 0, 0, 1, 0, 0],    # Replace C3 with your value
    [0, 0, 0, 0, 1, 0],    # Replace C4 with your value
    [0, 0, 0, 0, 0, 0]
    ])
Z=np.copy(Z0)
A = [[-a,b,0.,0.,0.,0.,0.,0.,0.,0.],
     [-c,-d,0.,0.,0.,0.,-1./q,0.,0.,0.],
     [0,1,-a,b,0.,0.,0.,0.,0.,0.],
     [0.,0.,-c,-d,0.,0.,0.,0.,0.,0.],
     [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
     [0.,0.,0.,0.,0.,a,c,0.,0.,0.],
     [0.,0.,0.,0.,0.,-b,d,-1.,0.,0.],
     [0.,0.,0.,0.,0.,0.,0.,a,c,0.],
     [0.,0.,0.,-1./s2,0.,0.,0.,-b,d,0.],
     [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]

#1./q**2 * Z[6,0]
Phi = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0],   
    [0, 0, 0, 0, 0, 0],    
    [0, 0, 0, 0, 0, 0],   
    [0, 0, 0, 0, 0, 0],    
    [-0.5*q*q*Z[6,0]*Z[6,0], -1./q**2* -b *Z[6,0], -1./q**2* d *Z[6,0], -1./q**2* (-1) *Z[6,0], 0, 0]
    ])

M = [[1.,0.,0.,0.,0.],
     [0.,1.,0.,0.,0.],
     [0.,0.,1.,0.,0.],
     [0.,0.,0.,1.,0.],
     [0.,0.,0.,0.,0.],]

# Initial guesses for the parameters
C_init = np.array([0.0, 0.0, 0.0, 0.0, q])  # Initial guesses for [p1(0), ..., p4(0), q(0)]

###############

Zhist=[]
Zhist.append(Z0)

C=np.copy(C_init)

#boucle sur 
Z[4:8,0] = C 
for t in tt: 
    # Compute derivative
    dZdt = A @ Z + Phi     
    # Euler step
    Z = Z + dt * dZdt   
    # Store result
    Zhist.append(Z.copy())
C = 


qevol=[]
for i in range(len(Zhist)-1):
    qevol.append(Zhist[i][4,0])
plt.plot(tt,qevol)
plt.show()

p2evol=[]
for i in range(len(Zhist)-1):
    p2evol.append(Zhist[i][6,0])
plt.plot(tt,p2evol)
plt.show()