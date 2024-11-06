import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shape import *

# Domain Discretization
Lx = 300
Ly = 75
charL = 25             # Characteristic length in Reynolds number (Diameter for cylinder, Width for Square)
locX  = Lx/4
locY  = Ly/2
delx = 1
dely = 1
gridx = int(Lx/delx) + 1
gridy = int(Ly/dely) + 1

# Flow & Solver Conditions
Uin  = 0.01
delT = 1
endT = 20000
tau  = 0.519
cs   = np.sqrt(1/3)*(delx/delT)
k_viscosity = np.square(cs)*delT*(tau/delT-0.5)
print("Kinematic viscosity is",k_viscosity)
print("Reynolds number",(Uin*charL)/k_viscosity)
rho0 = 1
Cx   = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1))
Cy   = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1))
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)
indxR= np.array([0,3,4,1,2,7,8,5,6])

# Distribution functions
Feq = np.ones((gridy,gridx,D))
F   = np.ones((gridy,gridx,D))
Ft  = np.ones((gridy,gridx,D))
cylinder = np.full((gridy,gridx),False)

# arrays storing values of boundary fluid nodes near bluff bodies and initialization
rho = np.ones((gridy,gridx))
Ux = np.zeros((gridy,gridx))
Uy = np.zeros((gridy,gridx))
Ux[:,:] = Uin
for i,cx,cy,w in zip(indx,Cx,Cy,wt):
    F[:,:,i] = w * rho * (1 + (Ux * cx + Uy * cy) / (cs ** 2) + (Ux * cx + Uy * cy) ** 2 / (2 * cs ** 4) - (Ux ** 2 + Uy ** 2 / (2 * cs ** 2)))


cylinder = shape(gridy,gridx,locX,locY,charL,'circle')


# Main iteration loop
for t in tqdm(np.arange(1,endT+1,delT)):

    #Outflow condition
    F[:,-1,[3,6,7]] = F[:,-2,[3,6,7]]
    # F[:,-1,:] = 2*F[:,-2,:] - F[:,-3,:]

    # Calculating field variables
    rho = np.sum(F[:, :], axis=2)
    Ux = np.sum(F[:, :] * Cx, axis=2) / rho
    Uy = np.sum(F[:, :] * Cy, axis=2) / rho

    #Dirichlet boundary condition
    rho[:, 0] = (F[:, 0, 0] + F[:, 0, 2] + F[:, 0, 4] + 2 * (F[:, 0, 3] + F[:, 0, 6] + F[:, 0, 7])) / (1 - Uin)
    Ux[1:-1,0] = Uin
    Uy[1:-1,0] = 0
    Ux[cylinder] = 0
    Uy[cylinder] = 0

    #Eq distribution function
    for i,cx,cy,w in zip(indx,Cx,Cy,wt):
        Feq[:, :, i] = w * rho * (1 + (Ux * cx + Uy * cy) / (cs ** 2) + (Ux * cx + Uy * cy) ** 2 / (2 * cs ** 4) - (Ux ** 2 + Uy ** 2 / (2 * cs ** 2)))

    F[:,0,:] = Feq[:,0,:]

    #Collision
    Ft = F - (1/tau)*(F-Feq)

    #bounceback
    for i in range(9):
        Ft[cylinder,i] = F[cylinder,indxR[i]]

    #streaming
    # Fstream = np.copy(Ft)
    for i, cx, cy in zip(indx, Cx, Cy):
        F[:, :, i] = np.roll(Ft[:, :, i], [cx, cy], axis=[1, 0])

    # # enforce boundary condition at inlet
    F[1:-1, 0, :] = Feq[1:-1, 0, :]



    if (t%100==0 and t>5000):
        plt.imshow(np.sqrt(Ux**2+Uy**2))
        plt.pause(0.016)
        plt.cla()

