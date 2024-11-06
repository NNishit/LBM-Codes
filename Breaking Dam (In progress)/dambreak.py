# Breaking Dam Simulation

import numpy as np
from LBM.freesurface.lbm import init
from LBM.freesurface.lbm import time_evolve

# Import LBM Libraries

# Domain Initialization
Lx = 170
Ly = 170
dely = 1
delx = 1
gridx = int(Lx/delx)
gridy = int(Ly/dely)
wx = 70    # initial Reservoir/Fluid filled node coordinates
wy = 130   # Fluid fill Height

# Flow & Solver Conditions
g = 0.1      # acceleration due to gravity (in -y direction)
delT = 1     # time step
endT = 700   # simulation end time
tau = 1      # relaxation parameter (SRT)
cs   = np.sqrt(1/3)*(delx/delT)
k_viscosity = np.square(cs)*delT*(tau/delT-0.5)
Re = wy*np.sqrt(g*wy)/k_viscosity
print("Kinematic viscosity is",k_viscosity)
print("Reynolds number is",Re)

# LBM constants (D2Q9 Lattice)
Cx   = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1))
Cy   = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1))
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)
indxR= np.array([0,3,4,1,2,7,8,5,6])


# Initialize Variables
Feq,F,Ft,rho,Ux,Uy,mass,eta,flag = init.init(gridx, gridy, D)

# Initialize DFs
for i,w in zip(indx,wt):
    F[:,:,i] = w

# Setting up the scene with filled, interface and empty nodes
for x in np.arange(0,gridx):
    for y in np.arange(0,gridy):
        if (x <= wx and y <= wy):
            mass[y,x] = rho[y,x]
            flag[y,x] = 1           # fluid nodes
            if (x == wx or y == wy):
                mass[y,x] = 0.5*rho[y,x]
                flag[y,x] = 0       # interface nodes
        else:
            mass[y,x] = 0
            flag[y,x] = -1          # empty nodes

eta = mass/rho

flagT = time_evolve.evolve(Feq, F, Ft, rho, Ux, Uy, g, tau, mass, eta, flag, endT)   # Start the simulation
