# Flow past bluff bodies
#                       -Created by Nishit P
#                                Slip
##                    ############################
##                    #                          #
##            Inlet   #     ******               # Outlet
##                    #     ******               #
##                    #     ******               #
##                    #     Bluff body           #
##                    ############################
#                                Slip

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Domain Discretization
Lx = 200
Ly = 75
charL = 20             # Characteristic length in Reynolds number (Diameter for cylinder, Width for Square)
locX  = Lx/4
locY  = Ly/2
delx = 1
dely = 1
gridx = int(Lx/delx) + 1
gridy = int(Ly/dely) + 1

# Flow & Solver Conditions
Uin  = 0.02
delT = 1
endT = 10000
tau  = 0.51
cs   = np.sqrt(1/3)*(delx/delT)
k_viscosity = np.square(cs)*delT*(tau/delT-0.5)
print("Kinematic viscosity is",k_viscosity)
print("Reynolds number",(Uin*charL)/k_viscosity)
rho0 = 1
rhoout = 1
Cx   = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1))
Cy   = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1))
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)
indxR= np.array((0,1,4,3,2,7,8,5,6))

# Distribution functions
Feq = np.ones((gridy,gridx,D))
F   = np.ones((gridy,gridx,D))
Ft  = np.ones((gridy,gridx,D))
cylinder = np.full((gridy,gridx),False)
# arrays storing values of boundary fluid nodes near bluff bodies
rho = np.ones((gridy,gridx))
Ux = np.zeros((gridy,gridx))
Uy = np.zeros((gridy,gridx))
Ux[:,:] = Uin
for i,cx,cy,w in zip(indx,Cx,Cy,wt):
    F[:,:,i] = w * rho * (1 + (Ux * cx + Uy * cy) / (cs ** 2) + (Ux * cx + Uy * cy) ** 2 / (2 * cs ** 4) - (Ux ** 2 + Uy ** 2 / (2 * cs ** 2)))

#Identify fluid nodes which will hit bluff body
for x in np.arange(0,gridx,1):
    for y in np.arange(0,gridy,1):
        if ((x - locX)**2 + (y - locY)**2 <= (charL/2)**2):
                cylinder[y][x] = True

# Main iteration loop
for t in tqdm(np.arange(1,endT+1,delT)):

    # Calculating field variables
    rho = np.sum(F[:,:], axis = 2)
    Ux = np.sum(F[:,:] * Cx, axis = 2) / rho
    Uy = np.sum(F[:,:] * Cy, axis = 2) / rho
    rho[1:-1, 0] = (F[1:-1, 0, 0] + F[1:-1, 0, 2] + F[1:-1, 0, 4] + 2 * (F[1:-1, 0, 3] + F[1:-1, 0, 6] + F[1:-1, 0, 7])) / (1 - Uin)
    Ux[1:-1, 0] = Uin
    Ux[cylinder] = 0
    Uy[cylinder] = 0


    # Equilibrium Distribution functions
    for i,cx,cy,w in zip(indx,Cx,Cy,wt):
        Feq[:, :, i] = w * rho * (1 + (Ux * cx + Uy * cy) / (cs ** 2) + (Ux * cx + Uy * cy) ** 2 / (2 * cs ** 4) - (Ux ** 2 + Uy ** 2 / (2 * cs ** 2)))

    # Collision
    Ft = F - (1 / tau) * (F - Feq)

    bndryF = Ft[cylinder, :]
    bndryF[:, indx] = bndryF[:, indxR]
    F[cylinder, :] = bndryF

    #Streaming
    for i, cx, cy in zip(indx, Cx, Cy):
        Ft[:, :, i] = np.roll(Ft[:, :, i], cx, axis=1)
        Ft[:, :, i] = np.roll(Ft[:, :, i], cy, axis=0)

    Ft[1:-1, 0, :] = Feq[1:-1, 0, :]
    F = Ft

    F[:, -1, [3, 6, 7]] = F[:, -2, [3, 6, 7]]









    if (t%100==0):
        plt.imshow(np.sqrt(Ux**2+Uy**2))
        plt.pause(0.01)
        plt.cla()
