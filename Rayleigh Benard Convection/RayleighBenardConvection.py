# Rayleigh Benard Convection
#                           - By Nishit P


#Domain Description:
#                         Wall with cold plate
##                     #########################
##                     #                       #
##                     #                       #
##            Periodic #                       #   Periodic
##                     #                       #
##                     #                       #
##                     #                       #
##                     #########################
##                              Hot plate
##                      Gravity acting downwards


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Domain Discretization
Lx = 100
Ly = 50
dely = 1
delx = 1
gridx = int(Lx/delx) + 1
gridy = int(Ly/dely)

# Flow & Solver Conditions
T2 = 0
T1 = 1
To = (T1 + T2)/2
delT = 1
endT = 80000
tauf  = 0.8
taug = 0.8
cs   = np.sqrt(1/3)*(delx/delT)
k_viscosity = np.square(cs)*delT*(tauf/delT-0.5)
thermal_diff = np.square(cs)*delT*(taug/delT-0.5)
g = 0.01
alpha  = 1
rho0 = 1
Rayleigh = (g*alpha*(T1-T2)*(Ly**3))/(k_viscosity*thermal_diff)
print("Kinematic viscosity is",k_viscosity)
print("Thermal diffusivity is",thermal_diff)
print("Rayleigh number is",Rayleigh)

#LBM constants
Cx   = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1))
Cy   = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1))
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)
indxR= np.array([0,3,4,1,2,7,8,5,6])

# Fluid Distribution functions
Feq = np.ones((gridy,gridx,D))
F   = np.ones((gridy,gridx,D))
Ft  = np.ones((gridy,gridx,D))

# Thermal Distribution functions
Geq = np.ones((gridy,gridx,D))
G   = np.ones((gridy,gridx,D))
Gt  = np.ones((gridy,gridx,D))

#F and G Initialization
for i,w in zip(indx,wt):
    F[:,:,i] = w
for i,w in zip(indx,wt):
    G[:, :, i] = w * T2
    G[0, :, i] = w * T1
    G[1,gridx//2,i] = w * (T1 + (T1/10))

# Field Initialization
rho = np.ones((gridy,gridx))
Ux = np.zeros((gridy,gridx))
Uy = np.zeros((gridy,gridx))
T = np.zeros((gridy,gridx))
Fb = np.zeros((gridy,gridx))

# Main iteration loop
for t in tqdm(np.arange(1,endT+1,delT)):

    # Calculating field variables
    T = np.sum(G[:, :], axis=2)
    rho = np.sum(F[:, :], axis=2)
    Ux = np.sum(F[:, :] * Cx, axis=2) / rho
    Uy = np.sum(F[:, :] * Cy, axis=2) / rho

    #Thermal Collision
    for i, cx, cy, w in zip(indx, Cx, Cy, wt):
        Geq[:, :, i] = w * T * (1 + 3*(cx * Ux + cy * Uy))
    Gt = G - (1 / taug) * (G - Geq)

    #Fluid Collision
    for i,cx,cy,w in zip(indx,Cx,Cy,wt):
        Feq[:, :, i] = w*rho*(1 + (Ux*cx+Uy*cy)/(cs**2) + (Ux*cx+Uy*cy)**2/(2*cs**4) - (Ux**2+Uy**2/(2*cs**2)))

    for i, cx, cy, w in zip(indx, Cx, Cy, wt):
        Ft[:, :, i] = F[:, :, i] + delT * (-1 * (F[:, :, i] - Feq[:, :, i]) / tauf) + 3*w*g*alpha*(T-To)*cy

    #BounceBack
    Ft[0, :, indx] = F[0, :, indxR]
    Ft[-1, :, indx] = F[-1, :, indxR]

    #Thermal + Fluid Stream in X and Y Direction
    for i, cx, cy in zip(indx, Cx, Cy):
        F[:, :, i] = np.roll(Ft[:, :, i], [cx, cy], axis=[1, 0])
        G[:, :, i] = np.roll(Gt[:, :, i], [cx, cy], axis=[1, 0])

    #Temperature Dirichlet

    G[0, 1:-1, 2] = -1 * Gt[0, 1:-1, 4] + 2 * (1/9) * T1
    G[-1, 1:-1, 4] = -1 * Gt[-1, 1:-1, 2] + 2 * (1/9) * T2


    # if (t%3==0 and t>0):
    #     plt.imshow(np.sqrt(Ux[1:-1,:]**2+Uy[1:-1,:]**2), origin ='lower')
    #     plt.pause(0.016)
    #     plt.cla()

    if (t%100==0 and t>0):
        plt.imshow(T[1:-1,:], origin ='lower')
        plt.pause(0.016)
        plt.cla()

    # #Plotting Contour
    # if t > 0 and t % 3 == 0:
    #     x = np.linspace(0, Lx, num=gridx)
    #     y = np.linspace(dely / 2, Ly - 0.5, num=gridy)
    #     plt.subplot(111)
    #     plt.contour(x, y, Ux, Uy)
    #     plt.title('Velocity Contour')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.draw()
    #     plt.pause(0.0001)
    #     plt.clf()
