## Poiseuille Flow
##                - Created by Nishit P
#                                No Slip
##                    ############################
##                    #                          #
##       Periodic     #                          # Periodic
##                    #                          #
##                    #                          #
##                    ############################
#                                No Slip

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shape as bluffbody
#Domain info and discretization
Lx = 100    #Length of channel
W = 20      #Channel Width
delx = 1
dely = 1
gridx = int(Lx/delx + 1)
gridy = int(W/dely + 2)  #Considering a solid node layer outside the fluid domain at dely/2 distance from boundary

#Flow & Solver Conditions
delT = 1
endT = 5000
tau  = 0.8
cs   = np.sqrt(1/3)*(delx/delT)
k_viscosity = np.square(cs)*delT*(tau/delT-0.5)
print("Kinematic viscosity is",k_viscosity)
rho0 = 1
Cx   = [0, 1, 0, -1, 0, 1, -1, -1, 1]
Cxnp = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1))
Cy   = [0, 0, 1, 0, -1, 1, 1, -1, -1]
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)
indxR= np.array((0,1,4,3,2,7 ,8,5,6))
dpdx = 1e-5

# Distribution functions
Feq = np.zeros((gridy,gridx,D))
F   = np.zeros((gridy,gridx,D))
Ft  = np.zeros((gridy,gridx,D))

for i,w in zip(indx,wt):
    F[:,:,i] = w*rho0

# Main iteration loop
for x in tqdm(range(1,endT+1,int(delT))):
    # print("Running time step",x)
    # Calculating field variables
    rho = np.sum(F[1:gridy-1,:], axis = 2)
    Ux = np.sum(F[1:gridy-1,:] * Cx, axis = 2) / rho + dpdx/2
    Uy = np.sum(F[1:gridy-1,:] * Cy, axis = 2) / rho

    # Equilibrium Distribution functions
    for i,cx,cy,w in zip(indx,Cx,Cy,wt):
        Feq[1:gridy-1,:,i] = w*rho*(1 + (Ux*cx+Uy*cy)/(cs**2) + (Ux*cx+Uy*cy)**2/(2*cs**4) - (Ux**2+Uy**2/(2*cs**2)))
    # Collision
    for i, cx, cy, w in zip(indx, Cx, Cy, wt):
        Ft[1:gridy-1,:,i] = F[1:gridy-1,:,i] + delT*(-1*(F[1:gridy-1,:,i]-Feq[1:gridy-1,:,i])/tau + dpdx*w*(1-delT/(2*tau))*((cx-Ux)/(cs**2)+((cx*Ux)*cx+(cy*Uy)*cx)/(cs**4)))  #With source term

    # Streaming
    for x in range(0,gridx,1):
        for y in range(1,gridy-1,1):
            for i,cx,cy in zip(indx,Cx,Cy):
                iy = y + cy
                ix = x + cx
                if ix < 0:
                    ix = gridx-1
                if ix > gridx-1:
                    ix = 0
                F[iy,ix,i] = Ft[y,x,i]
    # Bounceback boundary condition
    for x in range(0,gridx,1):
        for y in range(0,gridy,1):
            for i,cx,cy in zip(indx, Cx, Cy):
                iy = y - cy
                ix = x - cx
                if ix < 0:
                        ix = gridx - 1
                if ix > gridx - 1:
                        ix = 0
                if y == 0 and (i == 7 or i == 4 or i == 8):
                    F[iy, ix, indxR[i]] = F[y, x, i]
                elif y == gridy - 1 and (i == 6 or i == 2 or i == 5):
                    F[iy, ix, indxR[i]] = F[y, x, i]



# Analytical Solution
dely =1
D = W
ypoints = np.linspace(dely/2,D-0.5, num=D)
anal = 1/(2*k_viscosity)*(dpdx)*((D/2)**2-(ypoints-D/2)**2)
plt.plot(anal,ypoints,color='r',label='analytical')
plt.plot(Ux[:,100],ypoints,color='b',label='LBM')
plt.xlabel("Uy")
plt.ylabel("Y")
plt.title("Poiseuille flow")
plt.legend()
plt.show()
