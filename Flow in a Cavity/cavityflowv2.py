# Lid driven Cavity flow!
#                          -version2
# Implemented Mesh Refinement
# NEBB Condition
# Works only for low Reynolds numbers (Uw<Cs)
# Created by - Nishit P
#                    Ux = Uw (Wall velocity), Uy =0
#                     #########################
#                     #                       #
#                     #                       #
#            Ux,Uy=0  #                       #  Ux,Uy =0
#                     #                       #
#                     #                       #
#                     #                       #
#                     #########################
#                              Ux,Uy = 0
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Grid discretization
H = 50
delxy = 1
gridxy = int(H/delxy) + 1


# Flow & Solver conditions
Uw   = 0.1                          # Top wall velocity
delT = 1                            # Dont put delT less than 1
endT = 2000
tau  = 0.8
cs   = np.sqrt(1/3)*(delxy/delT)
k_viscosity = np.square(cs)*delT*(tau/delT-0.5)
print("Kinematic viscosity is",k_viscosity)
print("Reynolds number",(Uw*H)/k_viscosity)
rho0 = 1
Cx   = [0, 1, 0, -1, 0, 1, -1, -1, 1]
Cy   = [0, 0, 1, 0, -1, 1, 1, -1, -1]
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)
indxR= np.array((0,3,4,1,2,7,8,5,6))

# Distribution functions
Feq = np.zeros((gridxy,gridxy,D))
F   = np.zeros((gridxy,gridxy,D))
Ft  = np.zeros((gridxy,gridxy,D))
ID = np.zeros((gridxy,gridxy))

for i,w in zip(indx,wt):
    F[:,:,i] = w*rho0

for t in tqdm(range(1,endT+1,int(delT))):

    # Calculating field variables
    rho = np.sum( F[:,:], axis=2 )
    Ux = np.sum( F[:,:] * Cx, axis=2 ) / rho
    Uy = np.sum( F[:,:] * Cy, axis=2 ) / rho
    # Umag = (Ux)**2 + (Uy)**2



    # Equilibrium Distribution functions
    for i, cx, cy, w in zip(indx, Cx, Cy, wt):
        Feq[:, :, i] = w * rho * (1 + (Ux * cx + Uy * cy) / (cs ** 2) + (Ux * cx + Uy * cy) ** 2 / (2 * cs ** 4) - (Ux ** 2 + Uy ** 2 / (2 * cs ** 2)))

    # Collision
    for i, cx, cy, w in zip(indx, Cx, Cy, wt):
        Ft[:, :,i] = F[:, :, i] + delT * (-1 * (F[:, :, i] - Feq[:, :, i]) / tau)

    # Streaming
    for x in np.arange(0, gridxy, 1):
        for y in np.arange(0, gridxy, 1):
            for i,cx,cy in zip(indx,Cx,Cy):
                iy = y + cy
                ix = x + cx
                if ix < 0: ix = 0
                if ix > gridxy - 1: ix = gridxy - 1
                if iy < 0: iy = gridxy - 1
                if iy > gridxy - 1: iy = 0
                F[iy, ix, i] = Ft[y, x, i]

        # BC
    for x in np.arange(0, gridxy, 1):
        for y in np.arange(0, gridxy, 1):
            if x == 0 and y == 0:
                F[y, x, 1] = F[y, x, 3]
                F[y, x, 2] = F[y, x, 4]
                F[y, x, 5] = F[y, x, 7]
                F[y, x, 6] = 0
                F[y, x, 8] = 0
                F[y, x, 0] = rho0 - 1 * (F[y, x, 1] + F[y, x, 2] + F[y, x, 3] + F[y, x, 4] + F[y, x, 5] + F[y, x, 6] + F[y, x, 7] + F[y, x, 8])
            if x == 0 and y == gridxy - 1:
                F[y, x, 1] = F[y, x, 3]
                F[y, x, 4] = F[y, x, 2]
                F[y, x, 8] = F[y, x, 6]
                F[y, x, 5] = 0
                F[y, x, 7] = 0
                F[y, x, 0] = rho0 - 1 * (F[y, x, 1] + F[y, x, 2] + F[y, x, 3] + F[y, x, 4] + F[y, x, 5] + F[y, x, 6] + F[y, x, 7] + F[y, x, 8])
            if x == gridxy - 1 and y == 0:
                F[y, x, 3] = F[y, x, 1]
                F[y, x, 2] = F[y, x, 4]
                F[y, x, 6] = F[y, x, 8]
                F[y, x, 5] = 0
                F[y, x, 7] = 0
                F[y, x, 0] = rho0 - 1 * (F[y, x, 1] + F[y, x, 2] + F[y, x, 3] + F[y, x, 4] + F[y, x, 5] + F[y, x, 6] + F[y, x, 7] + F[y, x, 8])
            if x == gridxy - 1 and y == gridxy - 1:
                F[y, x, 3] = F[y, x, 1]
                F[y, x, 4] = F[y, x, 2]
                F[y, x, 7] = F[y, x, 5]
                F[y, x, 6] = 0
                F[y, x, 8] = 0
                F[y, x, 0] = rho0 - 1 * (F[y, x, 1] + F[y, x, 2] + F[y, x, 3] + F[y, x, 4] + F[y, x, 5] + F[y, x, 6] + F[y, x, 7] + F[y, x, 8])
            elif y == gridxy - 1:
                rhow = F[y, x, 0] + F[y, x, 3] + F[y, x, 1] + 2 * (F[y, x, 2] + F[y, x, 6] + F[y, x, 5])
                F[y, x, 4] = F[y, x, 2]
                F[y, x, 8] = F[y, x, 6] - (F[y, x, 1] - F[y, x, 3]) / 2 + (rhow * Uw) / 2
                F[y, x, 7] = F[y, x, 5] + (F[y, x, 1] - F[y, x, 3]) / 2 - (rhow * Uw) / 2
            elif y == 0:
                # rhow = F[y, x, 0] + F[y, x, 3] + F[y, x, 1] + 2 * (F[y, x, 4] + F[y, x, 7] + F[y, x, 8])
                F[y, x, 2] = F[y, x, 4]
                F[y, x, 6] = F[y, x, 8] + (F[y, x, 1] - F[y, x, 3]) / 2
                F[y, x, 5] = F[y, x, 7] - (F[y, x, 1] - F[y, x, 3]) / 2
            elif x == 0:
                # rhow = F[y, x, 0] + F[y, x, 2] + F[y, x, 4] + 2 * (F[y, x, 6] + F[y, x, 3] + F[y, x, 7])
                F[y, x, 1] = F[y, x, 3]
                F[y, x, 8] = F[y, x, 6] + (F[y, x, 2] - F[y, x, 4]) / 2
                F[y, x, 5] = F[y, x, 7] - (F[y, x, 2] - F[y, x, 4]) / 2
            elif x == gridxy - 1:
                # rhow = F[y, x, 0] + F[y, x, 2] + F[y, x, 4] + 2 * (F[y, x, 1] + F[y, x, 5] + F[y, x, 8])
                F[y, x, 3] = F[y, x, 1]
                F[y, x, 6] = F[y, x, 8] - (F[y, x, 2] - F[y, x, 4]) / 2
                F[y, x, 7] = F[y, x, 5] + (F[y, x, 2] - F[y, x, 4]) / 2





     #Plotting Contour
    if t > 300 and t % 3 == 0:
        x = np.linspace(0, gridxy + 1, num= gridxy)
        y = np.linspace(0, gridxy + 1, num= gridxy)

        plt.subplot(111)
        plt.streamplot(x, y, Ux, Uy, color='blue', density=1.5)
        plt.title('Velocity Streamlines')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

# #Plotting Solution
#
# ypoints = np.linspace(0, H, num=H)
# Uxplot = [0,Ux[:,25],0]
# plt.plot(Uxplot,ypoints,color='b',label='LBM')
# plt.xlabel("Ux")
# plt.ylabel("Y")
# plt.title("Lid driven cavity flow")
# plt.legend()
# plt.show()

