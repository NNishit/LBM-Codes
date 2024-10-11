## Lid driven Cavity flow!   -version1
## Created by - Nishit P
                    # Ux = Uw (Wall velocity), Uy =0
##                     #########################
##                     #                       #
##                     #                       #
##            Ux,Uy=0  #                       #  Ux,Uy =0
##                     #                       #
##                     #                       #
##                     #                       #
##                     #########################
##                              Ux,Uy = 0
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Grid discretization
H = 50
delxy = 1
gridxy = int(H/delxy + 2)           # Fluid + 1 layer of Solid node out of the boundary throughout the domain

# Flow & Solver conditions
Uw   = 0.05                            # Top wall velocity
delT = 1                            # Dont put delT less than 1
endT = 2000
tau  = 0.525
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

#Identify fluid domain
for x in np.arange(0,gridxy,delxy):
    for y in np.arange(0,gridxy,delxy):
        if ((x>0 and x<gridxy-1) and (y>0 and y<gridxy-1)):
            ID[y,x] = 1


# Main Iteration Loop

for t in tqdm(range(1,endT+1,int(delT))):

    # Calculating field variables
    rho = np.sum( F[ 1 : gridxy - 1, 1 : gridxy - 1], axis=2 )
    Ux = np.sum( F[ 1 : gridxy - 1, 1 : gridxy - 1] * Cx, axis=2 ) / rho
    Uy = np.sum( F[ 1 : gridxy - 1, 1 : gridxy - 1] * Cy, axis=2 ) / rho
    Umag = (Ux)**2 + (Uy)**2

    # Equilibrium Distribution functions
    for i, cx, cy, w in zip(indx, Cx, Cy, wt):
        Feq[1 : gridxy - 1, 1 : gridxy - 1, i] = w * rho * (1 + (Ux * cx + Uy * cy) / (cs ** 2) + (Ux * cx + Uy * cy) ** 2 / (2 * cs ** 4) - (Ux ** 2 + Uy ** 2 / (2 * cs ** 2)))

    # Collision
    for i, cx, cy, w in zip(indx, Cx, Cy, wt):
        Ft[ 1 : gridxy - 1, 1 : gridxy - 1, i] = F[ 1 : gridxy - 1, 1 : gridxy - 1, i] + delT * (-1 * (F[ 1 : gridxy - 1, 1 : gridxy - 1, i] - Feq[ 1 : gridxy - 1, 1 : gridxy - 1, i]) / tau)

    # Streaming
    for x in np.arange(1,gridxy-1,1):
        for y in np.arange(1,gridxy-1,1):
            for i,cx,cy in zip(indx,Cx,Cy):
                iy = y + cy
                ix = x + cx
                F[iy, ix, i] = Ft[y, x, i]
    #BC
    for x in range(0, gridxy, delxy):
        for y in range(0, gridxy, delxy):
            for i, cx, cy, w in zip(indx, Cx, Cy, wt):
                iy = y - cy
                ix = x - cx
                if ix < 0: ix = 0
                if ix > gridxy - 1: ix = gridxy - 1
                if iy < 0: iy = 0
                if iy > gridxy - 1: iy = gridxy - 1

                if y == gridxy - 1:
                    ##Moving wall
                    if ID[iy, ix] == 1:
                        F[iy, ix, indxR[i]] = F[y, x, i] - 2 * w * rho0 * (Uw * cx) / (cs ** 2)
                elif (y == 0 or x == 0 or x == gridxy - 1):
                    ##Bounceback
                    if ID[iy, ix] == 1:
                        F[iy, ix, indxR[i]] = F[y, x, i]




##Plotting Contour
    if t>300 and t%3==0:
        x = np.linspace(delxy / 2, H - 0.5, num=H)
        y = np.linspace(delxy / 2, H - 0.5, num=H)
        Z = Umag
        plt.subplot(111)
        plt.streamplot(x, y, Ux,Uy, color = 'blue', density = 1.5)
        plt.title('Velocity Streamlines')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

# x = np.linspace(delxy / 2, H - 0.5, num=H)
# y = np.linspace(delxy / 2, H - 0.5, num=H)
# Z = Umag
# plt.subplot(111)
# plt.streamplot(x, y, Ux,Uy, color = 'blue', density = 2)
# plt.title('Velocity Streamlines')
# plt.xlabel('X')
# plt.ylabel('Y')`
# plt.show()




# #Plotting Solution
#
# ypoints = np.linspace(delxy/2,H-0.5, num=H)
# plt.plot(Ux[:,25],ypoints,color='b',label='LBM')
# plt.xlabel("Ux")
# plt.ylabel("Y")
# plt.title("Lid driven cavity flow")
# plt.legend()
# plt.show()


