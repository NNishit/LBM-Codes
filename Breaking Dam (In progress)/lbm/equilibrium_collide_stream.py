import numpy as np

# Assumption!! Cs = root(1/3)
# LBM constants (D2Q9 Lattice)
Cx   = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1))
Cy   = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1))
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)
indxR= np.array([0,3,4,1,2,7,8,5,6])

def calc_feq(rho, Ux, Uy, cx, cy, wt):
    Feq = wt*rho*(1 + 3*(Ux*cx+Uy*cy) + 9*(Ux*cx+Uy*cy)**2/2 - 3*(Ux**2+Uy**2)/2)
    return Feq

def calc_eta(mass,rho,flagT):
    if flagT == -1:
        return 0
    elif flagT == 1:
        return 1
    else:
        if rho>0:
            eta = mass/rho
            if eta > 1:
                eta = 1
            elif eta < 0:
                eta = 0
            return eta

        return 0.5


def collide(Feq, F, Ft, rho, Ux, Uy, g, tau, mass, eta, flagT, t):
    for x in np.arange(np.shape(F)[1]):
        for y in np.arange(np.shape(F)[0]):
            if (flagT[t,y,x] != -1):
                for i, w, cx, cy in zip(indx, wt, Cx, Cy):
                    Ft[y, x, i] = F[y, x, i] - 1 * (F[y, x, i] - calc_feq(rho[y, x], Ux[y, x], Uy[y, x], cx, cy, w)) / tau + (rho[y,x] * cy * g * w)
    return Ft

def stream(Feq, F, Ft, rho, Ux, Uy, g, tau, mass, eta, flagT, t):
    for x in np.arange(np.shape(F)[1]):
        for y in np.arange(np.shape(F)[0]):
            # Omitting gas/empty cells from the loop
            if flagT[t, y, x] == -1:
                continue
            # Fluid
            elif flagT[t, y, x] == 1:
                for i, cx, cy in zip(indx, Cx, Cy):
                    xneb = x + cx
                    yneb = y + cy                                                #Neighbour cells

                    if xneb < 0 or xneb >= np.shape(F)[1] or yneb < 0 or yneb >= np.shape(F)[0]:
                        continue

                    if flagT[t, yneb, xneb] == 1:                                #Fluid-Fluid
                        F[yneb, xneb, i] = Ft[y, x, i]
                    elif flagT[t, yneb, xneb] == 0:                              #Fluid-interface
                        F[yneb, xneb, i] = Ft[y, x, i]
                        mass[y, x] += Ft[yneb, xneb, indxR[i]] - Ft[y, x, i]
                    else:
                        F[y, x, indx] = Ft[y, x, indxR]                          #obstacle

            # Interface
            elif flagT[t, y, x] == 0:
                eta = calc_eta(mass[y,x],rho[y,x],flagT[t,y,x])
                for i, cx, cy, w in zip(indx, Cx, Cy, wt):
                    xneb = x + cx
                    yneb = y + cy                                                #Neighbour cells

                    if xneb < 0 or xneb >= np.shape(F)[1] or yneb < 0 or yneb >= np.shape(F)[0]:
                        continue

                    if flagT[t, yneb, xneb] == -1:                               #Interface-Solid (Reconstruct DFs)
                        F[y,x,indxR[i]] = calc_feq(1, Ux[y, x], Uy[y, x], cx, cy, w) + calc_feq(1, Ux[y, x], Uy[y, x], Cx[indx[i]], Cy[indx[i]], wt[indx[i]]) - Ft[y,x,i]
                    elif flagT[t, yneb, xneb] == 1:                              #Interface-Fluid
                        F[yneb, xneb, i] = Ft[y, x, i]
                        mass[y, x] += Ft[yneb, xneb, indxR[i]] - Ft[y, x, i]
                    elif flagT[t, yneb, xneb] == 0:                              #Interface-Interface
                        F[yneb, xneb, i] = Ft[y, x, i]
                        eta_neb = calc_eta(mass[yneb, xneb], rho[yneb, xneb], flagT[t, yneb, xneb])
                        mass[y, x] += ((eta + eta_neb)/2)*(mass_exchange(Ft, y, x, yneb, xneb, i, flagT, t))
                    else:
                        F[y, x, indx] = Ft[y, x, indxR]                          # obstacle

                for i, cx, cy, w in zip(indx, Cx, Cy, wt):
                    if 0.5 * surface_normals(cx, cy, x, y, t, mass, rho, flagT) >= 0:
                        F[y, x, indxR[i]] = calc_feq(1, Ux[y, x], Uy[y, x], cx, cy, w) + calc_feq(1, Ux[y, x], Uy[y, x], Cx[indx[i]], Cy[indx[i]], wt[indx[i]]) - Ft[y, x, i]

    rho, Ux, Uy = calc_rho_vel(F)

    return F, rho, Ux, Uy, mass


def surface_normals(cx, cy, x, y, t, mass, rho, flagT):
    surface_normal = (calc_eta(mass[y, x-1], rho[y, x-1], flagT[t, y, x-1]) - calc_eta(mass[y, x+1], rho[y, x+1], flagT[t, y, x+1]))*cx + (calc_eta(mass[y-1, x], rho[y-1, x], flagT[t, y-1, x]) - calc_eta(mass[y+1, x], rho[y+1, x], flagT[t, y+1, x]))*cy
    return surface_normal

def neighbour(y,x,t,flagT):
    flag1 = []
    for i,cx,cy in zip(indx,Cx,Cy):
        x1 = x + cx
        y1 = y + cy
        if x1 >= 0 or x1 < np.shape(flagT)[2] or y1 >= 0 or y1 < np.shape(flagT)[1]:
            continue
        flag1 = np.append(flag1, flagT[t, y1, x1])

    return flag1

def mass_exchange(Ft,y,x,yneb,xneb,i,flagT,t):
    flag1 = neighbour(y, x, t, flagT)
    flag2 = neighbour(yneb, xneb, t, flagT)

    if 1 not in flag1:
        if 1 not in flag2:
            return Ft[yneb, xneb, indxR[i]] - Ft[y, x, i]
        else:
            return -Ft[y,x,i]
    elif -1 not in flag1:
        if -1 not in flag2:
            return Ft[yneb, xneb, indxR[i]] - Ft[y, x, i]
        else:
            return Ft[yneb, xneb, indxR[i]]
    else:
        if 1 not in flag2:
            return Ft[yneb, xneb, indxR[i]]
        elif -1 not in flag2:
            return -Ft[y,x,i]
        else:
            return Ft[yneb, xneb, indxR[i]] - Ft[y, x, i]

def calc_rho_vel(F):

    rho = np.sum(F[:, :], axis=2)
    Ux = np.sum(F[:, :] * Cx, axis=2) / rho
    Uy = np.sum(F[:, :] * Cy, axis=2) / rho
    return rho,Ux,Uy





