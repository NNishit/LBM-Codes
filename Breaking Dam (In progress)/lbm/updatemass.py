import numpy as np
from lbm.equilibrium_collide_stream import calc_eta, calc_feq, neighbour, surface_normals

Cx   = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1))
Cy   = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1))
wt   = np.array((4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36))
D = 9
indx = np.arange(D)


# Free Surface constants
offset = 0.001 # prevents newly formed interface cells from being re-converted
garbage = 0.1  # removing interface cell artifacts


def update_mass(F, Ft, Feq, rho, Ux, Uy, g, tau, mass, eta, flagTprev, t):
    flagTemp = np.zeros((F.shape[0],F.shape[1]))
    flagTnew = np.copy(flagTprev[t])

    # interface to filled/empty nodes
    for x in np.arange(np.shape(F)[1]):
        for y in np.arange(np.shape(F)[0]):
            if flagTprev[t,y,x] == 0:
                if (mass[y,x] > (1 + offset)*rho[y,x]) or ((mass[y,x] > (1-garbage)*rho[y,x]) and -1 not in neighbour(x,y,t,flagTprev)):
                    flagTemp[y, x] = 1      # change interface to filled / fluid

                elif (mass[y,x] < (0 - offset)*rho[y,x]) or ((mass[y,x] < (garbage)*rho[y,x]) and 1 not in neighbour(x,y,t,flagTprev)):
                    flagTemp[y, x] = -1   # change interface to empty

    # Interface filled to fluid node
    for x in np.arange(np.shape(F)[1]):
        for y in np.arange(np.shape(F)[0]):
            if flagTemp[y,x] == 1:
                for i,cx,cy in zip(indx,Cx,Cy):
                    xnew = x + cx
                    ynew = y + cy

                    if flagTprev[t, ynew, xnew] == -1:
                        flagTnew[ynew, xnew] = 0            # convert neighbouring empty cells to interface
                        F[ynew, xnew, i] = calc_feq(avg_surround(ynew,xnew,rho,Ux,Uy,flagTnew),cx,cy,wt)



    # Interface empty to empty node
    for x in np.arange(np.shape(F)[1]):
        for y in np.arange(np.shape(F)[0]):
            if flagTemp[y,x] == -1:
                for i,cx,cy in zip(indx,Cx,Cy):
                    xnew = x + cx
                    ynew = y + cy

                    if flagTprev[t, ynew, xnew] == 1:
                        flagTnew[ynew, xnew] = 0            # convert neighbouring fluid cells to interface



    # Mass Conservation
    for x in np.arange(np.shape(F)[1]):
        for y in np.arange(np.shape(F)[0]):

            if flagTemp[y,x] == 1:
                ex = mass[y,x] - rho[y,x]
                mass[y,x] = rho[y,x]

            elif flagTemp[y,x] == -1:
                ex = mass[y,x]
                mass[y,x] = 0

            else:
                continue

            fraction = np.zeros((9))
            fractionTotal = 0

            for i, cx, cy in zip(indx, Cx, Cy):
                xnew = x + cx
                ynew = y + cy
                if flagTnew[ynew, xnew] == 0:
                    if flagTemp[y, x] == 1:
                        fraction[i] = surface_normals(cx, cy, x, y, t, mass, rho, flagTnew) if surface_normals(cx, cy, x, y, t, mass, rho, flagTnew) > 0 else 0
                        fractionTotal += fraction[i]
                    elif flagTemp[y, x] == -1:
                        fraction[i] = -1*surface_normals(cx, cy, x, y, t, mass, rho, flagTnew) if surface_normals(cx, cy, x, y, t, mass, rho, flagTnew) < 0 else 0
                        fractionTotal += fraction[i]

            for i, cx, cy in zip(indx, Cx, Cy):
                xnew = x + cx
                ynew = y + cy

                if flagTnew[ynew, xnew] == 0:
                    mass[ynew, xnew] += ex*fraction[i]/fractionTotal


    for x in np.arange(np.shape(F)[1]):
        for y in np.arange(np.shape(F)[0]):
            if flagTemp[y,x] == 1:
                flagTnew[y,x] = 1
            elif flagTemp[y,x] == -1:
                flagTnew[y,x] = 1


    return F,mass,eta,flagTnew









def avg_surround(x,y,rho,Ux,Uy,flagTnew):
    c = 0
    rhosum = 0
    Uxsum = 0
    Uysum = 0
    for i,cx,cy in zip(indx,Cx,Cy):
        xnew = x + cx
        ynew = y + cy

        if flagTnew[ynew,xnew] == 0 or flagTnew[ynew,xnew] == 1:
            c += 1
            rhosum += rho[ynew,xnew]
            Uxsum += Ux[ynew, xnew]
            Uysum += Uy[ynew, xnew]

    rhosum = rhosum / c
    Uxsum = Uxsum / c
    Uysum = Uysum / c

    return rhosum,Uxsum,Uysum



