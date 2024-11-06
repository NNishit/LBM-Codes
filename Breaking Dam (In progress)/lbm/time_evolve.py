import numpy as np
from LBM.freesurface.lbm import equilibrium_collide_stream
from LBM.freesurface.lbm import updatemass


def evolve(Feq, F, Ft, rho, Ux, Uy, g, tau, mass, eta, flag, endT):

    flagT = np.zeros((endT,*flag.shape))
    flagT[0] = flag

    for t in np.arange(0,endT):

        # collide

        Ft = equilibrium_collide_stream.collide(Feq, F, Ft, rho, Ux, Uy, g, tau, mass, eta, flagT, t)

        # stream

        F, rho, Ux, Uy, mass = equilibrium_collide_stream.stream(Feq, F, Ft, rho, Ux, Uy, g, tau, mass, eta, flagT, t)

        # update cell type

        F, mass, eta,flagT[t] = updatemass.update_mass(Feq, F, Ft, rho, Ux, Uy, g, tau, mass, eta, flagT, t)

    return flagT