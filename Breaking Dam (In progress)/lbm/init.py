#Initialize Field variables

import numpy as np
def init(gridx,gridy,D):
    # Field Initialization
    Feq  = np.zeros((gridy, gridx, D))  # Eq DFs
    F    = np.zeros((gridy, gridx, D))  # DF at time t
    Ft   = np.zeros((gridy, gridx, D))  # DF at time t+1
    rho  = np.ones((gridy, gridx))      # density
    Ux   = np.zeros((gridy, gridx))     # velocity in x direction
    Uy   = np.zeros((gridy, gridx))     # velocity in y
    mass = np.zeros((gridy, gridx))     # mass of fluid at each node (0 at empty node, rho at filled node, some value between these for interface nodes)
    eta  = np.zeros((gridy, gridx))     # mass fraction of fluid at each node (0 to 1, from empty to filled)
    flag = np.zeros((gridy, gridx))     # a flag to set 1 for filled, 0 for interface, -1 for empty
    return Feq,F,Ft,rho,Ux,Uy,mass,eta,flag
