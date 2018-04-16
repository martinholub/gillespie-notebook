#!/usr/env/bin python

# Imports and Setup
import ssa_routine as ssa
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot') # use "ggplot" style for graphs
pltparams = {'legend.fontsize':14,'axes.labelsize':18,'axes.titlesize':18,
             'xtick.labelsize':12,'ytick.labelsize':12,'figure.figsize':(7.5,7.5),}
plt.rcParams.update(pltparams)

# Model Parameters
sigma = 100
V = 1e-14
to_nanomolar = 1e-9
omega = 6.022e23*V*to_nanomolar
t_end = 200
k = np.asarray([100, 1000, 10, 4, 2/omega, 1, 4/omega, 1, 4*sigma/omega, 1])
q0 = np.array([  10, 10,10,10, 0, 0, 0])
names = ['G1', 'P1', 'G2', 'P2', r'$\xi_1^{I}$', r'$\xi_2^{I_1}$', r'$\xi_2^{I_2}$']
#dist = ['poiss', 'poiss', 'poiss', 'gauss', 'gauss', 'poiss', 'gauss']
dist = ["none"]*7
## State change matrix associated with each reaction
nu = np.array([ [0, 1, 0, 0, 0, 0, 0], #G1    -> G1+P1
                [0, 0, 0, 2, 0, 0, 0], #G2    -> G2+P2
                [0,-1, 0, 0, 0, 0, 0], #P1    -> _
                [0, 0, 0,-1, 0, 0, 0], #P2    -> _
                [-1,0, 0,-1, 1, 0, 0], #G1+P2 -> C1
                [1, 0, 0, 1,-1, 0, 0], #C1    -> G1+P2
                [0,-1,-1, 0, 0, 1, 0], #P1+G2 -> C2a
                [0, 1, 1, 0, 0,-1, 0], #C2a   -> P1+G2
                [0,-1, 0, 0, 0,-1, 1], #P1+C2a-> C2b 
                [0, 1, 0, 0, 0, 1,-1]  #C2b   -> P1+C2a
              ], dtype = np.int)
## Molecularity of species entering each reaction
psi = np.array([[1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]], dtype = np.int)

mp_toggle = ssa.modelParameters(k, nu, psi, q0, t_end, names, dist)

#Simulation Parameters
n_paths = 10
do_load = 0 # default 0 
do_save = 1 # default 1 
np.random.seed(56)

if __name__ == "__main__":
    ssa.main(mp_toggle, n_paths)