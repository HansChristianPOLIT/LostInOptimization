import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ simulate full life-cycle """

    # unpack (to help numba optimize)
    p = sim.p
    m = sim.m
    c = sim.c
    a = sim.a
    
    for t in range(par.simT):
        for i in prange(par.simN): # in parallel
            
            # a. beginning of period states
            if t == 0:
                p[t,i] = 1
                m[t,i] = 1
            elif t<=par.Tr:
                p[t,i] = sim.psi[t,i]*p[t-1,i]*par.L[t-1] # t-1 eller t for L? 
                m[t,i] = par.R*a[t-1,i] + (1-par.tax_rate)*sim.xi[t,i]*p[t,i]
            else:
                p[t,i] = p[t-1,i]*par.L[t-1]
                m[t,i] = par.R*a[t-1,i] + (1-par.tax_rate)*p[t,i] 

            # b. choices
            c[t,i] = linear_interp.interp_2d(par.grid_p,par.grid_m,sol.c[t],p[t,i],m[t,i])
            a[t,i] = m[t,i]-c[t,i]