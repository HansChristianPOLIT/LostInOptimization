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
    
    for t in range(int(par.simT)):
        for b in range(len(par.Betas)):
            for i in prange(par.simN): # in parallel
   
                # a. beginning of period states
                if t == 0:
                    p[t,b,i] = 1
                    m[t,b,i] = 1
                elif t<=par.Tr:
                    p[t,b,i] = sim.psi[t,i]*p[t-1,b,i]*par.L[t-1]
                    m[t,b,i] = par.R*a[t-1,b,i] + (1-par.tax_rate_vec[t])*sim.xi[t,i]*p[t,b,i]
                else:
                    p[t,b,i] = p[t-1,b,i]*par.L[t-1]
                    m[t,b,i] = par.R*a[t-1,b,i] + (1-par.tax_rate_vec[t])*p[t,b,i]

                # b. choices
                c[t,b,i] = linear_interp.interp_2d(par.grid_p,par.grid_m,sol.c[t,b],p[t,b,i],m[t,b,i])
                a[t,b,i] = m[t,b,i]-c[t,b,i]

@njit(parallel=True)
def lifecycle_rand(sim,sol,par):
    """ simulate full life-cycle """

    # unpack (to help numba optimize)
    p = sim.p_rand
    m = sim.m_rand
    c = sim.c_rand
    a = sim.a_rand
    
    # use pre-assigned beta indices for each individual
    beta_indices = sim.beta_rand

    for t in range(int(par.simT)):
        for i in prange(par.simN): # in parallel
            b = beta_indices[i]  # individual i's beta index
            
            # a. beginning of period states
            if t == 0:
                p[t,i] = 1
                m[t,i] = 1
            elif t<=par.Tr:
                p[t,i] = sim.psi[t,i]*p[t-1,i]*par.L[t-1] # t-1 eller t for L? 
                m[t,i] = par.R*a[t-1,i] + (1-par.tax_rate_vec[t])*sim.xi[t,i]*p[t,i]
            else:
                p[t,i] = p[t-1,i]*par.L[t-1]
                m[t,i] = par.R*a[t-1,i] + (1-par.tax_rate_vec[t])*p[t,i]

            # b. choices
            c[t,i] = linear_interp.interp_2d(par.grid_p,par.grid_m,sol.c[t,b],p[t,i],m[t,i])
            a[t,i] = m[t,i]-c[t,i]


