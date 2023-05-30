import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import trans
import utility

@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ simulate full life-cycle """

    # unpack
    p = sim.p
    n = sim.n
    m = sim.m
    c = sim.c
    d = sim.d
    a = sim.a
    discrete = sim.discrete
    
    for t in range(par.T):
        for b in range(len(par.Betas)):
            for i in prange(par.simN):

                # a. beginning of period states
                if t == 0:
                    p[t,b,i] = trans.p_plus_func(sim.p0[b,i],sim.psi[t,i],par,t-1)
                    n[t,b,i] = trans.n_plus_func(sim.d0[b,i],par)   
                    m[t,b,i] = trans.m_plus_func(sim.a0[b,i],p[t,b,i],sim.xi[t,i],par,t)
                else:
                    p[t,b,i] = trans.p_plus_func(p[t-1,b,i],sim.psi[t,i],par,t-1)
                    n[t,b,i] = trans.n_plus_func(d[t-1,b,i],par)
                    m[t,b,i] = trans.m_plus_func(a[t-1,b,i],p[t,b,i],sim.xi[t,i],par,t)

                # b. optimal choices and post decision states
                optimal_choice(t,b,p[t,b,i],n[t,b,i],m[t,b,i],discrete[t,b,i:],d[t,b,i:],c[t,b,i:],a[t,b,i:],sol,par)

            
@njit            
def optimal_choice(t,b,p,n,m,discrete,d,c,a,sol,par):

    x = trans.x_plus_func(m,n,par)

    # a. discrete choice
    inv_v_keep = linear_interp.interp_3d(par.grid_p,par.grid_n,par.grid_m,sol.inv_v_keep[t,b],p,n,m)
    inv_v_adj = linear_interp.interp_2d(par.grid_p,par.grid_x,sol.inv_v_adj[t,b],p,x)    
    adjust = inv_v_adj > inv_v_keep
    
    # b. continuous choices
    if adjust:

        discrete[0] = 1
        
        d[0] = linear_interp.interp_2d(
            par.grid_p,par.grid_x,sol.d_adj[t,b],
            p,x)

        c[0] = linear_interp.interp_2d(
            par.grid_p,par.grid_x,sol.c_adj[t,b],
            p,x)

        tot = d[0]+c[0]
        if tot > x: 
            d[0] *= x/tot
            c[0] *= x/tot
            a[0] = 0.0
        else:
            a[0] = x - tot
            
    else: 
            
        discrete[0] = 0

        d[0] = n

        c[0] = linear_interp.interp_3d(
            par.grid_p,par.grid_n,par.grid_m,sol.c_keep[t,b],
            p,n,m)

        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]        

@njit            
def euler_errors(sim,sol,par):
    
    # unpack
    euler_error = sim.euler_error
    euler_error_c = sim.euler_error_c
    
    for b in range(len(par.Betas)):  # loop over different beta values
        beta = par.Betas[b]
        
        for i in prange(par.simN):

            discrete_plus = np.zeros(1)
            d_plus = np.zeros(1)
            c_plus = np.zeros(1)
            a_plus = np.zeros(1)

            for t in range(par.T-1):

                    constrained = sim.a[t,b,i] < par.euler_cutoff

                    if constrained:

                        euler_error[t,b,i] = np.nan
                        euler_error_c[t,b,i] = np.nan
                        continue

                    else:

                        RHS = 0.0
                        for ishock in range(par.Nshocks):

                            # i. shocks
                            psi = par.psi[ishock]
                            psi_w = par.psi_w[ishock]
                            xi = par.xi[ishock]
                            xi_w = par.xi_w[ishock]

                            # ii. next-period states
                            p_plus = trans.p_plus_func(sim.p[t,b,i],psi,par,t) # remember to double check "t-1"
                            n_plus = trans.n_plus_func(sim.d[t,b,i],par) 
                            m_plus = trans.m_plus_func(sim.a[t,b,i],p_plus,xi,par,t)

                            # iii. weight
                            weight = psi_w*xi_w

                            # iv. next-period choices
                            optimal_choice(t+1,b,p_plus,n_plus,m_plus,discrete_plus,d_plus,c_plus,a_plus,sol,par)


                            # v. next-period marginal utility
                            RHS += weight*beta*par.R*utility.marg_func(c_plus[0],d_plus[0],par)

                        euler_error[t,b,i] = sim.c[t,b,i] - utility.inv_marg_func(RHS,sim.d[t,b,i],par)
                        euler_error_c[t,b,i] = sim.c[t,b,i]

@njit(parallel=True)
def calc_utility(sim,sol,par):
    """ calculate utility for each individual """

    # unpack
    u = sim.utility
    
    for t in range(par.T):
        for b in range(len(par.Betas)):
            beta = par.Betas[b]
            
            for i in prange(par.simN):

                u[i] += beta**t*utility.func(sim.c[t,b,i],sim.d[t,b,i],par)


