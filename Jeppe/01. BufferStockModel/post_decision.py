import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility

@njit(parallel=True)
def compute_wq(t,sol,par,compute_w=False,compute_q=False):
    """ compute the post-decision functions w and/or q """

    # this is a variant of Algorithm 5 in Druedahl (2019): A Guide to Solve Non-Convex Consumption-Saving Problems 

    # unpack (helps numba optimize)
    w = sol.w
    q = sol.q

    # loop over outermost post-decision state
    for ip in range(par.Np): # in parallel

        # a. permanent income
        p = par.grid_p[ip]

        # b. allocate containers and initialize at zero
        m_plus = np.empty(par.Na)
        if compute_w:
            w[ip,:] = 0
            v_plus = np.empty(par.Na)
        if compute_q:
            q[ip,:] = 0
            c_plus = np.empty(par.Na)

        # c. loop over shocks and then end-of-period assets
        for ishock in range(par.Nshocks):
            
            # i. shocks
            psi = par.psi[ishock]
            psi_w = par.psi_w[ishock]
            xi = par.xi[ishock]
            xi_w = par.xi_w[ishock]

            # ii. next-period income
            p_plus = p*psi
            y_plus = p_plus*xi
            
            # iii. prepare interpolation in p direction
            prep = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)

            # iv. weight
            weight = psi_w*xi_w

            # v. next-period cash-on-hand and interpolate
            for ia in range(par.Na):
                m_plus[ia] = par.R*par.grid_a[ia] + (1-par.tax_rate)*y_plus
            
            # v_plus
            if compute_w:
                linear_interp.interp_2d_only_last_vec_mon(prep,par.grid_p,par.grid_m,sol.v[t+1],p_plus,m_plus,v_plus)
                
            # c_plus
            if compute_q:
                linear_interp.interp_2d_only_last_vec_mon(prep,par.grid_p,par.grid_m,sol.c[t+1],p_plus,m_plus,c_plus)

            # vi. accumulate all
            if compute_w:
                for ia in range(par.Na):
                    w[ip,ia] += weight*par.beta*v_plus[ia]            
            if compute_q:
                for ia in range(par.Na):
                    q[ip,ia] += weight*par.R*par.beta*utility.marg_func(c_plus[ia],par)
