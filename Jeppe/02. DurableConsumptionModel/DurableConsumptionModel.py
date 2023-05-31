# -*- coding: utf-8 -*-
"""DurableConsumptionModel

Solves a consumption-saving model with a durable consumption good and non-convex adjustment costs with either:

A. vfi: value function iteration
B. nvfi: nested value function iteration
C. negm: nested endogenous grid point method

"""

##############
# 1. imports #
##############

import time
import numpy as np

# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav.grids import nonlinspace # grids
from consav.quadrature import create_PT_shocks # income shocks

# local modules
import last_period
import post_decision
import vfi
import nvfi
import negm
import simulate
import figs

class DurableConsumptionModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def settings(self):
        """ choose settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'

        # d. list not-floats for safe type inference
        self.not_floats = ['solmethod','T','t','simN','sim_seed',
                           'Npsi','Nxi','Nm','Np','Nn','Nx','Na','Nshocks',
                           'do_print','do_print_period','do_marg_u']


    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # horizon and life cycle
        par.Tmin = 20 # age when entering the model
        par.T = 80 - par.Tmin # age of death
        par.Tr = 60 - par.Tmin # retirement age
        par.L = np.ones(par.T-1) # retirement profile
        par.L[par.Tr-1] = 0.67 # drop in permanent income at retirement age

        # tax (extension)
        par.tax_rate = 0.3 # tax rate
        par.tax_rate_vec = par.tax_rate * np.ones(par.T) # allows for tax policies
        
        # preferences
        par.beta = 0.96
        par.Delta_dispersion = 0.02
        par.Betas = np.linspace(par.beta - par.Delta_dispersion,par.beta + par.Delta_dispersion, num=3)
        par.rho = 2.0
        par.alpha = 0.9
        par.d_ubar = 1e-2

        # returns and income
        par.R = 1.03
        par.tau = 0.10
        par.delta = 0.15

        par.sigma_psi = 0.1
        par.Npsi = 5
        par.sigma_xi = 0.1
        par.Nxi = 5
        par.pi = 0.0
        par.mu = 0.5
        
        # grids
        par.Np = 50
        par.p_min = 1e-4
        par.p_max = 3.0
        par.Nn = 50
        par.n_max = 3.0
        par.Nm = 100
        par.m_max = 10.0    
        par.Nx = 100
        par.x_max = par.m_max + par.n_max
        par.Na = 100
        par.a_max = par.m_max+1.0

        # simulation
        par.sigma_p0 = 0.2
        par.mu_d0 = 0.8
        par.sigma_d0 = 0.2
        par.mu_a0 = 0.2
        par.sigma_a0 = 0.1
        par.simN = 100000
        par.sim_seed = 1998
        par.euler_cutoff = 0.02

        # misc
        par.solmethod = 'nvfi'
        par.t = 0
        par.tol = 1e-8
        par.do_print = False
        par.do_print_period = False
        par.do_marg_u = False # calculate marginal utility for use in egm
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.create_grids()
        self.solve_prep()
        self.simulate_prep()
            
    def create_grids(self):
        """ construct grids for states and shocks """
        
        par = self.par

        if par.solmethod in ['negm']:
            par.do_marg_u = True

        # a. states        
        par.grid_p = nonlinspace(par.p_min,par.p_max,par.Np,1.1)
        par.grid_n = nonlinspace(0,par.n_max,par.Nn,1.1)
        par.grid_m = nonlinspace(0,par.m_max,par.Nm,1.1)
        par.grid_x = nonlinspace(0,par.x_max,par.Nx,1.1)
        
        # b. post-decision states
        par.grid_a = nonlinspace(0,par.a_max,par.Na,1.1)
        
        # c. shocks
        shocks = create_PT_shocks(
            par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,
            par.pi,par.mu)
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # d. set seed
        np.random.seed(par.sim_seed)

        # e. timing
        par.time_w = np.zeros((par.T,3))
        par.time_keep = np.zeros((par.T,3))
        par.time_adj = np.zeros((par.T,3))
        par.time_adj_full = np.zeros((par.T,3))

    def checksum(self,simple=False,T=1):
        """ calculate and print checksum """

        par = self.par
        sol = self.sol

        if simple:
            print(f'checksum, inv_v_keep: {np.mean(sol.inv_v_keep[0]):.8f}')
            print(f'checksum, inv_v_adj: {np.mean(sol.inv_v_adj[0]):.8f}')
                
            return

        print('')
        for t in range(T):
            for b in range(len(par.Betas)):
                if t < par.T-1:
                    print(f'checksum, inv_w: {np.mean(sol.inv_w[t,b]):.8f}')
                    print(f'checksum, q: {np.mean(sol.q[t,b]):.8f}')

                print(f'checksum, c_keep: {np.mean(sol.c_keep[t,b]):.8f}')
                print(f'checksum, d_adj: {np.mean(sol.d_adj[t,b]):.8f}')
                print(f'checksum, c_adj: {np.mean(sol.c_adj[t,b]):.8f}')
                print(f'checksum, inv_v_keep: {np.mean(sol.inv_v_keep[t,b]):.8f}')
                print(f'checksum, inv_marg_u_keep: {np.mean(sol.inv_marg_u_keep[t,b]):.8f}')
                print(f'checksum, inv_v_adj: {np.mean(sol.inv_v_adj[t,b]):.8f}')
                print(f'checksum, inv_marg_u_adj: {np.mean(sol.inv_marg_u_adj[t,b]):.8f}')
                print('')


    #########
    # solve #
    #########

    def precompile_numba(self):
        """ solve the model with very coarse grids and simulate with very few persons"""

        par = self.par

        tic = time.time()

        # a. define
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_period'] = False
        fastpar['T'] = 2
        fastpar['Np'] = 3
        fastpar['Nn'] = 3
        fastpar['Nm'] = 3
        fastpar['Nx'] = 3
        fastpar['Na'] = 3
        fastpar['simN'] = 2

        # b. apply
        for key,val in fastpar.items():
            prev = getattr(par,key)
            setattr(par,key,val)
            fastpar[key] = prev

        self.allocate()

        # c. solve
        self.solve(do_assert=False)

        # d. simulate
        self.simulate()

        # e. reiterate
        for key,val in fastpar.items():
            setattr(par,key,val)
        
        self.allocate()

        toc = time.time()
        if par.do_print:
            print(f'numba precompiled in {toc-tic:.1f} secs')

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        # a. standard
        num_betas = len(par.Betas)
        keep_shape = (par.T,num_betas,par.Np,par.Nn,par.Nm)
        
        sol.c_keep = np.zeros(keep_shape)
        sol.inv_v_keep = np.zeros(keep_shape)
        sol.inv_marg_u_keep = np.zeros(keep_shape)

        adj_shape = (par.T,num_betas,par.Np,par.Nx)
        sol.d_adj = np.zeros(adj_shape)
        sol.c_adj = np.zeros(adj_shape)
        sol.inv_v_adj = np.zeros(adj_shape)
        sol.inv_marg_u_adj = np.zeros(adj_shape)
            
        post_shape = (par.T-1,num_betas,par.Np,par.Nn,par.Na)
        sol.inv_w = np.nan*np.zeros(post_shape)
        sol.q = np.nan*np.zeros(post_shape)
        sol.q_c = np.nan*np.zeros(post_shape)
        sol.q_m = np.nan*np.zeros(post_shape)

    def solve(self,do_assert=True):
        """ solve the model
        
        Args:

            do_assert (bool,optional): make assertions on the solution
        
        """

        tic = time.time()

        # backwards induction
        for t in reversed(range(self.par.T)):
            print(t)
            
            for b in range(len(self.par.Betas)):  # iterate over beta indices
                print(b)
                           
                beta = self.par.Betas[b]  # set the current beta value
                
                self.par.t = t

                with jit(self) as model:

                    par = model.par
                    sol = model.sol

                    # i. last period
                    if t == par.T-1:

                        last_period.solve(t,b,sol,par)

                        if do_assert:
                            assert np.all((sol.c_keep[t,b] >= 0) & (np.isnan(sol.c_keep[t,b]) == False))
                            assert np.all((sol.inv_v_keep[t,b] >= 0) & (np.isnan(sol.inv_v_keep[t,b]) == False))
                            assert np.all((sol.d_adj[t,b] >= 0) & (np.isnan(sol.d_adj[t,b]) == False))
                            assert np.all((sol.c_adj[t,b] >= 0) & (np.isnan(sol.c_adj[t,b]) == False))
                            assert np.all((sol.inv_v_adj[t,b] >= 0) & (np.isnan(sol.inv_v_adj[t,b]) == False))

                    # ii. all other periods
                    else:

                        # o. compute post-decision functions
                        tic_w = time.time()

                        if par.solmethod in ['nvfi']:
                            post_decision.compute_wq(t,b,sol,par,beta)
                        elif par.solmethod in ['negm']:
                            post_decision.compute_wq(t,b,sol,par,beta,compute_q=True)                

                        toc_w = time.time()
                        par.time_w[t,b] = toc_w-tic_w
                        if par.do_print:
                            print(f'  w computed in {toc_w-tic_w:.1f} secs')

                        if do_assert and par.solmethod in ['nvfi','negm']:
                            assert np.all((sol.inv_w[t,b] > 0) & (np.isnan(sol.inv_w[t,b]) == False)), t 
                            if par.solmethod in ['negm']:                                                       
                                assert np.all((sol.q[t,b] > 0) & (np.isnan(sol.q[t,b]) == False)), t

                        # oo. solve keeper problem
                        tic_keep = time.time()

                        if par.solmethod == 'vfi':
                            vfi.solve_keep(t,b,sol,par,beta)
                        elif par.solmethod == 'nvfi':                
                            nvfi.solve_keep(t,b,sol,par)
                        elif par.solmethod == 'negm':
                            negm.solve_keep(t,b,sol,par)                                       

                        toc_keep = time.time()
                        par.time_keep[t,b] = toc_keep-tic_keep
                        if par.do_print:
                            print(f'  solved keeper problem in {toc_keep-tic_keep:.1f} secs')

                        if do_assert:
                            assert np.all((sol.c_keep[t,b] >= 0) & (np.isnan(sol.c_keep[t,b]) == False)), t
                            assert np.all((sol.inv_v_keep[t,b] >= 0) & (np.isnan(sol.inv_v_keep[t,b]) == False)), t

                        # ooo. solve adjuster problem
                        tic_adj = time.time()

                        if par.solmethod == 'vfi':
                            vfi.solve_adj(t,b,sol,par,beta)
                        elif par.solmethod in ['nvfi','negm']:
                            nvfi.solve_adj(t,b,sol,par)

                        toc_adj = time.time()
                        par.time_adj[t,b] = toc_adj-tic_adj
                        if par.do_print:
                            print(f'  solved adjuster problem in {toc_adj-tic_adj:.1f} secs')

                        if do_assert:
                            assert np.all((sol.d_adj[t,b] >= 0) & (np.isnan(sol.d_adj[t,b]) == False)), t
                            assert np.all((sol.c_adj[t,b] >= 0) & (np.isnan(sol.c_adj[t,b]) == False)), t
                            assert np.all((sol.inv_v_adj[t,b] >= 0) & (np.isnan(sol.inv_v_adj[t,b]) == False)), t

                    # iii. print
                    toc = time.time()
                    if par.do_print or par.do_print_period:
                        print(f' t = {t} solved in {toc-tic:.1f} secs')

    ############
    # simulate #
    ############

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # a. initial and final
        sim.p0 = np.zeros((len(par.Betas),par.simN))
        sim.d0 = np.zeros((len(par.Betas),par.simN))
        sim.a0 = np.zeros((len(par.Betas),par.simN))

        sim.utility = np.zeros((len(par.Betas),par.simN))

        # b. states and choices
        sim_shape = (par.T,len(par.Betas),par.simN)
        sim.p = np.zeros(sim_shape)
        sim.m = np.zeros(sim_shape)
        sim.n = np.zeros(sim_shape)
        sim.discrete = np.zeros(sim_shape,dtype=np.int)
        sim.d = np.zeros(sim_shape)
        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        sim.y = np.zeros(sim_shape)
        
        # c. euler
        euler_shape = (par.T-1,len(par.Betas),par.simN)
        sim.euler_error = np.zeros(euler_shape)
        sim.euler_error_c = np.zeros(euler_shape)
        sim.euler_error_rel = np.zeros(euler_shape)

        # d. shocks
        sim.psi = np.zeros((par.T,par.simN))
        sim.xi = np.zeros((par.T,par.simN))

    def simulate(self,do_utility=False,do_euler_error=False):
        """ simulate the model """

        par = self.par
        sol = self.sol
        sim = self.sim

        tic = time.time()

        # a. random shocks
        sim.p0[:,:] = np.random.lognormal(mean=0,sigma=par.sigma_p0,size=(len(par.Betas),par.simN))
        sim.d0[:,:] = par.mu_d0*np.random.lognormal(mean=0,sigma=par.sigma_d0,size=(len(par.Betas),par.simN))
        sim.a0[:,:] = par.mu_a0*np.random.lognormal(mean=0,sigma=par.sigma_a0,size=(len(par.Betas),par.simN))

        I = np.random.choice(par.Nshocks,
            size=(par.T,par.simN), 
            p=par.psi_w*par.xi_w)
        sim.psi[:,:] = par.psi[I]
        sim.xi[:,:] = par.xi[I]

        # b. call
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim

            simulate.lifecycle(sim,sol,par)

        toc = time.time()
        
        if par.do_print:
            print(f'model simulated in {toc-tic:.1f} secs')

        # d. euler errors
        def norm_euler_errors(model):
            return np.log10(abs(model.sim.euler_error/model.sim.euler_error_c)+1e-8)

        tic = time.time()        
        if do_euler_error:

            with jit(self) as model:

                par = model.par
                sol = model.sol
                sim = model.sim

                simulate.euler_errors(sim,sol,par,beta)
            
            sim.euler_error_rel[:] = norm_euler_errors(self)
        
        toc = time.time()
        if par.do_print:
            print(f'euler errors calculated in {toc-tic:.1f} secs')

        # e. utility
        tic = time.time()        
        if do_utility:
            simulate.calc_utility(sim,sol,par)
        
        toc = time.time()
        if par.do_print:
            print(f'utility calculated in {toc-tic:.1f} secs')    
    
    ########
    # figs #
    ########

    def decision_functions(self):
        figs.decision_functions(self)

    def egm(self):        
        figs.egm(self)

    def lifecycle(self):        
        figs.lifecycle(self)
        
    def lifecycle_rand(self):        
        figs.lifecycle_rand(self)

    ###########
    # analyze #
    ###########

    def analyze(self,solve=True,do_assert=True,**kwargs):
        
        par = self.par

        for key,val in kwargs.items():
            setattr(par,key,val)
        
        self.create_grids()

        # solve and simulate
        if solve:
            self.precompile_numba()
            self.solve(do_assert)
        self.simulate(do_euler_error=True,do_utility=True)

        # print
        self.print_analysis()

    def print_analysis(self):

        par = self.par
        sim = self.sim

        def avg_euler_error(model,I):
            if I.any():
                return np.nanmean(model.sim.euler_error_rel.ravel()[I])
            else:
                return np.nan

        def percentile_euler_error(model,I,p):
            if I.any():
                return np.nanpercentile(model.sim.euler_error_rel.ravel()[I],p)
            else:
                return np.nan

        # population
        keepers = sim.discrete[:-1,:].ravel() == 0
        adjusters = sim.discrete[:-1,:].ravel() > 0
        everybody = keepers | adjusters

        # print
        time = par.time_w+par.time_adj+par.time_keep
        txt = f'Name: {self.name} (solmethod = {par.solmethod})\n'
        txt += f'Grids: (p,n,m,x,a) = ({par.Np},{par.Nn},{par.Nm},{par.Nx},{par.Na})\n'
        
        txt += 'Timings:\n'
        txt += f' total: {np.sum(time):.1f}\n'
        txt += f'     w: {np.sum(par.time_w):.1f}\n'
        txt += f'  keep: {np.sum(par.time_keep):.1f}\n'
        txt += f'   adj: {np.sum(par.time_adj):.1f}\n'
        txt += f'Utility: {np.mean(sim.utility):.6f}\n'
        
        txt += 'Euler errors:\n'
        txt += f'     total: {avg_euler_error(self,everybody):.2f} ({percentile_euler_error(self,everybody,5):.2f},{percentile_euler_error(self,everybody,95):.2f})\n'
        txt += f'   keepers: {avg_euler_error(self,keepers):.2f} ({percentile_euler_error(self,keepers,5):.2f},{percentile_euler_error(self,keepers,95):.2f})\n'
        txt += f' adjusters: {avg_euler_error(self,adjusters):.2f} ({percentile_euler_error(self,adjusters,5):.2f},{percentile_euler_error(self,adjusters,95):.2f})\n'

        txt += 'Moments:\n'
        txt += f' adjuster share: {np.mean(sim.discrete):.3f}\n'
        txt += f'         mean c: {np.mean(sim.c):.3f}\n'
        txt += f'          var c: {np.var(sim.c):.3f}\n'

        txt += f'         mean d: {np.mean(sim.d):.3f}\n'
        txt += f'          var d: {np.var(sim.d):.3f}\n'

        print(txt)