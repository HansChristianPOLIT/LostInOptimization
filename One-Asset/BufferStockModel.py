# -*- coding: utf-8 -*-
"""BufferStockModel

Solves the Deaton-Carroll buffer-stock consumption model with either:

A. vfi: standard value function iteration
B. nvfi: nested value function iteration
C. egm: endogenous grid point method

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
from consav.misc import elapsed

# local modules
import utility
import last_period
import post_decision
import vfi
import nvfi
import egm
import simulate
import figs

############
# 2. model #
############

class BufferStockModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. list not-floats for safe type inference
        self.not_floats = ['T','Npsi','Nxi','Nm','Np','Na','do_print','simT','simN','sim_seed','Nshocks']
        
    def setup(self):
        """ set baseline parameters """   

        par = self.par

        # a. solution method
        par.solmethod = 'nvfi'
        
        # b. horizon and life cycle
        par.Tmin = 20 # age when entering the model
        par.T = 80 - par.Tmin # age of death
        par.Tr = 60 - par.Tmin # retirement age
        par.L = np.ones(par.T-1) # retirement profile
        par.L[par.Tr-1] = 0.67 #0.67 # drop in permanent income at retirement age
        
        # c. preferences
        par.beta = 0.95
        par.Delta_dispersion = 0.02
        par.Betas = np.linspace(par.beta - par.Delta_dispersion,par.beta + par.Delta_dispersion, num=3)
        
        par.rho = 2.0 
        
        # tax (extension)
        par.tax_rate = 0.51 # tax rate
        par.tax_rate_vec = par.tax_rate * np.ones(par.T) # Used for simulating elasticities
        par.eps = 0.0065648 # deduced tax rebate from Kaplan & Violante 2022
        
        # d. returns and income
        par.R = 1.03
        par.sigma_psi = 0.1
        par.Npsi = 6
        par.sigma_xi = 0.1
        par.Nxi = 6
        par.pi = 0.0
        par.mu = 0.5
        
        # e. grids (number of points)
        par.Nm = 600
        par.Np = 400
        par.Na = 800

        # f. misc
        par.tol = 1e-8
        par.do_print = True

        # g. simulation
        par.simT = par.T
        par.simN = 1000
        par.sim_seed = 1998
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        self.create_grids()
        self.solve_prep()
        self.simulate_prep()
        self.simulate_prep_rand()

    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        # a. states (unequally spaced vectors of length Nm)
        par.grid_m = nonlinspace(1e-6,20,par.Nm,1.1)
        par.grid_p = nonlinspace(1e-4,10,par.Np,1.1)
        
        # b. post-decision states (unequally spaced vector of length Na)
        par.grid_a = nonlinspace(1e-6,20,par.Na,1.1)
        
        # c. shocks (qudrature nodes and weights using GaussHermite)
        shocks = create_PT_shocks(
            par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,
            par.pi,par.mu)
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # d. set seed
        np.random.seed(self.par.sim_seed)

    def checksum(self):
        """ print checksum """

        return np.mean(self.sol.c[0])

    #########
    # solve #
    #########

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol
        
        num_betas = len(par.Betas)

        sol.c = np.nan*np.ones((par.T,num_betas,par.Np,par.Nm))        
        sol.v = np.nan*np.zeros((par.T,num_betas,par.Np,par.Nm))
        sol.w = np.nan*np.zeros((num_betas,par.Np,par.Na))
        sol.q = np.nan*np.zeros((num_betas,par.Np,par.Na))

    def solve(self):
        """ solve the model using solmethod """

        with jit(self) as model: # can now call jitted functions

            par = model.par
            sol = model.sol

            num_betas = len(par.Betas)  # number of beta values

            # backwards induction
            for t in reversed(range(par.T)):
                for b in range(num_betas):  # iterate over beta indices

                    beta = par.Betas[b]  # set the current beta value

                    t0 = time.time()

                    # a. last period
                    if t == par.T-1:

                        last_period.solve(t,b,sol,par)

                    # b. all other periods
                    else:

                        # i. compute post-decision functions
                        t0_w = time.time()

                        compute_w,compute_q = False,False
                        if par.solmethod in ['nvfi']: compute_w = True
                        elif par.solmethod in ['egm']: compute_q = True

                        if compute_w or compute_q:

                            post_decision.compute_wq(t,b,sol,par,beta,compute_w=compute_w,compute_q=compute_q)

                        t1_w = time.time()

                        # ii. solve bellman equation
                        if par.solmethod == 'vfi':
                            vfi.solve_bellman(t,b,sol,par,beta)                    
                        elif par.solmethod == 'nvfi':
                            nvfi.solve_bellman(t,b,sol,par)
                        elif par.solmethod == 'egm':
                            egm.solve_bellman(t,b,sol,par)                    
                        else:
                            raise ValueError(f'unknown solution method, {par.solmethod}')

                    # c. print
                    if par.do_print:
                        msg = f' t = {t} solved in {elapsed(t0)}'
                        if t < par.T-1:
                            msg += f' (w: {elapsed(t0_w,t1_w)})'                
                        print(msg)

    ############
    # simulate #
    ############

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim
        num_betas = len(par.Betas)
        
        # a. allocate
        sim_shape = (par.simT,num_betas,par.simN)
        sim.p = np.nan*np.zeros(sim_shape)
        sim.m = np.nan*np.zeros(sim_shape)
        sim.c = np.nan*np.zeros(sim_shape)
        sim.a = np.nan*np.zeros(sim_shape)
        sim.y = np.nan*np.zeros(sim_shape)
        
        # MPC
        sim.mpc = np.nan*np.zeros(sim_shape)
        sim.c_eps = np.zeros(sim_shape)
        
        # random uniform draws of time preference
        sim.beta = np.random.choice(par.Betas, par.simN)

        # b. draw random shocks
        sim.psi = np.ones((par.simT,par.simN))
        sim.xi = np.ones((par.simT,par.simN))

    def simulate(self):
        """ simulate model """

        with jit(self) as model: # can now call jitted functions 

            par = model.par
            sol = model.sol
            sim = model.sim
            
            t0 = time.time()

            # a. allocate memory and draw random numbers
            I = np.random.choice(par.Nshocks,
                size=(par.T,par.simN), 
                p=par.psi_w*par.xi_w)

            sim.psi[:] = par.psi[I]
            sim.xi[:] = par.xi[I]

            # b. simulate
            simulate.lifecycle(sim,sol,par)
            
            if par.do_print:
                print(f'model simulated in {elapsed(t0)}')

    def simulate_prep_rand(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim
        num_betas = len(par.Betas)

        # a. allocate
        sim_rand_shape = (par.simT,par.simN)
        sim.p_rand = np.nan*np.zeros(sim_rand_shape)
        sim.m_rand = np.nan*np.zeros(sim_rand_shape)
        sim.c_rand = np.nan*np.zeros(sim_rand_shape)
        sim.a_rand = np.nan*np.zeros(sim_rand_shape)
        sim.y_rand = np.nan*np.zeros(sim_rand_shape)
        
        # MPC
        sim.mpc_rand = np.nan*np.zeros(sim_rand_shape)
        sim.c_eps_rand = np.nan*np.zeros(sim_rand_shape)

        # b. draw random shocks
        sim.psi = np.ones(sim_rand_shape)
        sim.xi = np.ones(sim_rand_shape)

        # c. assign a random beta index to each individual
        sim.beta_rand = np.random.choice(num_betas, par.simN)
    
    def simulate_rand(self):
        """ simulate rand model """

        with jit(self) as model: # can now call jitted functions 

            par = model.par
            sol = model.sol
            sim = model.sim

            t0 = time.time()

            # a. allocate memory and draw random numbers
            I = np.random.choice(par.Nshocks,
                size=(par.T,par.simN), 
                p=par.psi_w*par.xi_w)

            sim.psi[:] = par.psi[I]
            sim.xi[:] = par.xi[I]

            # b. simulate
            simulate.lifecycle_rand(sim,sol,par)

            if par.do_print:
                print(f'model simulated in {elapsed(t0)}')


    ########
    # figs #
    ########

    def consumption_function(self,t=0):
        figs.consumption_function(self,t)

    def consumption_function_interact(self):
        figs.consumption_function_interact(self)
          
    def lifecycle(self):
        figs.lifecycle(self)    
        
    def beta_check(self):
        figs.beta_check(self)
            
    def beta_check_simple(self):
        figs.beta_check_simple(self)