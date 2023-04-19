""" BufferStockModel

Solves the Deaton-Carroll buffer-stock one-asset consumption model with either:
A. vfi: standard value function iteration - start with this
#B. nvfi: nested value function iteration
#C. egm: endogenous grid point method

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
import vfi
import simulate
import figs


############
# 2. model #
############

class OneAssetModelClass(ModelClass):

    def settings(self):
        """ Fundamental settings. """

        # a. namespaces
        self.namespaces = [] # keep track of par,sol, sim,

        # b. other attributes
        self.other_attrs = []

        # c. savefolder 
        self.savefolder = 'saved' 

         # d. list not-floats for safe type inference when JIT compilation with Numba
        self.not_floats = ['T','Npsi','Nxi','Nm','Np','Na','do_print','do_simple_w','simT','simN','sim_seed','Nshocks'] 

    def setup(self):
        """ Set baseline parameters. """

        # unpack 
        par = self.par

        # a. solution method
        par.solmethod = 'vfi'

        # b. horizon 
        par.T = 5 # Time horizon

        # c. preferences
        par.beta = 0.96
        par.rho = 2.0

        # d. returns and iccome
        par.R = 1.03
        par.sigma_psi = 0.1 # SD of permanent income process
        # think of vector of points and vector of weights to approximate distribution
        par.Npsi = 6 # no. of quadrature points to approximate distribution of permanent income shock
        par.sigma_xi = 0.1 # SD of transitory income process
        par.Nxi = 6 # no. of quadrature points to approximate distribution of transitory income shock
        par.pi = 0.1 # probability of unemployment
        par.mu = 0.5 # unemployment benefits

        # e. grids (number of points)
        par.Nm = 600 # nodes for m grid
        par.Np = 400 # nodes for p grid
        par.Na = 800 # nodes for a grid

        # f. misc
        par.tol = 1e-8
        par.do_print = True
        par.do_simple_w = False

        # g. simulation
        par.simT = par.T
        par.simN = 1_000
        par.sim_seed = 1998


    def allocate(self):
        """ Allocate model components. """

        self.create_grids()
        self.solve_prep()
        self.simulate_prep()

    def create_grids(self):
        """ Construct grids for states and shocks. """

        # unpack 
        par = self.par

        # a. states (using non-linearly spaced vector)
        par.grid_m = nonlinspace(1e-6,20,par.Nm,1.1)
        par.grid_p = nonlinspace(1e-4,10,par.Np,1.1)

        # b. post-decision states (non-linearly spaced vector)
        par.grid_a = nonlinspace(1e-6,20,par.Na,1.1)

        # c. shocks (quadrature nodes and weights using GaussHermite)
        shocks = create_PT_shocks(
            par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,
            par.pi,par.mu)
        # unpack values
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # d. set seed
        np.random.seed(self.par.sim_seed) # ensures reproducibility

    def checksum(self):
        """ Print checksum. """

        return np.mean(self.sol.c[0]) # mean of consumption in first period
    #########
    # solve #
    #########

    def solve_prep(self):
        """ Allocate memory for solution. """

        # unpack
        par = self.par
        sol = self.sol

        sol.c = np.nan*np.ones((par.T,par.Np,par.Nm))
        sol.v = np.nan*np.zeros((par.T,par.Np,par.Nm))
        sol.w = np.nan*np.zeros((par.Np,par.Na))
        sol.q = np.nan*np.zeros((par.Np,par.Na))

    def solve(self):
        """ Solve the model using solmethod. """

        with jit(self) as model: # can now call jitted functions

            # unpack 
            par = model.par
            sol = model.sol

            # backwards induction
            for t in reversed(range(par.T)):

                t0 = time.time()

                # a. last period
                if t == par.T-1:

                    last_period.solve(t,sol,par)

                # b. all other periods
                else:
                    t1_w = time.time()

                    if par.solmethod == 'vfi':
                        vfi.solve_bellman(t,sol,par)

                    else:
                        raise ValueError(f'Unknown solution method, {par.solmethod}')
                # c. print
                if par.do_print:
                    msg = f' t = {t} solved in {elapsed(t0)}'
                    if t < par.T-1:
                        msg += f' (w: {elapsed(t1_w)})'
                    print(msg)

    ############
    # simulate #
    ############

    def simulate_prep(self):
        """ Allocate memory for the simulation. """

        # unpack
        par = self.par
        sim = self.sim

        # a. allocate
        sim.p = np.nan*np.zeros((par.simT,par.simN)) # NaN helps identify problems 
        sim.m = np.nan*np.zeros((par.simT,par.simN))
        sim.c = np.nan*np.zeros((par.simT,par.simN))
        sim.a = np.nan*np.zeros((par.simT,par.simN))

        # b. draw random shocks
        sim.psi = np.ones((par.simT,par.simN))
        sim.xi = np.ones((par.simT,par.simN))


    def simulate(self):
        "Simulate the model. """

        with jit(self) as model: # can now call jitted functions

            # unpack
            par = model.par
            sol = model.sol
            sim = model.sim

            t0 = time.time()

            # a. allocate memory and draw random numbers
            I = np.random.choice(par.Nshocks, size=(par.T,par.simN),p=par.psi_w*par.xi_w) #  Draws random shock indices from combined probabilities of par.psi_w and par.xi_w

            sim.psi[:] = par.psi[I]
            sim.xi[:] = par.xi[I]

            # b. simulate 
            simulate.lifecycle(sim,sol,par)

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
        
    def plot_age_consumption(self):
        figs.plot_age_consumption(self)
            
        