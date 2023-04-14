import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

import ipywidgets as widgets

def consumption_function(model,t):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')

    p,m = np.meshgrid(par.grid_p, par.grid_m,indexing='ij')

    # c. plot consumption
    ax.plot_surface(p,m,sol.c[t,:,:],edgecolor='none',cmap=cm.viridis)
    ax.set_title(f'$c$ ($t = {t})$',pad=10)

    # d. details
    ax.grid(True)
    ax.set_xlabel('$p_t$')
    ax.set_xlim([par.grid_p[0],par.grid_p[-1]])
    ax.set_ylabel('$m_t$')
    ax.set_ylim([par.grid_m[0],par.grid_m[-1]])
    ax.invert_xaxis()

    plt.show()

def consumption_function_interact(model):

    widgets.interact(consumption_function,
        model=widgets.fixed(model),
        t=widgets.Dropdown(description='t', 
            options=list(range(model.par.T)), value=0),
        )          

def lifecycle(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)

    # Settings
    ax.grid(b=True, which='major', linestyle='-', linewidth=0.5, color='0.9')
    ax.set_xlim([0.0, par.T - 1])
    ax.set_xlabel('Age', size=13)
    ax.set_ylabel('Simulation Variable', size=13)

    simvarlist = [('m', '$m_t$', '-'),
                  ('c', '$c_t$', '--'),
                  ('a', '$a_t$', '-.')]

    age = np.arange(par.T)

    for simvar, simvarlatex, linestyle in simvarlist:
        simdata = getattr(sim, simvar)
        ax.plot(age, np.mean(simdata, axis=1), lw=2, label=simvarlatex, linestyle=linestyle, color='0.4')

    ax.legend(frameon=True, edgecolor='k', facecolor='white', framealpha=1, fancybox=False, loc=2)
    ax.grid(True)

    
def plot_age_consumption(model):
    par = model.par
    sol = model.sol
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    median_p_index = par.Np // 2

    
    simvarlist = [('m', '$m_t$', '-'),
                  ('c', '$c_t$', '--'),
                  ('a', '$a_t$', '-.')]

    for age in range(par.T):
        ax.plot(par.grid_m, sol.c[age, median_p_index, :], label=f'age = {age}', linestyle='-', color='0.4')

    ax.set_xlabel(r'Cash-on-hand, $m_t$',size=13)
    ax.set_ylabel(r'Consumption, $c(m_t)$',size=13)
    ax.set_xlim([np.min(par.grid_m), 5])
    ax.set_ylim([0, 5])
    ax.set_title(r'Consumption function')

    ax.legend(frameon=True, edgecolor='k', facecolor='white', framealpha=1, fancybox=False)
    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5, color='0.9')

    plt.show()