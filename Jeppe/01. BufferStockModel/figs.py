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
    fig = plt.figure(figsize=(12,12))

    simvarlist = [('p','(A): Permanent income, $p_t$'),
                  ('m','(B): Cash-on-hand, $m_t$'),
                  ('c','(C): Consumption, $c_t$'),
                  ('a','(D): Savings, $a_t$')]

    age = np.arange(par.T)
    for i,(simvar,simvarlatex) in enumerate(simvarlist):

        ax = fig.add_subplot(3,2,i+1)

        simdata = getattr(sim,simvar+'_rand')[:par.T,:]

        ax.plot(age,np.percentile(simdata,90,axis=1),
            ls='--',lw=1,color='purple', label='90% quantile')
        ax.plot(age,np.percentile(simdata,75,axis=1),
            ls='--',lw=1,color='black', label='75% quantile')
        ax.plot(age,np.mean(simdata,axis=1),lw=2, label = 'mean')
        ax.plot(age,np.percentile(simdata,50,axis=1),
            ls='--',lw=1,color='orange', label='50% quantile')
        ax.plot(age,np.percentile(simdata,25,axis=1),
            ls='--',lw=1,color='red', label='25% quantile')
        ax.plot(age,np.percentile(simdata,10,axis=1),
            ls='--',lw=1,color='green', label='10% quantile')

        ax.set_title(simvarlatex)
        if par.T > 10:
            ax.xaxis.set_ticks(age[::5])
        else:
            ax.xaxis.set_ticks(age)

        ax.grid(True)
        if simvar in ['c','a']:
            ax.set_xlabel('age')

        if simvar in ['p']:
            legend = ax.legend(frameon=True,prop={'size': 8})
            frame = legend.get_frame()
            frame.set_edgecolor('black')

    plt.show()
