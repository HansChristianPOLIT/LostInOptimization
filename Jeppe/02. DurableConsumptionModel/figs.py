import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets

from consav import linear_interp

# local modules
import utility

######################
# decision functions #
######################

def _decision_functions(model,t,i_p,name):

    if name == 'discrete':
        _discrete(model,t,i_p)
    elif name == 'adj':
        _adj(model,t,i_p)
    elif name == 'keep':
        _keep(model,t,i_p)
    elif name == 'post_decision' and t <= model.par.T-2:
        _w(model,t,i_p)        

def decision_functions(model):
    widgets.interact(_decision_functions,
        model=widgets.fixed(model),
        t=widgets.Dropdown(description='t', 
            options=list(range(model.par.T)), value=0),
        i_p=widgets.Dropdown(description='ip', 
            options=list(range(model.par.Np)), value=np.int(model.par.Np/2)),
        name=widgets.Dropdown(description='name', 
            options=['discrete','adj','keep','post_decision'], value='discrete')
        )

def _discrete(model,t,i_p):

    par = model.par

    # a. interpolation
    n, m = np.meshgrid(par.grid_n,par.grid_m,indexing='ij')
    x = m + (1-par.tau)*n
    
    inv_v_adj = np.zeros(x.size)
    linear_interp.interp_1d_vec(par.grid_x,model.sol.inv_v_adj[t,i_p,:,],x.ravel(),inv_v_adj)
    inv_v_adj = inv_v_adj.reshape(x.shape)

    # f. best discrete choice
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    I = inv_v_adj > model.sol.inv_v_keep[t,i_p,:,:]

    x = m[I].ravel()
    y = n[I].ravel()
    ax.scatter(x,y,s=2,label='adjust')
    
    x = m[~I].ravel()
    y = n[~I].ravel()
    ax.scatter(x,y,s=2,label='keep')
        
    ax.set_title(f'optimal discrete choice ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$)',pad=10)

    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # g. details
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_xlim([par.grid_m[0],par.grid_m[-1]])
    ax.set_ylabel('$n_t$')
    ax.set_ylim([par.grid_n[0],par.grid_n[-1]])
    
    plt.show()

def _adj(model,t,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_b = fig.add_subplot(1,2,1)
    ax_v = fig.add_subplot(1,2,2)
    
    # c. plot consumption
    ax_b.plot(par.grid_x,sol.d_adj[t,i_p,:],lw=2)
    ax_b.set_title(f'$d^{{adj}}$ ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$)',pad=10)

    # d. plot value function
    ax_v.plot(par.grid_x,sol.inv_v_adj[t,i_p,:],lw=2)
    ax_v.set_title(f'neg. inverse $v^{{adj}}$ ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$)',pad=10)

    # e. details
    for ax in [ax_b,ax_v]:
        ax.grid(True)
        ax.set_xlabel('$x_t$')
        ax.set_xlim([par.grid_x[0],par.grid_x[-1]])

    plt.show()

def _w(model,t,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1,projection='3d')

    n,a = np.meshgrid(par.grid_n, par.grid_a,indexing='ij')

    # c. plot consumption
    ax.plot_surface(n,a,sol.inv_w[t,i_p,:,:],cmap=cm.viridis,edgecolor='none')
    ax.set_title(f'neg. inverse  $w$ ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$)',pad=10)

    # d. details
    ax.grid(True)
    ax.set_xlabel('$n_t$')
    ax.set_xlim([par.grid_n[0],par.grid_n[-1]])
    ax.set_ylabel('$a_t$')
    ax.set_ylim([par.grid_a[0],par.grid_a[-1]])
    ax.invert_xaxis()

    plt.show()

def _keep(model,t,i_p):

    # unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_c = fig.add_subplot(1,2,1,projection='3d')
    ax_v = fig.add_subplot(1,2,2,projection='3d')

    n,m = np.meshgrid(par.grid_n, par.grid_m,indexing='ij')

    # c. plot consumption
    ax_c.plot_surface(n,m,sol.c_keep[t,i_p,:,:],cmap=cm.viridis,edgecolor='none')
    ax_c.set_title(f'$c^{{keep}}$ ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$)',pad=10)

    # d. plot value function
    ax_v.plot_surface(n,m,sol.inv_v_keep[t,i_p,:,:],cmap=cm.viridis,edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{keep}}$ ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$)',pad=10)

    # e. details
    for ax in [ax_c,ax_v]:

        ax.grid(True)
        ax.set_xlabel('$n_t$')
        ax.set_xlim([par.grid_n[0],par.grid_n[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([par.grid_m[0],par.grid_m[-1]])
        ax.invert_xaxis()

    plt.show()

#######
# egm #
#######

def egm(model):
    widgets.interact(_egm,
        model=widgets.fixed(model),
        t=widgets.Dropdown(description='t', 
            options=list(range(model.par.T-1)), value=0),
        i_p=widgets.Dropdown(description='ip', 
            options=list(range(model.par.Np)), value=np.int(model.par.Np/2)),
        i_n=widgets.Dropdown(description='in', 
            options=list(range(model.par.Nn)), value=np.int(model.par.Nn/2))
        )

def _egm(model,t,i_p,i_n):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_c = fig.add_subplot(1,2,1)
    ax_v = fig.add_subplot(1,2,2)
    
    # c. plot before
    c_vec = sol.q_c[t,i_p,i_n]
    m_vec = sol.q_m[t,i_p,i_n]
    ax_c.plot(m_vec,c_vec,'o',MarkerSize=0.5,label='before')
    ax_c.set_title(f'$c$ ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$, $n = {par.grid_n[i_n]:.2f}$)',pad=10)

    inv_v_vec = np.zeros(par.Na)
    for i_a in range(par.Na):
        inv_v_vec[i_a] = utility.func(c_vec[i_a],par.grid_n[i_n],par) + (-1/sol.inv_w[t,i_p,i_n,i_a])
    inv_v_vec = -1.0/inv_v_vec

    ax_v.plot(m_vec,inv_v_vec,'o',MarkerSize=0.5,label='before')
    ax_v.set_title(f'neg. inverse $v$ ($t = {t}$, $p = {par.grid_p[i_p]:.2f}$, $n = {par.grid_n[i_n]:.2f}$)',pad=10)

    # d. plot after
    c_vec = sol.c_keep[t,i_p,i_n,:]
    ax_c.plot(par.grid_m,c_vec,'o',MarkerSize=0.5,label='after')
    
    inv_v_vec = sol.inv_v_keep[t,i_p,i_n,:]
    ax_v.plot(par.grid_m,inv_v_vec,'o',MarkerSize=0.5,label='after')

    # e. details
    ax_c.legend()
    ax_c.set_ylabel('$c_t$')
    ax_c.set_ylim([c_vec[0],c_vec[-1]])
    ax_v.set_ylim([inv_v_vec[0],inv_v_vec[-1]])
    for ax in [ax_c,ax_v]:
        ax.grid(True)
        ax.set_xlabel('$m_t$')
        ax.set_xlim([par.grid_m[0],par.grid_m[-1]])

    plt.show()

#############
# lifecycle #
#############

def lifecycle(model):

    # a. unpack
    par = model.par
    sim = model.sim

    simvarlist = [('p','(A): Permanent income, $p_t$'),
                  ('n','(B): Housing stock, $n_t$'),
                  ('m','(C): Cash-on-hand, $m_t$'),
                  ('c','(D): Consumption, ,$c_t$'),
                  ('a','(E): Savings, ,$a_t$'),
                  ('discrete','(F): ,Adjuster share')]

    # b. figure
    fig, axs = plt.subplots(2, 3, figsize=(12, 12))

    age = np.arange(par.T)
    for i, (simvar, simvarlatex) in enumerate(simvarlist): # Loop over different simulation variables
        ax = axs[i // 3, i % 3]  # specify the current subplot

        for b in range(len(par.Betas)): # Loop over different beta values
            # get simulation data for this beta value
            simdata = getattr(sim, simvar)[:, b, :]

            # calculate mean over individual dimension for each beta
            mean_data = np.mean(simdata, axis=1)

            # plot mean data
            ax.plot(age, mean_data, label=f'{simvarlatex}, Beta {b+1}')

        ax.set_title(simvarlatex)
        if par.T > 10:
            ax.xaxis.set_ticks(age[::5])
        else:
            ax.xaxis.set_ticks(age)

        ax.grid(True)
        ax.set_xlabel('age')
        legend = ax.legend(frameon=True,prop={'size': 8})
        frame = legend.get_frame()
        frame.set_edgecolor('black')

    # adjust layout for better visualization
    plt.tight_layout()
    plt.show()

    
def lifecycle_rand(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(12,12))

    simvarlist = [('p','(A): Permanent income, $p_t$'),
                  ('n','(B): Housing stock, $n_t$'),
                  ('m','(C): Cash-on-hand, $m_t$'),
                  ('c','(D): Consumption, $c_t$'),
                  ('a','(E): Savings, $a_t$'),
                  ('discrete','(F): Adjuster share (in percent)')]

    age = np.arange(par.T)
    for i,(simvar,simvarlatex) in enumerate(simvarlist):

        ax = fig.add_subplot(3,2,i+1)

        simdata_raw = getattr(sim,simvar)[:par.T,:]  # get the raw simulation data
        simdata = np.mean(simdata_raw, axis=1)  # aggregate over the second dimension

        if simvar == 'discrete':
            simdata *= 100  # Convert to percentage

        # mean graph
        ax.plot(age,np.mean(simdata,axis=1),lw=2, label = 'mean')

        # quantile graphs 
        ax.plot(age,np.percentile(simdata,90,axis=1),
            ls='--',lw=1,color='purple', label='90% quantile')
        ax.plot(age,np.percentile(simdata,75,axis=1),
            ls='--',lw=1,color='black', label='75% quantile')
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
        if simvar in ['a','discrete']:
            ax.set_xlabel('age')

        if simvar == 'discrete':
            ax.set_ylabel('percent')

        if simvar in ['p']:
            legend = ax.legend(frameon=True,prop={'size': 8})
            frame = legend.get_frame()
            frame.set_edgecolor('black')

    plt.show()
