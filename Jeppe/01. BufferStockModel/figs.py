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

    simvarlist = [('p','(A): Permanent Income, $p_t$'),
                  ('m','(B): Cash-on-Hand, $m_t$'),
                  ('c','(C): Consumption, $c_t$'),
                  ('a','(D): Savings, $a_t$')]

    age = np.arange(par.T)
    for i,(simvar,simvarlatex) in enumerate(simvarlist):
        ax = fig.add_subplot(3,2,i+1)

        simdata = getattr(sim,simvar+'_rand')[:par.T,:]

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

        # Change x-axis labels to show 20 years later
        xticks = np.arange(0, 61, 5)  # make sure it includes the last value you want to display
        ax.set_xticks(xticks)
        new_xticks = xticks + 20
        ax.set_xticklabels(new_xticks)
        ax.set_xlim(-3, 62)  # increase the upper limit of x-axis

        ax.grid(True)
        ax.set_xlabel('Age')

        if simvar in ['p']:
            legend = ax.legend(frameon=True,prop={'size': 8})
            frame = legend.get_frame()
            frame.set_edgecolor('black')
            
    plt.subplots_adjust(hspace=0.27)  
            
    plt.savefig("one_asset_lifecycle_plot.pdf", bbox_inches='tight')

    plt.show()

def beta_check(model):

        # a. unpack
        par = model.par
        sim = model.sim

        # b. figure
        fig, axs = plt.subplots(len(par.Betas), 1, figsize=(10, 8))

        simvarlist = [('m','$m_t$'),
                    ('c','$c_t$'),
                    ('a','$a_t$'),
                    ('y','$y_t$')]

        age = np.arange(par.T)

        for b in range(len(par.Betas)):  # loop over beta dimension
            ax = axs[b]  # specify the current subplot

            for simvar, simvarlatex in simvarlist:
                # get simulation data
                simdata = getattr(sim, simvar)

                # calculate mean over individual dimension for each beta
                mean_data = np.mean(simdata[:, b, :], axis=1)

                # plot mean data
                ax.plot(age, mean_data, label=simvarlatex)

            # Change x-axis labels to show 20 years later
            xticks = np.arange(0, 61, 5)  # make sure it includes the last value you want to display
            ax.set_xticks(xticks)
            new_xticks = xticks + 20
            ax.set_xticklabels(new_xticks)
            ax.set_xlim(-3, 62)  # increase the upper limit of x-axis

            ax.grid(True)
            ax.set_xlabel('age')
            ax.set_title(f'Beta {b+1}')
            ax.legend()

        # adjust layout for better visualization
        plt.tight_layout()
        plt.show()


def beta_check_simple(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure()

    simvarlist = [('m','$m_t$'),
                ('c','$c_t$'),
                ('a','$a_t$'),
                ('y','$y_t$'),
                ('p','$p_t$')]

    age = np.arange(par.T)
    ax = fig.add_subplot(1,1,1)

    for simvar,simvarlatex in simvarlist:

        simdata = getattr(sim,simvar+'_rand')
        ax.plot(age,np.mean(simdata,axis=1),lw=2,label=simvarlatex)


    # Change x-axis labels to show 20 years later
    xticks = np.arange(0, 61, 5)  # make sure it includes the last value you want to display
    ax.set_xticks(xticks)
    new_xticks = xticks + 20
    ax.set_xticklabels(new_xticks)
    ax.set_xlim(-3, 62)  # increase the upper limit of x-axis

    ax.legend(frameon=True)
    ax.grid(True)
    ax.set_xlabel('age')