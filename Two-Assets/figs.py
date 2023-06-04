import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets

from consav import linear_interp

# local modules
import utility

#############
# lifecycle #
#############

def lifecycle(model):

    # a. unpack
    par = model.par
    sim = model.sim

    simvarlist = [('p','(A): Permanent Income, $p_t$'),
                  ('y','(B): Transitory Income, $y_t$'),
                  ('n','(C): Housing Stock, $n_t$'),
                  ('d','(D): Housing Stock Chosen, $d_t$'),
                  ('m','(E): Cash-on-Hand, $m_t$'),
                  ('c','(F): Consumption, $c_t$'),
                  ('a','(G): Savings, $a_t$'),
                  ('discrete','(H): Adjuster Share (in Percent)')]

    # b. figure
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # c. beta mapping
    beta_map = {0: 0.93, 1: 0.95, 2: 0.97}

    age = np.arange(par.T)
    for i, (simvar, simvarlatex) in enumerate(simvarlist): # Loop over different simulation variables
        ax = axs[i // 4, i % 4]  # specify the current subplot

        for b in range(len(par.Betas)): # Loop over different beta values
            # get simulation data for this beta value
            simdata = getattr(sim, simvar)[:, b, :]

            # calculate mean over individual dimension for each beta
            mean_data = np.mean(simdata, axis=1)
            
            if simvar == 'discrete':
                mean_data *= 100  # Convert to percentage

            # plot mean data
            ax.plot(age, mean_data, label=r'$\beta=$' + str(beta_map[b]))
            

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
        
        if simvar == 'discrete':
            ax.set_ylabel('Percent')
            
        if simvar in ['p']:
            legend = ax.legend(frameon=True,prop={'size': 8})
            frame = legend.get_frame()
            frame.set_edgecolor('black')
            
    # remove the last, empty subplot if there is one
    if len(simvarlist) < np.prod(axs.shape):
        fig.delaxes(axs.flatten()[len(simvarlist)])

    # adjust layout for better visualization
    plt.tight_layout()

    if not os.path.exists("../plots"):
        os.makedirs("../plots")
    
    plt.savefig("../plots/two_asset_Lifecycle_sepBeta.pdf", bbox_inches='tight')
    
    plt.show()

    
def lifecycle_rand(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig, axs = plt.subplots(2, 4, figsize=(16,8))

    simvarlist = [('p','(A): Permanent Income, $p_t$'),
                  ('y','(B): Transitory Income, $y_t$'),
                  ('n','(C): Housing Stock, $n_t$'),
                  ('d','(D): Housing Stock Chosen, $d_t$'),
                  ('m','(E): Cash-on-Hand, $m_t$'),
                  ('c','(F): Consumption, $c_t$'),
                  ('a','(G): Savings, $a_t$'),
                  ('discrete','(H): Adjuster Share (in Percent)')]

    age = np.arange(par.T)
    for i,(simvar,simvarlatex) in enumerate(simvarlist):

        ax = axs[i // 4, i % 4]  # specify the current subplot

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

        # Change x-axis labels to show 20 years later
        xticks = np.arange(0, 61, 5)  # make sure it includes the last value you want to display
        ax.set_xticks(xticks)
        new_xticks = xticks + 20
        ax.set_xticklabels(new_xticks)
        ax.set_xlim(-3, 62)  # increase the upper limit of x-axis

        ax.grid(True)
        ax.set_xlabel('Age')

        if simvar == 'discrete':
            ax.set_ylabel('Percent')
            

        if simvar in ['p']:
            legend = ax.legend(frameon=True,prop={'size': 8})
            frame = legend.get_frame()
            frame.set_edgecolor('black')
            
    # remove the last, empty subplot if there is one
    if len(simvarlist) < np.prod(axs.shape):
        fig.delaxes(axs.flatten()[len(simvarlist)])

    plt.subplots_adjust(hspace=0.29, wspace=0.3)  

    if not os.path.exists("../plots"):
        os.makedirs("../plots")
    
    plt.savefig("../plots/two_asset_lifecycle_plot.pdf", bbox_inches='tight')

    plt.show()
