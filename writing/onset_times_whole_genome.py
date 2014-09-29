import setup
from os.path import join, isfile, dirname
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits, iterate_fits
from scalers import LogScaler
from dev_stages import dev_stages
from plots import save_figure
from project_dirs import cache_dir, fit_results_relative_path
from utils.misc import ensure_dir

fontsize = 30

def compute_change_distribution(shape, thetas, from_age, to_age, n_bins=50, b_normalize=True):
    assert shape.cache_name() == 'sigmoid' # we use parameter h explicitly
    bin_edges, bin_size = np.linspace(from_age, to_age, n_bins+1, retstep=True)
    change_vals = np.zeros(n_bins)
    for t in thetas:
        a,h,mu,w = t
        edge_vals = shape.f(t,bin_edges)
        changes = np.abs(edge_vals[1:] - edge_vals[:-1])
        # ignore change magnitude per gene - take only distribution of change times
        change_vals += changes / abs(h)
    if b_normalize:
        change_vals /= sum(change_vals)
    return bin_edges, change_vals

def get_onset_times(data, fitter, R2_threshold, b_force=False):
    filename = join(cache_dir(),fit_results_relative_path(data,fitter) + '.pkl')
    if isfile(filename):
        print 'Loading onset distribution from {}'.format(filename)
        with open(filename) as f:
            bin_edges, change_vals = pickle.load(f)
    else:
        print 'Computing...'
        fits = get_all_fits(data, fitter)        
        thetas = [fit.theta for fit in iterate_fits(fits, R2_threshold=R2_threshold)]
        stages = [stage.scaled(age_scaler) for stage in dev_stages]
        low = min(stage.from_age for stage in stages)
        high = max(stage.to_age for stage in stages) 
        bin_edges, change_vals = compute_change_distribution(fitter.shape, thetas, low, high, n_bins=50)    

        print 'Saving result to {}'.format(filename)
        ensure_dir(dirname(filename))   
        with open(filename,'w') as f:
            pickle.dump((bin_edges,change_vals),f)
    return bin_edges, change_vals
    
def plot_onset_times(bin_edges, change_vals):
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    ax.plot(bin_centers, change_vals, linewidth=3)
    ax.legend(loc='best', fontsize=18, frameon=False)
    ax.set_ylabel('expression change magnitude', fontsize=fontsize)

    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=fontsize, fontstretch='condensed', rotation=90)    
    yticks = ax.get_yticks()
    yticks = [yticks[0], yticks[-1]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:g}'.format(t) for t in yticks], fontsize=fontsize)
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    return fig

cfg.verbosity = 1
age_scaler = LogScaler()

pathway = 'all'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')
R2_threshold = 0.5

bin_edges, change_vals = get_onset_times(data, fitter, R2_threshold)
fig = plot_onset_times(bin_edges, change_vals)
filename = 'RP/change-distribution-{}.png'.format(pathway)
save_figure(fig, filename, under_results=True)
plt.close('all')
