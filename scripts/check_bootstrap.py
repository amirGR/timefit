import setup
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from utils.misc import disable_all_warnings
from load_data import GeneData
from scalers import LogScaler
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits
from sigmoid_change_distribution import add_change_distributions
from plots import save_figure
from dev_stages import dev_stages

def plot_bootstrap_histograms(data, fit, n_bins, n_samples):
    from_age, to_age = data.age_range
    bin_edges, bin_size = np.linspace(from_age, to_age, n_bins+1, retstep=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    def calc_changes(theta):
        a,h,mu,w = theta
        edge_vals = shape.f(theta,bin_edges)
        changes = np.abs(edge_vals[1:] - edge_vals[:-1])
        changes /= abs(h) # ignore change magnitude per gene - take only distribution of change times
        return changes

    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    for i in xrange(n_samples):
        changes = calc_changes(fit.theta_samples[:,i])
        label = 'bootstrap sample' if i==0 else None
        ax.plot(bin_centers, changes, 'g--', linewidth=2, label=label)

    # plot change distribution for the original fit
    changes = calc_changes(fit.theta)
    ax.plot(bin_centers, changes, 'r-', linewidth=3, label='original fit')

    # plot average of all samples
    changes = fit.change_histogram.change_vals
    ax.plot(bin_centers, changes, 'b-', linewidth=3 ,label='average of samples')
    ax.set_ylabel('expression change magnitude', fontsize=cfg.fontsize)

    ax.legend(loc='best', fontsize=cfg.fontsize, frameon=False)

    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=cfg.fontsize, fontstretch='condensed', rotation=90)    
    yticks = ax.get_yticks()
    yticks = []#yticks[0], yticks[-1]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.1g}'.format(t) for t in yticks], fontsize=cfg.fontsize)
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    return fig

def plot_bootstrap_fits(data, fit, n_bins, n_samples):
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    for i in xrange(n_samples):
        # compute change distribution for the bootstrap sample
        t = fit.theta_samples[:,i]
        x_smooth, y_smooth = shape.high_res_preds(t, np.array(data.age_range))
        label = 'bootstrap sample' if i==0 else None
        ax.plot(x_smooth, y_smooth, 'g--', linewidth=2, label=label)

    # plot global fit
    x_smooth, y_smooth = shape.high_res_preds(fit.theta, np.array(data.age_range))
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='original fit')
    ax.set_ylabel('expression change magnitude', fontsize=cfg.fontsize)

    ax.legend(loc='best', fontsize=cfg.fontsize, frameon=False)

    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=cfg.fontsize, fontstretch='condensed', rotation=90)    
    yticks = ax.get_yticks()
    yticks = []#yticks[0], yticks[-1]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.1g}'.format(t) for t in yticks], fontsize=cfg.fontsize)
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    return fig

pathway = 'serotonin'
g = 'HTR1E'
r = 'VFC'
n_bins = 50
n_samples = 10

disable_all_warnings()   
cfg.verbosity = 1
age_scaler = LogScaler()

data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')
fits = get_all_fits(data, fitter, allow_new_computation=False)
fits = add_change_distributions(data, fitter, fits, n_bins=n_bins)
ds_name = data.region_to_dataset()[r]
fit = fits[ds_name][(g,r)]

fig = plot_bootstrap_fits(data, fit, n_bins=n_bins, n_samples=n_samples)
save_figure(fig, 'bootstrap-fits-{}-{}.png'.format(g,r), under_results=True)
fig = plot_bootstrap_histograms(data, fit, n_bins=n_bins, n_samples=n_samples)
save_figure(fig, 'bootstrap-transition-distribution-{}-{}.png'.format(g,r), under_results=True)