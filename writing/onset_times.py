import setup
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData, load_17_pathways_breakdown
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits, restrict_genes, iterate_fits
from scalers import LogScaler
from dev_stages import dev_stages
from plots import save_figure
from project_dirs import cache_dir, fit_results_relative_path

fontsize = 30

def get_change_distribution_for_whole_genome(all_data, fitter):
    # NOTE: the distribution for all genes should be precomputed by running onset_times_whole_genome.py
    filename = join(cache_dir(),fit_results_relative_path(all_data,fitter) + '.pkl')
    print 'Loading whole genome onset distribution from {}'.format(filename)
    with open(filename) as f:
        bin_edges, change_vals = pickle.load(f)
    return bin_edges, change_vals
    
def compute_change_distribution(shape, thetas, from_age, to_age, n_bins=50, b_normalize=True):
    assert shape.cache_name() == 'sigmoid' # we use parameter h explicitly
    bin_edges, bin_size = np.linspace(from_age, to_age, n_bins+1, retstep=True)
    change_vals = np.zeros(n_bins)
    for g,r,t in thetas:
        a,h,mu,w = t
        edge_vals = shape.f(t,bin_edges)
        changes = np.abs(edge_vals[1:] - edge_vals[:-1])
        # ignore change magnitude per gene - take only distribution of change times
        change_vals += changes / abs(h)
    if b_normalize:
        change_vals /= sum(change_vals)
    return bin_edges, change_vals
    
def plot_onset_times(all_data, data, fitter, fits, dct_pathways, R2_threshold, b_unique):    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])

    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    low = min(stage.from_age for stage in stages)
    high = max(stage.to_age for stage in stages) 

    n_fits = sum(len(ds.gene_names) * len(ds.region_names) for ds in all_data.datasets)
    bin_edges, change_vals = get_change_distribution_for_whole_genome(all_data,fitter)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    ax.plot(bin_centers, change_vals, linewidth=5, label='whole genome ({} fits)'.format(n_fits))

    for i,(pathway_name, genes) in enumerate(sorted(dct_pathways.items())):
        pathway_fits = restrict_genes(fits,genes)    
        thetas = [(g,r,fit.theta) for dsname,g,r,fit in iterate_fits(pathway_fits, R2_threshold=R2_threshold, return_keys=True)]
        if not thetas:
            print 'Skipping {}. No fits left'.format(pathway_name)
            continue

        bin_edges, change_vals = compute_change_distribution(fitter.shape, thetas, low, high, n_bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        linestyles = ['-', '--', '-.']
        style = linestyles[int(i/7)]
        label = '{} ({} fits)'.format(pathway_name,len(thetas))
        ax.plot(bin_centers, change_vals, style, linewidth=3, label=label)
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

all_data = GeneData.load('both').scale_ages(age_scaler)
pathway = '17full'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')
fits = get_all_fits(data, fitter)

R2_threshold = 0.5
for b_unique in [False,True]:
    dct_pathways = load_17_pathways_breakdown(b_unique)
    dct_pathways['17 pathways'] = None
    for name,genes in dct_pathways.iteritems():
        fig = plot_onset_times(all_data, data, fitter, fits, {name:genes}, R2_threshold, b_unique)
        str_dir = 'unique' if b_unique else 'overlapping'
        str_unique = ' (unique)' if b_unique else ''
        filename = 'RP/{}/change-distributions-{}{}.png'.format(str_dir,name,str_unique)
        save_figure(fig, filename, under_results=True)

    # selected plots
    lst_pathways = ['17 pathways', 'Amphetamine addiction', 'Cholinergic synapse', 'Cocaine addiction', 'Glutamatergic synapse']
    dct_pathways = {k:dct_pathways[k] for k in lst_pathways}
    fig = plot_onset_times(all_data, data, fitter, fits, dct_pathways, R2_threshold, b_unique)
    str_dir = 'unique' if b_unique else 'overlapping'
    str_unique = ' (unique)' if b_unique else ''
    filename = 'RP/{}/selected-change-distributions{}.png'.format(str_dir,str_unique)
    save_figure(fig, filename, under_results=True)

plt.close('all')
