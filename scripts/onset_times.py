import setup
from os.path import join
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
from utils.misc import add_main_axes

cfg.verbosity = 1
age_scaler = LogScaler()

def unique_genes_only(dct_pathways):
    res = {}
    def count(dct,g):
        return sum(1 for pathway_genes in dct.itervalues() if g in pathway_genes)
    for pathway_name,genes in dct_pathways.iteritems():
        dct_counts = {g:count(dct_pathways,g) for g in genes}
        unique_genes = {g for g,c in dct_counts.iteritems() if c == 1}
        res[pathway_name] = unique_genes
    return res
    
def compute_change_distribution(shape, thetas, from_age, to_age, n_bins=50, b_normalize=True):
    bin_edges, bin_size = np.linspace(from_age, to_age, n_bins+1, retstep=True)
    change_vals = np.zeros(n_bins)
    for g,r,t in thetas:
        edge_vals = shape.f(t,bin_edges)
        changes = edge_vals[1:] - edge_vals[:-1]
        yrange = edge_vals[-1] - edge_vals[0] 
        # ignore change magnitude per gene - take only distribution of change times
        if yrange < 0.1:
            if cfg.verbosity >= 2:
                print 'Ignoring {}@{}. Changes are too small'.format(g,r)
            continue
        change_vals += changes / yrange
    if b_normalize:
        change_vals /= sum(change_vals)
    return bin_edges, change_vals
    
def plot_onset_times(shape, fits, dct_pathways, R2_threshold, b_unique):
    fig = plt.figure()
    ax = add_main_axes(fig)

    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    low = min(stage.from_age for stage in stages)
    high = max(stage.to_age for stage in stages) 

    for i,(pathway_name, genes) in enumerate(sorted(dct_pathways.items())):
        pathway_fits = restrict_genes(fits,genes)    
        thetas = [(g,r,fit.theta) for dsname,g,r,fit in iterate_fits(pathway_fits, R2_threshold=R2_threshold, return_keys=True)]
        if not thetas:
            print 'Skipping {}. No fits left'.format(pathway_name)
            continue

        bin_edges, change_vals = compute_change_distribution(shape, thetas, low, high, n_bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        linestyles = ['-', '--', '-.']
        style = linestyles[int(i/7)]
        ax.plot(bin_centers, change_vals, style, linewidth=3, label=pathway_name)
    ax.legend(loc='best', fontsize=18, frameon=False)

    str_unique = ' (unique genes)' if b_unique else ''
    ttl = 'Distribution of expression changes{}\n(R2 threshold={})'.format(str_unique, R2_threshold)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_ylabel('expression change magnitude', fontsize=cfg.fontsize)

    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=cfg.xtick_fontsize, fontstretch='condensed', rotation=90)    
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax-1E-4], '--', color='0.85')

    return fig
    
def main():
    cfg.verbosity = 1
    
    pathway = '17full'
    data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
    shape = Sigmoid(priors='sigmoid_wide')
    fitter = Fitter(shape, sigma_prior='normal')
    fits = get_all_fits(data, fitter)
    
    R2_threshold = 0.5
    for b_unique in [False,True]:
        dct_pathways = load_17_pathways_breakdown()
        if b_unique:
            dct_pathways = unique_genes_only(dct_pathways)
        dct_pathways['17 pathways'] = None
        fig = plot_onset_times(shape, fits, dct_pathways, R2_threshold, b_unique)
        if b_unique:
            filename = 'change-distributions (unique).png'
        else:
            filename = 'change-distributions.png'
        save_figure(fig, filename, under_results=True)

if __name__ == '__main__':
    main()