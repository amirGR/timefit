import setup
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits, restrict_genes
from scalers import LogScaler
from dev_stages import dev_stages
from plots import save_figure
from sigmoid_change_distribution import add_change_distributions, aggregate_change_distribution

fontsize = 30

def plot_onset_times(data, fitter, fits, dct_pathways, R2_threshold):
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])

    for i,(pathway_name, genes) in enumerate(sorted(dct_pathways.items())):
        pathway_fits = restrict_genes(fits,genes)
        bin_centers, weights, n_fits = aggregate_change_distribution(pathway_fits, R2_threshold=R2_threshold)
        linestyles = ['-', '--', '-.']
        style = linestyles[int(i/7)]
        label = '{} ({} fits)'.format(pathway_name, n_fits)
        ax.plot(bin_centers, weights, style, linewidth=3, label=label)
    ax.legend(loc='best', fontsize=18, frameon=False)
    ax.set_ylabel('expression change magnitude', fontsize=fontsize)

    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=fontsize, fontstretch='condensed', rotation=90)    
    yticks = ax.get_yticks()
    yticks = []#yticks[0], yticks[-1]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.1g}'.format(t) for t in yticks], fontsize=fontsize)
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    return fig

cfg.verbosity = 1
age_scaler = LogScaler()

pathway = 'serotonin' # '17full'
R2_threshold = 0.5

data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')
fits = get_all_fits(data, fitter, allow_new_computation=False)
fits = add_change_distributions(data, fitter, fits)

fig = plot_onset_times(data, fitter, fits, {pathway:data.gene_names}, R2_threshold)
save_figure(fig, 'change-distribution-{}.png'.format(pathway), under_results=True)
