from __future__ import print_function
import setup
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits, iterate_fits
from scalers import LogScaler
from plots import save_figure
from project_dirs import results_dir

fontsize = 30

def plot_score_distribution(R2_original, R2_shuffled):
    low,high = -1, 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    def do_hist(R2):
        counts,bin_edges = np.histogram(R2,50,range=(low,high))
        probs = counts / float(sum(counts))
        width = bin_edges[1] - bin_edges[0]
        return bin_edges[:-1],probs,width
    
    pos, probs, width = do_hist(R2_original)
    ax.bar(pos, probs, width=width, color='b', label='Original')

    pos, probs, width = do_hist(R2_shuffled)
    ax.bar(pos, probs, width=width, color='g', alpha=0.5, label='Shuffled')

    ax.legend(loc='best', fontsize=fontsize, frameon=False)
    ax.set_xlabel('test set $R^2$', fontsize=fontsize)
    ax.set_ylabel('probability', fontsize=fontsize)   
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig

def plot_z_scores(z_scores):
    low,high = -1, 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    counts,bin_edges = np.histogram(z_scores,50)
    probs = counts / float(sum(counts))
    width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_edges[:-1], probs, width=width, color='b')

    ax.set_xlabel('z score', fontsize=fontsize)
    ax.set_ylabel('probability', fontsize=fontsize)   
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()
pathway = '17full'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
data_shuffled = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler).shuffle()

shape = Sigmoid('sigmoid_wide')
fitter = Fitter(shape,sigma_prior='normal')
fits = get_all_fits(data,fitter,allow_new_computation=False)
fits_shuffled = get_all_fits(data_shuffled,fitter,allow_new_computation=False)
R2_pairs = [(fit.LOO_score,fit2.LOO_score) for fit,fit2 in iterate_fits(fits,fits_shuffled)]
R2 = np.array([r for r,r_shuffled in R2_pairs])
R2_shuffled = np.array([r_shuffled for r,r_shuffled in R2_pairs])

name = '{}-{}'.format(data.pathway,shape.cache_name())
fig = plot_score_distribution(R2,R2_shuffled)
save_figure(fig,'RP/R2-distribution-{}.png'.format(name), under_results=True, b_close=True)

mu_shuffled = np.mean(R2_shuffled)
std_shuffled = np.std(R2_shuffled)
z_scores = (R2-mu_shuffled)/std_shuffled
fig = plot_z_scores(z_scores)
save_figure(fig,'RP/R2-z-scores-{}.png'.format(name), under_results=True, b_close=True)

T, signed_rank_p_value = wilcoxon(R2, R2_shuffled)
maxShuffled = R2_shuffled.max()
nAbove = np.count_nonzero(R2 > maxShuffled)
nTotal = len(R2)
pct = 100.0 * nAbove/nTotal
filename = join(results_dir(),'RP/R2-distribution-{}.txt'.format(name))
with open(filename,'w') as f:
    print('shuffled = {:.2g} +/- {:.2g}'.format(mu_shuffled,std_shuffled), file=f)
    print('maximal shuffled score: {:.2g}'.format(maxShuffled), file=f)
    print('{:.2g}% ({}/{}) of scores are above maximal shuffled score'.format(pct,nAbove,nTotal), file=f)
    for z_threshold in [1,2,3,4,5]:
        nAbove = np.count_nonzero(z_scores > z_threshold)
        pct = 100.0 * nAbove/nTotal
        print('{:.2g}% ({}/{}) of z-scores are above {}'.format(pct,nAbove,nTotal,z_threshold), file=f)
    print('wilxocon signed-rank p-value = {:.2g}'.format(signed_rank_p_value), file=f)