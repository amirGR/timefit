import setup
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from utils.misc import disable_all_warnings
from all_fits import get_all_fits
from fit_score import loo_score
from plots import save_figure
from load_data import GeneData
from shapes.sigslope import Sigslope
from fitter import Fitter
from scalers import LogScaler
import scipy.stats

fontsize = 30

def plot_comparison_bar(scores_no_correlations, scores_with_correlations):
    _, pval = scipy.stats.wilcoxon(scores_no_correlations, scores_with_correlations)
    pval = pval/2  # one sided p-value
    print '*** wilcoxon signed rank p-value (one sided) = {:.3g}'.format(pval)
    
    mu = np.empty(2)
    se = np.empty(2)
    all_scores = [scores_no_correlations, scores_with_correlations]
    for i,scores in enumerate(all_scores):
        mu[i] = np.mean(scores)
        se[i] = scipy.stats.sem(scores)
    
    index = np.arange(2)
    bar_width = 0.8
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.bar(index, mu, yerr=se, width=bar_width, color='b', error_kw = {'ecolor': '0.3', 'linewidth': 2})  
    ax.set_ylabel('Mean $R^2$', fontsize=fontsize)
    ax.set_xticks(index + bar_width/2)
    ax.set_xticklabels(['Basic fit', 'Using correlations'], fontsize=fontsize)
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:g}'.format(t) for t in yticks], fontsize=fontsize)
    return fig

def plot_comparison_scatter(R2_pairs, pathway):
    basic = np.array([b for b,m in R2_pairs])
    multi = np.array([m for b,m in R2_pairs])
    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.scatter(basic, multi, alpha=0.8, label='data')
    ax.plot(np.mean(basic), np.mean(multi), 'rx', markersize=12, markeredgewidth=3, label='mean')
    ax.plot([-1, 1], [-1, 1],'k--')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ticks = [-1,1]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=fontsize)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=fontsize)
    ax.set_xlabel('$R^2$ for single gene fits', fontsize=fontsize)
    ax.set_ylabel('$R^2$ using correlations', fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False, loc='upper left')
    return fig

def analyze_one_region(data, fitter, fits, region):
    print 'Analyzing region {}...'.format(region)
    series = data.get_several_series(data.gene_names,region)
    ds_fits = fits[data.get_dataset_for_region(region)]
    
    def cache(iy,ix):
        g = series.gene_names[iy]
        fit = ds_fits[(g,region)]
        if ix is None:
            return fit.theta
        else:
            theta,sigma = fit.LOO_fits[ix]
            return theta    
    x = series.ages
    y = series.expression
    multi_gene_preds,_,_ = fitter.fit_multiple_series_with_cache(x,y,cache)
    
    R2_pairs = {}
    for i,g in enumerate(series.gene_names):
        y_real = y[:,i]
        y_basic = ds_fits[(g,region)].LOO_predictions
        y_multi_gene = multi_gene_preds[:,i]        
        basic_R2 = loo_score(y_real,y_basic)
        multi_gene_R2 = loo_score(y_real,y_multi_gene)
        if basic_R2 < -1 or multi_gene_R2 < -1:
            continue
        R2_pairs[(g,region)] = (basic_R2, multi_gene_R2)
    return R2_pairs

def print_best_improvements(dct_pairs):
    min_score = 0.25
    def one_diff(gr,score_pair):
        g,r = gr
        s_basic, s_with = score_pair
        d = s_with - s_basic
        if s_basic > min_score and s_with > min_score:
            return d,g,r
        else:
            return None
    diffs = [one_diff(gr,pair) for gr,pair in dct_pairs.iteritems()]
    diffs = filter(None, diffs)
    diffs.sort(reverse=True)
    for diff in diffs[:10]:
        print diff

def analyze_pathway(pathway, data, fitter, fits, html_only=False):
    print 80 * '='
    print 'Analyzing pathway {}'.format(pathway)
    print 80 * '='
    dct_pairs = {}
    for region in data.region_names:
        dct_pairs.update( analyze_one_region(data, fitter, fits, region) )
        
    print_best_improvements(dct_pairs)
    
    pairs = dct_pairs.values()
    fig = plot_comparison_scatter(pairs,pathway)
    filename = join('RP','correlation-diff-scatter-{}.png'.format(pathway))
    save_figure(fig, filename, b_close=True, under_results=True)

    scores_no_correlations, scores_with_correlations = zip(*pairs)
    fig = plot_comparison_bar(scores_no_correlations, scores_with_correlations)
    filename = join('RP','correlation-diff-bar-{}.png'.format(pathway))
    save_figure(fig, filename, b_close=True, under_results=True)

disable_all_warnings()
cfg.verbosity = 1
age_scaler = LogScaler()
shape = Sigslope('sigslope80')
fitter = Fitter(shape, sigma_prior='normal')

pathways = ['cannabinoids', 'serotonin']
for pathway in pathways:
    data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
    fits = get_all_fits(data, fitter, allow_new_computation=False)
    analyze_pathway(pathway, data, fitter, fits)

