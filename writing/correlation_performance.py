import setup
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from utils.misc import disable_all_warnings
from all_fits import get_all_fits
from fit_score import loo_score
from plots import save_figure, plot_gene_correlations_single_region
from load_data import GeneData
from shapes.sigslope import Sigslope
from fitter import Fitter
from scalers import LogScaler
import scipy.stats

fontsize = 30

def plot_comparison_bar(tuples, several_levels=False):
    levels = zip(*tuples)
    if not several_levels:
        levels = levels[:2]
        
    scores_no_correlations, scores_with_correlations = levels[0], levels[1]
    _, pval = scipy.stats.wilcoxon(scores_no_correlations, scores_with_correlations)
    pval = pval/2  # one sided p-value
    print '*** wilcoxon signed rank p-value (one sided) = {:.3g}'.format(pval)
    
    mu = np.empty(len(levels))
    se = np.empty(len(levels))
    for i,scores in enumerate(levels):
        mu[i] = np.mean(scores)
        se[i] = scipy.stats.sem(scores)
    
    index = np.arange(len(levels))
    bar_width = 0.8
    fig = plt.figure()
    if several_levels:
        ax = fig.add_axes([0.12,0.15,0.8,0.8])
    else:
        ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.bar(index, mu, yerr=se, width=bar_width, color='b', error_kw = {'ecolor': '0.3', 'linewidth': 2})  
    ax.set_ylabel('Mean $R^2$', fontsize=fontsize)
    ax.set_xticks(index + bar_width/2)
    if several_levels:
        labels = range(len(levels))
        labels[0] = 'assuming\nindependence'
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_xlabel('Number of optimization iterations', fontsize=fontsize)
    else:
        ax.set_xticklabels(['assuming\nindependence', 'using\ncorrelations'], fontsize=fontsize)
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
    y = series.expression
    
    R2_tuples = {}
    for i,g in enumerate(series.gene_names):
        fit = ds_fits[(g,region)]
        y_real = y[:,i]
        y_basic = fit.LOO_predictions
        basic_R2 = loo_score(y_real,y_basic)
        scores = [basic_R2]
        for level in fit.with_correlations:
            y_multi_gene = level.LOO_predictions[series.original_inds]
            R2 = loo_score(y_real,y_multi_gene)
            scores.append(R2)
        if (np.array(scores) < -1).any():
            continue
        R2_tuples[(g,region)] = tuple(scores)
        
    region_fits = ds_fits[(None,region)]
    correlations = region_fits[0].correlations # get correlations after one optimization iteration
    return R2_tuples, correlations

def print_best_improvements(dct_scores, level=1):
    min_score = 0.25
    def one_diff(gr,score_tuple):
        g,r = gr
        s_basic = score_tuple[0]
        s_with = score_tuple[level]
        d = s_with - s_basic
        if s_basic > min_score and s_with > min_score:
            return d,g,r
        else:
            return None
    diffs = [one_diff(gr,scores) for gr,scores in dct_scores.iteritems()]
    diffs = filter(None, diffs)
    diffs.sort(reverse=True)
    for diff in diffs[:10]:
        print diff

def analyze_pathway(pathway, data, fitter, fits, html_only=False):
    print 80 * '='
    print 'Analyzing pathway {}'.format(pathway)
    print 80 * '='
    dct_tuples = {}
    for region in data.region_names:
        dct_region_tuples, region_correlations = analyze_one_region(data, fitter, fits, region)
        fig = plot_gene_correlations_single_region(region_correlations, region, data.gene_names)
        filename = join('RP','correlation-heat-map-{}-{}.png'.format(region,pathway))
        save_figure(fig, filename, b_close=True, under_results=True)
        dct_tuples.update(dct_region_tuples)
        
    print_best_improvements(dct_tuples)
    
    tuples = dct_tuples.values()
    pairs = [(x[0],x[1]) for x in tuples]
    fig = plot_comparison_scatter(pairs,pathway)
    filename = join('RP','correlation-diff-scatter-{}.png'.format(pathway))
    save_figure(fig, filename, b_close=True, under_results=True)

    fig = plot_comparison_bar(tuples)
    filename = join('RP','correlation-diff-bar-{}.png'.format(pathway))
    save_figure(fig, filename, b_close=True, under_results=True)
    fig = plot_comparison_bar(tuples, several_levels=True)
    filename = join('RP','correlation-diff-bar-several-levels-{}.png'.format(pathway))
    save_figure(fig, filename, b_close=True, under_results=True)

disable_all_warnings()
cfg.verbosity = 1
age_scaler = LogScaler()
shape = Sigslope('sigslope80')
fitter = Fitter(shape, sigma_prior='normal')

pathways = ['cannabinoids', 'serotonin']
for pathway in pathways:
    data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
    fits = get_all_fits(data, fitter, n_correlation_iterations=4, allow_new_computation=False)
    analyze_pathway(pathway, data, fitter, fits)

