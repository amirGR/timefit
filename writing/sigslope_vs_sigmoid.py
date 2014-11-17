import setup
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from shapes.sigslope import Sigslope
from fitter import Fitter
from all_fits import get_all_fits, iterate_fits
from scalers import LogScaler
from plots import save_figure
from dev_stages import dev_stages

fontsize = 30

def plot_one_series(series, shape, theta, yrange=None, show_title=False):
    x = series.ages
    y = series.single_expression    
    xmin, xmax = min(x), max(x)
    xmin = max(xmin,-2)

    fig = plt.figure()
    ax = fig.add_axes([0.08,0.15,0.85,0.8])

    # plot the data points
    ax.plot(x,y, 'ks', markersize=8)
    if yrange is None:
        ymin, ymax = ax.get_ylim()
    else:
        ymin, ymax = yrange

    # mark birth time with a vertical line
    birth_age = series.age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    # draw the fit
    x_smooth,y_smooth = shape.high_res_preds(theta,x)
    ax.plot(x_smooth, y_smooth, '-', linewidth=3)
        
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    # title
    if show_title:
        ttl = '{}@{}, {} fit'.format(series.gene_name, series.region_name, shape)
        ax.set_title(ttl, fontsize=fontsize)

    # set the development stages as x labels
    ax.set_xlabel('age', fontsize=fontsize)
    stages = [stage.scaled(series.age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=fontsize, fontstretch='condensed', rotation=90)    

    # set y ticks (first and last only)
    ax.set_ylabel('expression level', fontsize=fontsize)
    ticks = ax.get_yticks()
    ticks = np.array([ticks[0], ticks[-1]])
    ax.set_yticks(ticks)
    ax.set_yticklabels(['{:g}'.format(t) for t in ticks], fontsize=fontsize)
    
    return fig

def plot_comparison_scatter(data, shape1, fits1, shape2, fits2):
    pairs = [(f1.LOO_score, f2.LOO_score) for f1,f2 in iterate_fits(fits1,fits2)]
    scores1,scores2 = zip(*pairs)
    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.scatter(scores1, scores2, alpha=0.3)
    ax.plot([-1, 1], [-1, 1],'k--')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ticks = [-1,1]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=fontsize)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=fontsize)
    ax.set_xlabel('$R^2$ for {}'.format(shape1), fontsize=fontsize)
    ax.set_ylabel('$R^2$ for {}'.format(shape2), fontsize=fontsize)
    return fig

def plot_comparison_bar(data, shapes, all_fits, threshold_percentile=None):
    nShapes = len(shapes)

    mu = np.empty(nShapes)
    se = np.empty(nShapes)
    all_scores = []
    for i,fits in enumerate(all_fits):
        scores = np.array([f.LOO_score for f in iterate_fits(fits, R2_threshold=-1)])
        if threshold_percentile is not None:
            threshold_score = np.percentile(scores, 50)
            scores = scores[scores > threshold_score]
        mu[i] = np.mean(scores)
        se[i] = scipy.stats.sem(scores)
        all_scores.append(scores)
    t, pval = scipy.stats.ttest_ind(all_scores[0], all_scores[1], equal_var=False)
    pval_one_side = pval/2
    print '*** t-test (non-equal variance, one sided) t={}, pval={:.3g}'.format(t,pval_one_side)
    
    # reorder by mean score
    idx = np.argsort(mu)[::-1]
    mu = mu[idx]
    se = se[idx]
    shapes = [shapes[i] for i in idx]

    index = np.arange(nShapes)
    bar_width = 0.8
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.bar(index, mu, yerr=se, width=bar_width, color='b', error_kw = {'ecolor': '0.3', 'linewidth': 2})  
    ax.set_xlabel('shape', fontsize=fontsize)
    ax.set_ylabel('Mean $R^2$', fontsize=fontsize)
    ax.set_xticks(index + bar_width/2)
    ax.set_xticklabels([s.cache_name() for s in shapes], fontsize=fontsize)
    yticks = [0, 0.1, 0.2, 0.3]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:g}'.format(t) for t in yticks], fontsize=fontsize)
    return fig

def plot_comparison_over_R2_score(data, shapes, all_fits, zoom=None, nbins=50):
    if zoom is None:
        zoom = (-1,1)
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    zoom_max = 0
    for shape,fits in zip(shapes,all_fits):
        scores = np.array([f.LOO_score for f in iterate_fits(fits)])
        scores[scores < -0.999] = -0.999
        h,bins = np.histogram(scores,bins=nbins,density=True)
        xpos = (bins[:-1] + bins[1:])/2
        zoom_data = h[(xpos>=zoom[0]) & (xpos<=zoom[1])]
        zoom_max = max(max(zoom_data),zoom_max)
        ax.plot(xpos,h, linewidth=3, label=shape.cache_name())        
    ax.set_xlim(*zoom)
    ax.set_ylim(0,zoom_max*1.1)
    ax.legend(loc='best', fontsize=fontsize, frameon=False)
    ax.set_xlabel('test $R^2$ score', fontsize=fontsize)
    ax.set_ylabel("probability density", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()
data = GeneData.load('both').scale_ages(age_scaler)

sigmoid = Sigmoid(priors='sigmoid_wide')
sigslope = Sigslope(priors='sigslope80')
shapes = [sigmoid, sigslope]
fitters = [Fitter(shape, sigma_prior='normal') for shape in shapes]

#####################################################
# Example fits
#####################################################
GRs = [
    ('ABHD4','STC', (5, 8)), 
]
for g,r,yrange in GRs:
    for fitter in fitters:
        print 'Doing {}@{}...'.format(g,r)
        series = data.get_one_series(g,r)
        theta,_,_,_ = fitter.fit(series.ages, series.single_expression)
        fig = plot_one_series(series, fitter.shape, theta, yrange)
        save_figure(fig,'RP/fit-examples-{}-{}-{}.png'.format(fitter.shape.cache_name(), g,r), under_results=True)


#####################################################
# Comparison for whole pathway
#####################################################
pathway = '17full'
data = data.restrict_pathway(pathway)
fits = [get_all_fits(data,fitter,allow_new_computation=False) for fitter in fitters]


fig = plot_comparison_bar(data, shapes, fits)
save_figure(fig,'RP/sigslope-comparison-bar-{}.png'.format(data.pathway), under_results=True)

fig = plot_comparison_over_R2_score(data, shapes, fits)
save_figure(fig,'RP/sigslope-comparison-vs-R2-{}.png'.format(data.pathway), under_results=True)

fig = plot_comparison_scatter(data,shapes[0],fits[0],shapes[1],fits[1])
save_figure(fig,'RP/scatter-{}-{}-{}.png'.format(shapes[0],shapes[1],pathway), under_results=True)

plt.close('all')