import setup
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.base import Bunch
import config as cfg
from load_data import GeneData
from all_fits import get_all_fits, iterate_fits
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from bootstrap import bootstrap
from scalers import LogScaler
from plots import save_figure

fontsize = 30

def sign(b):
    return '+' if b else '-'
    
def latex_label(theta,sigma):
    if (theta,sigma) == (False,False):
        return 'None'
    elif (theta,sigma) == (False,True):
        return r'$\sigma$'
    elif (theta,sigma) == (True,False):    
        return r'$\theta$'
    elif (theta,sigma) == (True,True):    
        return r'Both'

def plot_bar(variations, q=None, show_title=False):
    index = np.arange(len(variations))
    bar_width = 0.8
    
    mu = np.empty(len(variations))
    se = np.empty(len(variations))
    for i,v in enumerate(variations):
        scores = v.LOO_scores
        if q is None:
            f = np.mean
        else:
            def f(vals): return np.percentile(vals,q)
        mu[i],se[i] = bootstrap(scores, f)
    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.bar(
        index, 
        mu, 
        yerr=se,
        width=bar_width,
        color='b',
        error_kw = {'ecolor': '0.3', 'linewidth': 2},
    )  
    ttl = ''
    if q is None:
        ttl = 'Mean $R^2$ dependence on priors'
        ylabel = 'mean $R^2$'
    else:
        ttl = '{} percentile $R^2$ dependence on priors'.format(q)
        ylabel = '{} percentile $R^2$'.format(q)
    if show_title:
        ax.set_title(ttl, fontsize=fontsize)
    ax.set_xlabel('which priors', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks(index + bar_width/2)
    ax.set_xticklabels([latex_label(v.theta,v.sigma) for v in variations])
    ticks = [0,0.1,0.2,0.3]
    ax.set_yticks(ticks)
    ax.set_yticklabels(['{:g}'.format(t) for t in ticks])
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylim(0,0.3)

    return fig

def plot_pctiles(variations, min_q, show_title=False):
    fig = plt.figure()
    ax = fig.add_axes([0.15,0.12,0.8,0.8])
    q = list(np.arange(min_q,101))
    for v in variations:
        pctiles = np.percentile(v.LOO_scores, q)    
        ax.plot(q,pctiles, linewidth=2, alpha=0.8, label=latex_label(v.theta,v.sigma))
    ax.set_xlabel('$R^2$ percentile', fontsize=fontsize)
    ax.set_ylabel('$R^2$ score', fontsize=fontsize)
    if show_title:
        ax.title('Dependence of $R^2$ percentiles on priors', fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='best', frameon=False)
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig

def plot_theta_diff_scatter(show_title=False):
    yFitter = Fitter(Sigmoid(priors_name))
    nFitter = Fitter(Sigmoid())
    yFits = get_all_fits(data,yFitter)
    nFits = get_all_fits(data,nFitter)
    pairs = [(nFit.LOO_score,yFit.LOO_score) for nFit,yFit in iterate_fits(nFits,yFits)]
    diff_pairs = [(n,y-n) for n,y in pairs if n is not None and y is not None]
    n,d = zip(*diff_pairs)
    
    fig = plt.figure()
    ax = fig.add_axes([0.15,0.12,0.8,0.8])
    ax.scatter(n, d, alpha=0.5)
    xlims = ax.get_xlim()
    ax.plot(xlims,[0, 0],'k--')
    ax.set_xlim(*xlims)
    if show_title:
        ax.title(r'Improvement from prior on $\theta$ vs. baseline $R^2$', fontsize=fontsize)
    ax.set_xlabel(r'$R^2$(no priors)', fontsize=fontsize)
    ax.set_ylabel(r'$R^2$($\theta$) - $R^2$(no priors)', fontsize=fontsize) 
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig
    
def analyze_variant(theta,sigma):
    theta_priors = priors_name if theta else None
    sigma_prior = priors_name if sigma else None
    shape = Sigmoid(theta_priors)
    fitter = Fitter(shape,sigma_prior)
    fits = get_all_fits(data,fitter,allow_new_computation=False)
    LOO_scores = [f.LOO_score for f in iterate_fits(fits) if f.LOO_score is not None]
    mu,sem = bootstrap(LOO_scores, np.mean)
    return Bunch(
        theta = theta,
        sigma = sigma,
        LOO_scores = LOO_scores,
        mu = mu,
        sem = sem,
    )

cfg.verbosity = 1
age_scaler = LogScaler()
pathway = 'serotonin'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
priors_name = 'sigmoid_empirical_serotonin'
variations = [analyze_variant(t,s) for t,s in product([False,True],[False,True])]

fig = plot_bar(variations)
save_figure(fig,'RP/prior-variations-bar.png', under_results=True)
fig = plot_pctiles(variations, min_q=5)
save_figure(fig,'RP/prior-variations-percentiles.png', under_results=True)
fig = plot_theta_diff_scatter()
save_figure(fig,'RP/prior-variations-scatter.png', under_results=True)

#plot_bar(variations,q=95)
#plot_pctiles(variations, min_q=80)
