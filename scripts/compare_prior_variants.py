import setup
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.base import Bunch
import config as cfg
from load_data import load_data
from all_fits import get_all_fits
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from shapes.poly import Poly
from bootstrap import bootstrap

from utils.misc import disable_all_warnings
disable_all_warnings()

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

def plot_bar(variations, q=None):
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
    
    plt.figure()
    plt.bar(
        index, 
        mu, 
        yerr=se,
        width=bar_width,
        color='b',
        error_kw = {'ecolor': '0.3', 'linewidth': 2},
    )  
    ttl = ''
    if q is None:
        ttl = 'Mean R2 dependence on priors'
        ylabel = 'Mean R2'
    else:
        ttl = '{} percentile R2 dependence on priors'.format(q)
        ylabel = '{} percentile R2'.format(q)
    plt.title(ttl, fontsize=cfg.fontsize)
    plt.xlabel('Which Priors', fontsize=cfg.fontsize)
    plt.ylabel(ylabel, fontsize=cfg.fontsize)
    plt.xticks(index + bar_width/2, [latex_label(v.theta,v.sigma) for v in variations], fontsize=cfg.fontsize)

def plot_pctiles(variations, min_q):
    plt.figure()
    q = list(np.arange(min_q,101))
    for v in variations:
        pctiles = np.percentile(v.LOO_scores, q)    
        plt.plot(q,pctiles, linewidth=2, alpha=0.8, label=latex_label(v.theta,v.sigma))
    plt.xlabel('Percentile', fontsize=cfg.fontsize)
    plt.ylabel('R2 score', fontsize=cfg.fontsize)
    plt.title('Dependence of R2 percentiles on priors', fontsize=cfg.fontsize)
    plt.legend(fontsize=cfg.fontsize, loc='upper left')

def plot_theta_diff_scatter(shape):
    yFitter = Fitter(shape,True,False)
    nFitter = Fitter(shape,False,False)
    yFits = get_all_fits(data,yFitter)
    nFits = get_all_fits(data,nFitter)
    pairs = [(nFits[k].LOO_score, yFits[k].LOO_score) for k in yFits.iterkeys()]
    diff_pairs = [(n,y-n) for n,y in pairs if n is not None and y is not None]
    n,d = zip(*diff_pairs)
    
    plt.figure()
    plt.scatter(n, d, alpha=0.5)
    xlims = plt.xlim()
    plt.plot(xlims,[0, 0],'k--')
    plt.xlim(*xlims)
    plt.title(r'Improvement from prior on $\theta$ vs. baseline R2', fontsize=cfg.fontsize)
    plt.xlabel(r'R2(No priors)', fontsize=cfg.fontsize)
    plt.ylabel(r'R2($\theta$) - R2(No priors)', fontsize=cfg.fontsize)    
    
data = load_data()
def analyze_variant(shape,theta,sigma):
    fitter = Fitter(shape,theta,sigma)
    fits = get_all_fits(data,fitter)
    LOO_scores = [f.LOO_score for f in fits.itervalues() if f.LOO_score is not None]
    mu,sem = bootstrap(LOO_scores, np.mean)
    return Bunch(
        theta = theta,
        sigma = sigma,
        LOO_scores = LOO_scores,
        mu = mu,
        sem = sem,
    )

shape = Sigmoid()
variations = [analyze_variant(shape,t,s) for t,s in product([False,True],[False,True])]
plot_bar(variations)
plot_bar(variations,q=95)
plot_pctiles(variations, min_q=5)
plot_pctiles(variations, min_q=80)
plot_theta_diff_scatter(shape)
