import setup
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from sklearn.datasets.base import Bunch
from load_data import SeveralGenesOneRegion
from plots import plot_series
from shapes.sigslope import Sigslope
from fitter import Fitter
from fit_score import loo_score
from utils.misc import init_array

cfg.fontsize = 18
cfg.xtick_fontsize = 18
cfg.ytick_fontsize = 18

def get_fitter():
    shape = Sigslope(priors='sigslope80')
    fitter = Fitter(shape, sigma_prior='normal')
    return fitter    

def get_series():
    n = 10
    rng = np.random.RandomState(cfg.random_seed)
    x = np.linspace(0,100,n) + rng.normal(0,0.1,size=n)
    x.sort()
    
    shape = Sigslope()
    t1 = (-1,40,50,0.2)
    y1 = shape.f(t1,x)
    t2 = (1,30,25,0.5)
    y2 = shape.f(t2,x)
    c = -0.95
    sigma = 100*np.array([[1, c], [c, 1]])
    noise = rng.multivariate_normal([0,0],sigma,n)
    y = np.c_[y1,y2] + noise
     
    return SeveralGenesOneRegion(
        expression = y, 
        ages = x, 
        gene_names = ['A','B'], 
        region_name = 'THERE', 
        original_inds = np.arange(n), 
        age_scaler = None,
    )

def plot_comparison_scatter(R2_pairs, region_name):
    basic_scores, multi_gene_scores = zip(*R2_pairs)
    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.scatter(basic_scores, multi_gene_scores, alpha=0.3)
    ax.plot([-1, 1], [-1, 1],'k--')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ticks = [-1,1]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=cfg.fontsize)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=cfg.fontsize)
    ax.set_xlabel('$R^2$ for single gene fits', fontsize=cfg.fontsize)
    ax.set_ylabel('multi gene $R^2$', fontsize=cfg.fontsize)
    ax.set_title('$R^2$ gain from gene correlations - region {}'.format(region_name), fontsize=cfg.fontsize)
    return fig

if __name__ == '__main__':
    cfg.verbosity = 2
    fitter = get_fitter()
    series = get_series()
    x = series.ages
    y = series.expression
    
    ##############################################################
    # check fit_multiple_series_with_cache
    ##############################################################
    fits = []
    for i,g in enumerate(series.gene_names):
        print 'Fitting series {}...'.format(i+1)
        theta, sigma, LOO_predictions,_ = fitter.fit(x,y[:,i],loo=True)
        fit = Bunch(
            theta = theta,
            LOO_predictions = LOO_predictions,
        )
        fits.append(fit)
            
    print 'Fitting with correlations...'
    levels = fitter.fit_multi(x, y, loo=True, n_iterations=2)
    res = levels[-1]
    print 'Theta:'
    for ti in res.theta:
        print '  {}'.format(ti)
    print 'Sigma:'
    print res.sigma
    plot_series(series, fitter.shape, res.theta, res.LOO_predictions)
    
    R2_pairs = []
    for i,g in enumerate(series.gene_names):
        y_real = y[:,i]
        y_basic = fits[i].LOO_predictions        
        y_multi_gene = res.LOO_predictions[:,i]  # no NANs in the generated data, so no need to handle the original_inds mess
        basic_R2 = loo_score(y_real,y_basic)
        multi_gene_R2 = loo_score(y_real,y_multi_gene)
        R2_pairs.append( (basic_R2, multi_gene_R2) )
    plot_comparison_scatter(R2_pairs,series.region_name)
    print 'R2_pairs = {}'.format(R2_pairs)
