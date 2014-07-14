import setup
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from sklearn.datasets.base import Bunch
from load_data import SeveralGenesOneRegion
from plots import plot_series
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from fit_score import loo_score

cfg.fontsize = 18
cfg.xtick_fontsize = 18
cfg.ytick_fontsize = 18

def get_fitter():
    shape = Sigmoid(priors='sigmoid_wide')
    fitter = Fitter(shape, sigma_prior='normal')
    return fitter    

def get_series():
    n = 100
    rng = np.random.RandomState(cfg.random_seed)
    x = np.linspace(0,100,n) + rng.normal(0,0.1,size=n)
    x.sort()
    
    sigmoid = Sigmoid()
    t1 = (-1,4,50,5)
    y1 = sigmoid.f(t1,x)
    t2 = (1,3,25,2)
    y2 = sigmoid.f(t2,x)
    c = -0.95
    sigma = [[1, c], [c, 1]]
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
    
    ##############################################################
    # check regular fitting of multi series
    ##############################################################
    theta, sigma, LOO_predictions,_ = fitter.fit(series.ages, series.expression, loo=True)
    print 'sigma:\n{}'.format(sigma)
    plot_series(series, fitter.shape, theta, LOO_predictions)

    ##############################################################
    # check fit_multiple_series_with_cache
    ##############################################################
    fits = []
    for i,g in enumerate(series.gene_names):
        x = series.ages
        y = series.expression[:,i]
        theta, sigma, LOO_predictions, LOO_fits = fitter.fit(x,y,loo=True)
        fit = Bunch(
            theta = theta,
            LOO_predictions = LOO_predictions,
            LOO_fits = LOO_fits,
        )
        fits.append(fit)
            
    def cache(iy,ix):
        fit = fits[iy]
        if ix is None:
            return fit.theta
        else:
            theta,sigma = fit.LOO_fits[ix]
            return theta    
    
    x = series.ages
    y = series.expression
    multi_gene_preds,_ = fitter.fit_multiple_series_with_cache(x,y,cache)
    
    R2_pairs = []
    for i,g in enumerate(series.gene_names):
        y_real = y[:,i]
        y_basic = fits[i].LOO_predictions
        y_multi_gene = multi_gene_preds[:,i]        
        basic_R2 = loo_score(y_real,y_basic)
        multi_gene_R2 = loo_score(y_real,y_multi_gene)
        R2_pairs.append( (basic_R2, multi_gene_R2) )
    plot_comparison_scatter(R2_pairs,series.region_name)
    print 'R2_pairs = {}'.format(R2_pairs)
