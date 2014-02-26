import setup
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import load_data
from all_fits import compute_fit
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from plots import plot_one_series

import utils
utils.disable_all_warnings()
    
regions = ['VFC']
genes = ['HTR1E']

cfg.sorted_regions = regions
cfg.pathways['small'] = genes
cfg.all_fits_n_jobs = 1
#cfg.inv_sigma_prior_mean = 100
#cfg.inv_sigma_prior_sigma = 0.01
cfg.sigmoid_theta_prior_mean = np.array([10, 5, 30, 2.5])
cfg.sigmoid_theta_prior_sigma = np.array([0.1, 5, 30, 2.5])


data = load_data(pathway='small').restrict_regions(regions)
series = data.get_one_series(0,0)
x = series.ages
y = series.expression

shape = Sigmoid()

def fit_simple(b_priors=False):
    fitter = Fitter(shape,b_priors,b_priors)
    theta,sigma = fitter.fit_simple(x,y)
    plot_one_series(series)
    x_smooth, y_smooth = shape.high_res_preds(theta,x)
    plt.plot(x_smooth,y_smooth, linewidth=2)
    a,h,mu,w = theta
    P_ttl = r'(a={a:.2f}, h={h:.2f}, $\mu$={mu:.2f}, w={w:.2f}, $\sigma$={sigma:.2f})'.format(**locals())
    plt.title('\n'.join([plt.gca().get_title(), P_ttl]))

def fit_loo(b_priors=False):
    fitter = Fitter(shape,b_priors,b_priors)
    fit = compute_fit(series,fitter)
    plot_one_series(series,fit=fit)
   
#variations = list(product([False,True],[False,True]))
#dct = {(t,s):get_all_fits(data,Fitter(shape,t,s)) for t,s in variations}
#dct_R2_mean = {k:np.mean([f.LOO_score for f in fits.itervalues()]) for k,fits in dct.iteritems()}
#dct_R2_median = {k:np.mean([f.LOO_score for f in fits.itervalues()]) for k,fits in dct.iteritems()}