import setup
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from load_data import GeneData
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from all_fits import get_all_fits, iterate_fits
import config as cfg
from scalers import LogScaler
from sklearn.datasets.base import Bunch

params = {
    'a': ('$a$', lambda f: f.theta[0]),
    'h': ('$h$', lambda f: f.theta[1]),
    'mu': (r'$\mu$', lambda f: f.theta[2]),
    'w': ('$w$', lambda f: f.theta[3]),
    'p': (r'$1/\sigma$', lambda f: 1/f.sigma),
}

def create_hist(flat_fits, p, low, high, draw=True, bins=20, fit_gamma=True, fit_normal=True):
    latex,getter = params[p]
    vals = np.array([getter(f) for f in flat_fits])
    vals = vals[(vals>low) & (vals<high)]
    pct_captured = int(100*len(vals)/len(flat_fits))
    if draw:
        plt.figure()
        plt.hist(vals,bins,normed=True,color='b')
        xmin,xmax = plt.xlim()
        plt.xlabel('x',fontsize=cfg.fontsize)
        plt.ylabel('p(x)',fontsize=cfg.fontsize)
        ttl1 = 'Distribution of parameter {} (Centeral mass: {}% of values)'.format(latex,pct_captured)
        ttl2 = '(created with low={}, high={})'.format(low,high)
        ttl = '\n'.join([ttl1,ttl2])
        if fit_gamma:
            alpha,loc,scale=stats.gamma.fit(vals)
            beta = 1/scale
            rv = stats.gamma(alpha,loc,scale)
            x = np.linspace(loc,xmax,100)
            prob = rv.pdf(x)
            plt.plot(x,prob,'g',linewidth=3)
            ttl_fit = r'Gamma fit: $\alpha$={:.3f}, $\beta$={:.3f}, $loc$={:.3f}'.format(alpha,beta,loc)
            ttl = '\n'.join([ttl, ttl_fit])
        if fit_normal:
            loc,sigma=stats.norm.fit(vals)
            rv = stats.norm(loc,sigma)
            x = np.linspace(xmin,xmax,100)
            prob = rv.pdf(x)
            plt.plot(x,prob,'k',linewidth=3)
            ttl_fit = r'Normal fit: $loc$={:.3f}, $\sigma$={:.3f}'.format(loc,sigma)
            ttl = '\n'.join([ttl, ttl_fit])
        plt.title(ttl)
    return vals

cfg.verbosity = 1
age_scaler = LogScaler()
pathway = 'serotonin'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
shape = Sigmoid()
fitter = Fitter(shape)
fits = get_all_fits(data,fitter, allow_new_computation=False)

def translate(g,r,fit):
    series = data.get_one_series(g,r)
    theta,sigma = fitter.translate_parameters_to_priors_scale(series.ages, series.single_expression, fit.theta, fit.sigma)
    a,h,mu,w = theta
    if h < 0:
        theta = (a+h,-h,mu,-w) # this is an equivalent sigmoid, with h now positive
    return Bunch(
        theta = theta,
        sigma = sigma,
    )
    
flat_fits = [translate(g,r,fit) for dsname,g,r,fit in iterate_fits(fits, return_keys=True)]

# This script is meant to be run as a setup, then run commands interactively, e.g.:
create_hist(flat_fits, 'a', -2, 1)
create_hist(flat_fits, 'h', -1, 3)
create_hist(flat_fits, 'w', -0.5, 1)
create_hist(flat_fits, 'mu', -2, 2)
create_hist(flat_fits, 'p', 0, 10)