import setup
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from load_data import load_data
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from all_fits import get_all_fits
from plots import save_figure
import config as cfg

data = load_data()
fitter = Fitter(Sigmoid(),False,False)
fits = get_all_fits(data,fitter)

params = {
    'a': ('$a$', lambda f: f.theta[0]),
    'h': ('$h$', lambda f: f.theta[1]),
    'mu': (r'$\mu$', lambda f: f.theta[2]),
    'w': ('$w$', lambda f: f.theta[3]),
    'p': (r'$1/\sigma$', lambda f: 1/f.sigma),
}

def create_hist(p, low, high, draw=True, bins=20, fit_gamma=False, fit_normal=False):
    latex,getter = params[p]
    vals = np.array([getter(f) for f in fits.itervalues()])
    vals = vals[(vals>low) & (vals<high)]
    pct_captured = int(100*len(vals)/len(fits))
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

#