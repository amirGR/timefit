import setup
import numpy as np
import config as cfg
from load_data import SeveralGenesOneRegion
from plots import plot_series
from shapes.sigmoid import Sigmoid
from fitter import Fitter

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
    sigma = [[1, -0.95], [-0.95, 1]]
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

if __name__ == '__main__':
    fitter = get_fitter()
    series = get_series()    
    theta, sigma, LOO_predictions = fitter.fit(series.ages, series.expression, loo=True)
    print 'sigma:\n{}'.format(sigma)
    plot_series(series, fitter.shape, theta, LOO_predictions)
