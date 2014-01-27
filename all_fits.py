# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:26:02 2014

@author: ronnie
"""

import pickle
from sklearn.datasets.base import Bunch
from sigmoid_fit import sigmoid, fit_sigmoid_simple, fit_sigmoid_loo
import config as cfg
import project_dirs

def _cache_file():
    from os.path import join    
    return join(project_dirs.cache_dir(), 'fits.pkl')

def get_all_fits(data):
    filename = _cache_file()
    
    # load the cache we have so far
    try:
        with open(filename) as f:
            fits = pickle.load(f)
    except:
        fits = {}
        
    # check if it already contains all the fits (heuristic by number of fits)
    if len(fits) == len(data.gene_names)*len(data.region_names):
        return fits    
        
    # compute the fits that are missing
    for ig,g in enumerate(data.gene_names):
        has_change = False
        for ir,r in enumerate(data.region_names):
            if (g,r) not in fits:
                has_change = True
                series = data.get_one_series(ig,ir)
                fits[(g,r)] = compute_fit(series)
    
        # save checkpoint after each gene
        if has_change:
            print 'Saving fits for gene {}'.format(g)
            with open(filename,'w') as f:
                pickle.dump(fits,f)
    
    return fits    
            
def compute_fit(series):
    print 'Computing fit for {}@{}'.format(series.gene_name, series.region_name)
    x = series.ages
    y = series.expression
    P = fit_sigmoid_simple(x,y)
    fit_predictions = sigmoid(P[:-1],x)
    LOO_predictions = fit_sigmoid_loo(x,y)
    return Bunch(
        seed = cfg.random_seed,
        P = P,
        fit_predictions = fit_predictions,
        LOO_predictions = LOO_predictions,
    )
