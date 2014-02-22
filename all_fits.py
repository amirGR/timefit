# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:26:02 2014

@author: ronnie
"""

import pickle
import numpy as np
from sklearn.datasets.base import Bunch
from sigmoid_fit import loo_score
import config as cfg
import project_dirs

def _cache_file(pathway, dataset, b_hadas_fits):
    from os.path import join
    if b_hadas_fits:
        return join(project_dirs.cache_dir(), dataset, 'fits-{}-hadas.pkl'.format(pathway))
    else:
        return join(project_dirs.cache_dir(), dataset, 'fits-{}.pkl'.format(pathway))

def get_all_fits(data, b_hadas_fits=False):
    filename = _cache_file(data.pathway, data.dataset, b_hadas_fits)
    
    # load the cache we have so far
    try:
        with open(filename) as f:
            fits = pickle.load(f)
    except:
        fits = {}
        
    # check if it already contains all the fits (heuristic by number of fits)
    if len(fits) == len(data.gene_names)*len(data.region_names):
        return compute_scores(data, fits)  
    
    assert len(data.gene_names) < 100, "So many genes... Not doing this!"
    
    # compute the fits that are missing
    for g in data.gene_names:
        has_change = False
        for r in data.region_names:
            if (g,r) not in fits:
                has_change = True
                series = data.get_one_series(g,r)
                fits[(g,r)] = compute_fit(series, b_hadas_fits)
    
        # save checkpoint after each gene
        if has_change:
            print 'Saving fits for gene {}'.format(g)
            with open(filename,'w') as f:
                pickle.dump(fits,f)
    
    return compute_scores(data, fits)  

def compute_scores(data,fits):
    for ig,g in enumerate(data.gene_names):
        for ir,r in enumerate(data.region_names):
            series = data.get_one_series(ig,ir)
            fit = fits[(g,r)]
            fit.fit_score = cfg.score(series.expression, fit.fit_predictions)
            fit.LOO_score = loo_score(series.expression, fit.LOO_predictions)
    return fits
   
def compute_fit(series, b_hadas_fits):
    print 'Computing fit for {}@{}'.format(series.gene_name, series.region_name)
    x = series.ages
    y = series.expression

    if b_hadas_fits:
        from sigmoid_fit_hadas import sigmoid, fit_sigmoid_simple, find_best_L, fit_sigmoid_loo      
        L = find_best_L(x,y)
        theta = fit_sigmoid_simple(x,y,L)
        p = 1 # YYY - compute this from residuals
        P = np.array(list(theta) + [p])
        fit_predictions = sigmoid(theta,x)
        LOO_predictions = fit_sigmoid_loo(x,y)
    else:
        from sigmoid_fit import sigmoid, fit_sigmoid_simple, fit_sigmoid_loo         
        P = fit_sigmoid_simple(x,y)
        fit_predictions = sigmoid(P[:-1],x)
        LOO_predictions = fit_sigmoid_loo(x,y)
    
    return Bunch(
        seed = cfg.random_seed,
        P = P,
        fit_predictions = fit_predictions,
        LOO_predictions = LOO_predictions,
    )
