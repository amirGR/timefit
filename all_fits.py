import pickle
import os
from os.path import dirname, join
from glob import glob
import numpy as np
from scipy.io import savemat
from sklearn.datasets.base import Bunch
from sklearn.externals.joblib import Parallel, delayed
from fit_score import loo_score
import config as cfg
from project_dirs import cache_dir, fit_results_relative_path
from utils import ensure_dir, list_of_strings_to_matlab_cell_array, init_array

def _cache_file(data, fitter):
    return join(cache_dir(), fit_results_relative_path(data,fitter) + '.pkl')

def _read_one_cache_file(filename):
    try:
        if cfg.verbosity > 0:
            print 'Reading fits from {}'.format(filename)
        with open(filename) as f:
            fits = pickle.load(f)
    except:
        fits = {}
    return fits
    
def _save_fits(fits, filename, k_of_n):
    if k_of_n is not None:
        k,n = k_of_n
        filename = '{}.{}-of-{}'.format(filename,k,n)
    with open(filename,'w') as f:
        pickle.dump(fits,f)

def _read_all_cache_files(basefile, gene_names, b_consolidate):
    # collect fits from basefile and all shard files
    fits = _read_one_cache_file(basefile)
    partial_files = set(glob(basefile + '*')) - {basefile}
    for filename in partial_files:
        fits.update(_read_one_cache_file(filename))

    # reduce to the set we need (especially if we're working on a shard)
    fits = {(g,r):v for (g,r),v in fits.iteritems() if g in set(gene_names)}
        
    if b_consolidate:
        _save_fits(fits,basefile,None)
        for filename in partial_files:
            os.remove(filename)

    return fits

def get_all_fits(data, fitter, k_of_n=None):
    gene_names = data.gene_names
    if k_of_n is not None:
        k,n = k_of_n
        gene_names = [g for i,g in enumerate(gene_names) if i%n == k-1] # k is one-based

    filename = _cache_file(data, fitter)
    ensure_dir(dirname(filename))
    fits = _read_all_cache_files(filename, gene_names, b_consolidate = (k_of_n is None))

    missing_genes = set(g for g in gene_names for r in data.region_names if (g,r) not in fits)
    if cfg.verbosity > 0:
        print 'Still need to compute fits for {} of {} genes'.format(len(missing_genes),len(gene_names))

    if not missing_genes:
        return compute_scores(data, fits)
        
    # compute the fits that are missing
    for g in gene_names:
        pool = Parallel(n_jobs=cfg.all_fits_n_jobs, verbose=cfg.all_fits_verbose)
        df = delayed(_compute_fit_job)
        changes = pool(df(data,g,r,fitter) for r in data.region_names if (g,r) not in fits)
        if not changes:
            continue
        
        # apply changes and save checkpoint after each gene
        for g2,r2,f in changes:
            fits[(g2,r2)] = f
        print 'Saving fits for gene {}'.format(g)
        _save_fits(fits, filename, k_of_n)
    
    return compute_scores(data, fits)  

def _compute_fit_job(data, g, r, fitter):
    import utils
    utils.disable_all_warnings()
    series = data.get_one_series(g,r)
    f = compute_fit(series,fitter)
    return g,r,f    
    
def compute_scores(data,fits):
    for (g,r),fit in fits.iteritems():
        series = data.get_one_series(g,r)
        try:
            if fit.fit_predictions is None:
                fit.fit_score = None
            else:
                fit.fit_score = cfg.score(series.expression, fit.fit_predictions)
        except:
            fit.fit_score = None
        try:
            fit.LOO_score = loo_score(series.expression, fit.LOO_predictions)
        except:
            fit.LOO_score = None
    return fits
   
def compute_fit(series, fitter):
    print 'Computing fit for {}@{} using {}'.format(series.gene_name, series.region_name, fitter)
    x = series.ages
    y = series.expression
    theta,sigma,LOO_predictions = fitter.fit(x,y,loo=True)
    if theta is None:
        print 'WARNING: Optimization failed during overall fit for {}@{} using {}'.format(series.gene_name, series.region_name, fitter)
        fit_predictions = None
    else:
        fit_predictions = fitter.predict(theta,x)
    
    return Bunch(
        fitter = fitter,
        seed = cfg.random_seed,
        theta = theta,
        sigma = sigma,
        fit_predictions = fit_predictions,
        LOO_predictions = LOO_predictions,
    )

def save_as_mat_file(fits, filename):
    print 'Saving mat file to {}'.format(filename)
    gene_names = sorted(list(set(g for g,r in fits.iterkeys())))
    n_genes = len(gene_names)
    gene_idx = {g:i for i,g in enumerate(gene_names)}

    region_names = sorted(list(set(r for g,r in fits.iterkeys())))
    region_idx = {r:i for i,r in enumerate(region_names)}
    n_regions = len(region_names)
    
    st_n_theta = set(len(fit.theta) for fit in fits.itervalues() if fit.theta is not None)
    assert len(st_n_theta) == 1, "Can't determine number of parameters. candidates={}".format(list(st_n_theta))
    n_theta = st_n_theta.pop()
    
    fit_scores = init_array(np.NaN, n_genes,n_regions)
    LOO_scores = init_array(np.NaN, n_genes,n_regions)
    theta = init_array(np.NaN, n_theta,n_genes,n_regions)
    for (g,r),fit in fits.iteritems():
        ig = gene_idx[g]
        ir = region_idx[r]
        fit_scores[ig,ir] = fit.fit_score
        LOO_scores[ig,ir] = fit.LOO_score
        if fit.theta is not None:
            theta[:,ig,ir] = fit.theta
    
    mdict = {
        'gene_names' : list_of_strings_to_matlab_cell_array(gene_names),
        'region_names' : list_of_strings_to_matlab_cell_array(region_names),
        'theta' : theta,
        'fit_scores' : fit_scores,
        'LOO_scores' : LOO_scores,
    }
    savemat(filename, mdict, oned_as='column')
    
def convert_format(filename, f_convert):
    """Utility function for converting the format of cached fits.
       See e.g. scripts/convert_fit_format.py
    """
    with open(filename) as f:
        fits = pickle.load(f)        
    print 'Found cache file with {} fits'.format(len(fits))
    
    print 'Converting...'
    new_fits = {k:f_convert(v) for k,v in fits.iteritems()}
    
    print 'Saving converted fits to {}'.format(filename)
    with open(filename,'w') as f:
        pickle.dump(new_fits,f)

