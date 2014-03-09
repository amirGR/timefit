import pickle
from os.path import dirname
import numpy as np
from scipy.io import savemat
from sklearn.datasets.base import Bunch
from sklearn.externals.joblib import Parallel, delayed
from fit_score import loo_score
import config as cfg
import project_dirs
from utils import ensure_dir, list_of_strings_to_matlab_cell_array, init_array

def _cache_file(pathway, dataset, fitter_name):
    from os.path import join
    return join(project_dirs.cache_dir(), dataset, 'fits-{}-{}.pkl'.format(pathway, fitter_name))

def get_all_fits(data,fitter):
    filename = _cache_file(data.pathway, data.dataset, fitter.cache_name())
    ensure_dir(dirname(filename))
    
    # load the cache we have so far
    try:
        with open(filename) as f:
            fits = pickle.load(f)
    except:
        fits = {}
        
    # check if it already contains all the fits (heuristic by number of fits)
    if len(fits) == len(data.gene_names)*len(data.region_names):
        return compute_scores(data, fits)  
    
    assert len(data.gene_names) < 500, "So many genes... Not doing this!"
    
    # compute the fits that are missing
    for g in data.gene_names:
        pool = Parallel(n_jobs=cfg.all_fits_n_jobs, verbose=cfg.all_fits_verbose)
        df = delayed(_compute_fit_job)
        changes = pool(df(data,g,r,fitter) for r in data.region_names if (g,r) not in fits)
        if not changes:
            continue
        
        # apply changes and save checkpoint after each gene
        for g2,r2,f in changes:
            fits[(g2,r2)] = f
        print 'Saving fits for gene {}'.format(g)
        with open(filename,'w') as f:
            pickle.dump(fits,f)
    
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
    theta,sigma,LOO_predictions = fitter.fit_loo(x,y)
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

