import pickle
from sklearn.datasets.base import Bunch
from sklearn.externals.joblib import Parallel, delayed
from fit_score import loo_score
import config as cfg
import project_dirs

def _cache_file(pathway, dataset, fitter_name):
    from os.path import join
    return join(project_dirs.cache_dir(), dataset, 'fits-{}-{}.pkl'.format(pathway, fitter_name))

def get_all_fits(data,fitter):
    if fitter is None:
        from shapes.sigmoid import Sigmoid
        from fitter import Fitter
        fitter = Fitter(Sigmoid())
    filename = _cache_file(data.pathway, data.dataset, fitter.cache_name())
    
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
        fit.fit_score = cfg.score(series.expression, fit.fit_predictions)
        fit.LOO_score = loo_score(series.expression, fit.LOO_predictions)
    return fits
   
def compute_fit(series, fitter):
    print 'Computing fit for {}@{} using {}'.format(series.gene_name, series.region_name, fitter)
    x = series.ages
    y = series.expression
    theta,sigma = fitter.fit_simple(x,y)
    assert theta is not None, "Optimization failed during overall fit"
    fit_predictions = fitter.predict(theta,x)
    LOO_predictions = fitter.fit_loo(x,y)
    
    return Bunch(
        fitter = fitter,
        seed = cfg.random_seed,
        theta = theta,
        sigma = sigma,
        fit_predictions = fit_predictions,
        LOO_predictions = LOO_predictions,
    )

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

