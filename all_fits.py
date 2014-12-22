from os.path import join
import cPickle as pickle
from itertools import product
import numpy as np
from scipy.io import savemat
from sklearn.datasets.base import Bunch
from fit_score import loo_score
import config as cfg
from project_dirs import cache_dir, fit_results_relative_path
from utils.misc import init_array, covariance_to_correlation
from utils.formats import list_of_strings_to_matlab_cell_array
from utils import job_splitting
import scalers

class Fits(dict):
    # This is just a placeholder for now, till I have the time to refactor this into
    # a real class and change all the code using it appropriately.
    # For now the inheritance from dict just allows adding additinal fields, like change_distribution_params 
    pass 

def get_all_fits(data, fitter, k_of_n=None, n_correlation_iterations=0, correlations_k_of_n=None, allow_new_computation=True):
    """Returns { dataset_name -> {(gene,region) -> fit} } for all datasets in 'data'.
    """
    return Fits({ds.name : _get_dataset_fits(data, ds, fitter, k_of_n, n_correlation_iterations, correlations_k_of_n, allow_new_computation) for ds in data.datasets})

def _get_dataset_fits(data, dataset, fitter, k_of_n, n_correlation_iterations, correlations_k_of_n, allow_new_computation):
    def arg_mapper(gr,f_proxy):
        g,r = gr
        series = dataset.get_one_series(g,r)
        return f_proxy(series,fitter)
        
    # sharding is done by gene, so plots.plot_and_save_all_genes can work on a shard
    # this also requires that the list of all genes be taken from the whole data
    # and not from each dataset. Otherwise we can get a mismatch between the genes 
    # in the shard for different datasets.
    dataset_fits = job_splitting.compute(
        name = 'fits',
        f = _compute_fit,
        arg_mapper = arg_mapper,
        all_keys = list(product(dataset.gene_names,dataset.region_names)),
        all_sharding_keys = data.gene_names,
        f_sharding_key = lambda gr: gr[0],
        k_of_n = k_of_n,
        base_filename = fit_results_relative_path(dataset,fitter),
        allow_new_computation = allow_new_computation,
    )
    
    if n_correlation_iterations > 0:
        # The problem is that if we're using a shard for the basic fits we won't have theta for all genes in a region
        # which is necessary for computing correlations in that region.
        assert k_of_n is None, "Can't perform correlation computations when sharding is enabled at the basic fit level" 
        _add_dataset_correlation_fits(dataset, fitter, dataset_fits, n_correlation_iterations, correlations_k_of_n, allow_new_computation)

    _add_scores(dataset, dataset_fits)
    
    return dataset_fits

def _add_dataset_correlation_fits(dataset, fitter, ds_fits, n_iterations, k_of_n, allow_new_computation):
    def arg_mapper(key, f_proxy):
        ir, loo_point = key
        r = dataset.region_names[ir]
        series = dataset.get_several_series(dataset.gene_names,r)
        basic_theta = [ds_fits[(g,r)].theta for g in dataset.gene_names]
        return f_proxy(series, fitter, basic_theta, loo_point, n_iterations)
        
    all_keys = []
    for ir,r in enumerate(dataset.region_names):
        all_keys.append((ir,None))
        series = dataset.get_several_series(dataset.gene_names,r)
        for iy,g in enumerate(dataset.gene_names):
            for ix in xrange(len(series.ages)):
                loo_point = (ix,iy)
                all_keys.append((ir,loo_point))
        
    def f_sharding_key(key): # keep all x points in the same shard for same r,iy
        r, loo_point = key
        if loo_point is None:
            return (r,None)
        else:
            ix,iy = loo_point
            return (r,iy)
        
    dct_results = job_splitting.compute(
        name = 'fits-correlations',
        f = _compute_fit_with_correlations,
        arg_mapper = arg_mapper,
        all_keys = all_keys,
        f_sharding_key = f_sharding_key,
        k_of_n = k_of_n,
        base_filename = fit_results_relative_path(dataset,fitter) + '-correlations-{}'.format(n_iterations),
        allow_new_computation = allow_new_computation,
    )
    _add_dataset_correlation_fits_from_results_dictionary(dataset, ds_fits, dct_results)
    
def _add_dataset_correlation_fits_from_results_dictionary(dataset, ds_fits, dct_results):
    """This function converts the results of the job_splitting which is a flat dictionary to structures which 
       are easier to use and integrated into the dataset fits
    """
    region_to_ix_original_inds = {}
    for ir,r in enumerate(dataset.region_names):
        series = dataset.get_several_series(dataset.gene_names,r)
        region_to_ix_original_inds[r] = series.original_inds
        
    for (ir,loo_point), levels in dct_results.iteritems():
        n_iterations = len(levels)
        r = dataset.region_names[ir]
        if loo_point is None:
            # Global fit - collect the parameters (theta, sigma, L) and compute a correlation matrix for the region
            # the hack of using the key (None,r) to store these results can be removed if/when dataset fits is changed from a dictionary to a class with several fields
            k = (None,r)
            if k not in ds_fits:
                ds_fits[k] = n_iterations*[None]
            for iLevel, level in enumerate(levels):
                ds_fits[k][iLevel] = level
                level.correlations = covariance_to_correlation(level.sigma)
        else:
            # LOO point - collect the predictions
            ix,iy = loo_point
            g = dataset.gene_names[iy]
            fit = ds_fits[(g,r)]
            if not hasattr(fit, 'with_correlations'):
                fit.with_correlations = [
                    Bunch(LOO_predictions=init_array(np.NaN, len(dataset.ages)))  # NOTE: we place the predictions at the original indexes (before NaN were removed by the get_series)
                    for _ in xrange(n_iterations)
                ]
            for iLevel, level_prediction in enumerate(levels):
                orig_ix = region_to_ix_original_inds[r][ix]
                fit.with_correlations[iLevel].LOO_predictions[orig_ix] = level_prediction
    

def _compute_fit_with_correlations(series, fitter, basic_theta, loo_point, n_iterations):
    if cfg.verbosity > 0:
        print 'Computing fit with correlations ({n_iterations} iterations) for LOO point {loo_point} at {series.region_name} using {fitter}'.format(**locals())
    return fitter.fit_multiple_series_with_cache(series.ages, series.expression, basic_theta, loo_point, n_iterations)


def _add_scores(dataset,dataset_fits):
    for (g,r),fit in dataset_fits.iteritems():
        if g is None:
            continue  # it's a region fit
            
        series = dataset.get_one_series(g,r)
        try:
            if fit.fit_predictions is None:
                fit.fit_score = None
            else:
                fit.fit_score = cfg.score(series.single_expression, fit.fit_predictions)
        except:
            fit.fit_score = None
        try:
            fit.LOO_score = loo_score(series.single_expression, fit.LOO_predictions)
        except:
            fit.LOO_score = None
            
        # add score for correlation LOO fits
        correlation_levels = getattr(fit, 'with_correlations', None)
        if correlation_levels is not None:
            for level in correlation_levels:
                y_real = series.single_expression
                y_pred = level.LOO_predictions[series.original_inds] # match the predictions to the indices of the single series after NaN are removed from it
                level.LOO_score = loo_score(y_real, y_pred)
            
    return dataset_fits

def _compute_fit(series, fitter):
    if cfg.verbosity > 0:
        print 'Computing fit for {}@{} using {}'.format(series.gene_name, series.region_name, fitter)
    x = series.ages
    y = series.single_expression
    theta,sigma,LOO_predictions,LOO_fits = fitter.fit(x,y,loo=True)
    if theta is None:
        print 'WARNING: Optimization failed during overall fit for {}@{} using {}'.format(series.gene_name, series.region_name, fitter)
        fit_predictions = None
        theta_samples = None
    else:
        fit_predictions = fitter.shape.f(theta,x)
        if fitter.shape.parameter_type() == object:
            theta_samples = None
        else:
            theta_samples = fitter.parametric_bootstrap(x, theta, sigma)
    
    return Bunch(
        fitter = fitter,
        seed = cfg.random_seed,
        theta = theta,
        sigma = sigma,
        fit_predictions = fit_predictions,
        LOO_predictions = LOO_predictions,
        theta_samples = theta_samples,
    )

def save_as_mat_files(data, fitter, fits, has_change_distributions):
    for dataset in data.datasets:
        filename = join(cache_dir(), fit_results_relative_path(dataset,fitter) + '.mat')
        dataset_fits = fits[dataset.name]
    
        print 'Saving mat file to {}'.format(filename)
        shape = fitter.shape
        
        gene_names = dataset.gene_names
        gene_idx = {g:i for i,g in enumerate(gene_names)}
        n_genes = len(gene_names)
        region_names = dataset.region_names
        region_idx = {r:i for i,r in enumerate(region_names)}
        n_regions = len(region_names)
        
        write_theta = shape.can_export_params_to_matlab()
        if write_theta:
            theta = init_array(np.NaN, shape.n_params(), n_genes,n_regions)
        else:
            theta = np.NaN
        
        fit_scores = init_array(np.NaN, n_genes,n_regions)
        LOO_scores = init_array(np.NaN, n_genes,n_regions)
        fit_predictions = init_array(np.NaN, *dataset.expression.shape)
        LOO_predictions = init_array(np.NaN, *dataset.expression.shape)
        high_res_predictions = init_array(np.NaN, cfg.n_curve_points_to_plot, n_genes, n_regions)
        scaled_high_res_ages = np.linspace(dataset.ages.min(), dataset.ages.max(), cfg.n_curve_points_to_plot)
        original_high_res_ages = scalers.unify(dataset.age_scaler).unscale(scaled_high_res_ages)
        if has_change_distributions:
            change_distribution_bin_centers = fits.change_distribution_params.bin_centers
            n_bins = len(change_distribution_bin_centers)
            change_distribution_weights = init_array(np.NaN, n_bins, n_genes, n_regions)
        else:
            change_distribution_bin_centers = []
            change_distribution_weights = []
        for (g,r),fit in dataset_fits.iteritems():
            series = dataset.get_one_series(g,r)
            ig = gene_idx[g]
            ir = region_idx[r]
            fit_scores[ig,ir] = fit.fit_score
            LOO_scores[ig,ir] = fit.LOO_score
            if write_theta and fit.theta is not None:
                theta[:,ig,ir] = fit.theta
            fit_predictions[series.original_inds,ig,ir] = fit.fit_predictions
            LOO_predictions[series.original_inds,ig,ir] = fit.LOO_predictions
            high_res_predictions[:,ig,ir] = shape.f(fit.theta, scaled_high_res_ages)
            change_weights = getattr(fit,'change_distribution_weights',None)
            if change_weights is not None:
                change_distribution_weights[:,ig,ir] = change_weights
        mdict = dict(
            gene_names = list_of_strings_to_matlab_cell_array(gene_names),
            region_names = list_of_strings_to_matlab_cell_array(region_names),
            theta = theta,
            fit_scores = fit_scores,
            LOO_scores = LOO_scores,
            fit_predictions = fit_predictions,
            LOO_predictions = LOO_predictions,
            high_res_predictions = high_res_predictions,
            high_res_ages = original_high_res_ages,
            change_distribution_bin_centers = change_distribution_bin_centers,
            change_distribution_weights = change_distribution_weights,
        )
        savemat(filename, mdict, oned_as='column')

def save_theta_text_files(data, fitter, fits):
    assert fitter.shape.cache_name() == 'spline', "save to text is only supported for splines at the moment"
    for dataset in data.datasets:
        filename = join(cache_dir(), fit_results_relative_path(dataset,fitter) + '.txt')
        dataset_fits = fits[dataset.name]    
        print 'Saving text file to {}'.format(filename)
        with open(filename, 'w') as f:
            for (g,r),fit in dataset_fits.iteritems():
                if fit.theta is None:
                    continue
                knots, coeffs, degree = fit.theta[0]
                knots = list(knots)
                coeffs = list(coeffs)
                gr_text = """\
Gene symbol: {g}
Region: {r}
Spline knots: {knots}
Spline coefficients: {coeffs}
Spline degree: {degree}
""".format(**locals())
                print >>f, gr_text

    
def restrict_genes(fits, genes):
    if genes is None:
        return fits
    genes = set(genes)
    new_fits = {}
    for ds_key,ds_fits in fits.iteritems():
        new_fits[ds_key] = {(g,r):fit for (g,r),fit in ds_fits.iteritems() if g in genes}
    return new_fits
    
def iterate_fits(fits, fits2=None, R2_threshold=None, allow_no_theta=False, return_keys=False):
    def fit_ok(fit):
        if not allow_no_theta and fit.theta is None:
            return False
        if R2_threshold is not None and fit.LOO_score < R2_threshold:
            return False
        return True
        
    for dsname,dsfits in fits.iteritems():
        for (g,r),fit in dsfits.iteritems():
            if g is None:
                continue  # skip region fits
            if not fit_ok(fit):
                continue
            if fits2 is None:
                if return_keys:
                    yield dsname,g,r,fit
                else:
                    yield fit
            else:
                fit2 = fits2[dsname][(g,r)]
                if not fit_ok(fit2):
                    continue
                if return_keys:
                    yield dsname,g,r,fit,fit2
                else:
                    yield fit,fit2

def iterate_region_fits(data, fits, allow_missing=False):
    for region in data.region_names:
        ds_fits = fits[data.get_dataset_for_region(region)]
        rfit = ds_fits.get( (None,region) )
        if rfit is None:
            assert allow_missing, "Data for region (correlation) fit is missing for region {}".format(region)
            continue
        yield region, rfit
    
def convert_format(filename, f_convert):
    """Utility function for converting the format of cached fits.
       See e.g. scripts/convert_fit_format.py
    """
    with open(filename) as f:
        dataset_fits = pickle.load(f)        
    print 'Found cache file with {} fits'.format(len(dataset_fits))
    
    print 'Converting...'
    new_dataset_fits = {k:f_convert(v) for k,v in dataset_fits.iteritems()}
    
    print 'Saving converted fits to {}'.format(filename)
    with open(filename,'w') as f:
        pickle.dump(new_dataset_fits,f)

