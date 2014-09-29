from os.path import join
import cPickle as pickle
from itertools import product
import numpy as np
from scipy.io import savemat
from sklearn.datasets.base import Bunch
from fit_score import loo_score
import config as cfg
from project_dirs import cache_dir, fit_results_relative_path
from utils.misc import init_array
from utils.formats import list_of_strings_to_matlab_cell_array
from utils import job_splitting
import scalers

class Fits(dict):
    # This is just a placeholder for now, till I have the time to refactor this into
    # a real class and change all the code using it appropriately.
    # For now the inheritance from dict just allows adding additinal fields, like change_distribution_params 
    pass 

def get_all_fits(data, fitter, k_of_n=None, allow_new_computation=True):
    """Returns { dataset_name -> {(gene,region) -> fit} } for all datasets in 'data'.
    """
    return Fits({ds.name : _get_dataset_fits(data,ds,fitter,k_of_n,allow_new_computation) for ds in data.datasets})

def _get_dataset_fits(data, dataset, fitter, k_of_n=None, allow_new_computation=True):
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
    _add_scores(dataset, dataset_fits)  
    return dataset_fits

def _add_scores(dataset,dataset_fits):
    for (g,r),fit in dataset_fits.iteritems():
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
    else:
        fit_predictions = fitter.shape.f(theta,x)
        
        # create bootstrap estimates of the parameters: resample points + add gaussian noise
        nSamples = cfg.n_parameter_estimate_bootstrap_samples
        dtype = fitter.shape.parameter_type()
        theta_samples = np.empty((len(theta),nSamples), dtype=dtype)
        rng = np.random.RandomState(cfg.random_seed)
        for iSample in range(nSamples):
            idx = np.floor(rng.rand(len(x))*len(x)).astype(int)
            x2 = x[idx]
            noise = rng.normal(0,sigma,x.shape)
            y2 = fit_predictions[idx] + noise
            theta_i, _, _,_ = fitter.fit(x2, y2)
            theta_samples[:,iSample] = theta_i
    
    return Bunch(
        fitter = fitter,
        seed = cfg.random_seed,
        theta = theta,
        sigma = sigma,
        fit_predictions = fit_predictions,
        LOO_predictions = LOO_predictions,
        LOO_fits = LOO_fits,
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
    
def restrict_genes(fits, genes):
    if genes is None:
        return fits
    genes = set(genes)
    new_fits = {}
    for ds_key,ds_fits in fits.iteritems():
        new_fits[ds_key] = {(g,r):fit for (g,r),fit in ds_fits.iteritems() if g in genes}
    return new_fits
    
def iterate_fits(fits, fits2=None, R2_threshold=None, allow_no_theta=False, return_keys=False):
    for dsname,dsfits in fits.iteritems():
        for (g,r),fit in dsfits.iteritems():
            if R2_threshold is not None and fit.LOO_score < R2_threshold:
                continue
            if not allow_no_theta and fit.theta is None:
                continue
            if fits2 is None:
                if return_keys:
                    yield dsname,g,r,fit
                else:
                    yield fit
            else:
                fit2 = fits2[dsname][(g,r)]
                if return_keys:
                    yield dsname,g,r,fit,fit2
                else:
                    yield fit,fit2
    
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

