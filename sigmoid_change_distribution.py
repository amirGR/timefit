import math
from os.path import join
import numpy as np
from sklearn.datasets.base import Bunch
from all_fits import iterate_fits
from project_dirs import cache_dir, fit_results_relative_path
from utils.misc import cache, init_array

def calc_change_distribution(shape, theta, bin_edges):
    a,h,mu,w = theta
    edge_vals = shape.f(theta,bin_edges)
    changes = np.abs(edge_vals[1:] - edge_vals[:-1])
    return changes / abs(h) # ignore change magnitude per gene - take only distribution of change times
    
def change_distribution_mean_and_std(bin_centers, weights):
    weights = np.array(weights, dtype=float) # in case it's a list
    weights /= np.sum(weights) # normalize to make it a PMF
    x0 = np.sum([x*w for x,w in zip(bin_centers,weights)])
    V = np.sum([w*(x-x0)**2 for x,w in zip(bin_centers,weights)])
    std = np.sqrt(V)
    return x0,std
    
def change_distribution_width_cumsum(bin_centers, weights, threshold=0.8):
    x_median, x_from, x_to = change_distribution_spread_cumsum(bin_centers, weights, threshold=0.8)
    return x_to - x_from

def change_distribution_spread_cumsum(bin_centers, weights, threshold=0.8):
    weights = np.array(weights, dtype=float) # in case it's a list
    weights /= np.sum(weights) # normalize to make it a PMF
    bin_width = bin_centers[1] - bin_centers[0] # we assume uniform bins here
    s = np.cumsum(weights)
    i_from = np.argmax(s > 0.5 - threshold/2.0)
    i_to = np.argmax(s > 0.5 + threshold/2.0)
    i_median = np.argmax(s > 0.5)
    x_from = bin_centers[i_from] - 0.5*bin_width
    x_to = bin_centers[i_to] + 0.5*bin_width
    x_median = bin_centers[i_median]
    return x_median, x_from, x_to

def add_change_distributions(data, fitter, fits, age_range=None, n_bins=50):
    """ Compute a histogram of "strength of transition" at different ages.
        The histogram is computed for each (gene,region) in fits and is added to the fit objects.
        Currently this function only works for sigmoid fits. It uses the h parameter explicitly,
        relies on monotonicity, etc. It is probably not too hard to generalize it to other shapes.
    """
    shape = fitter.shape
    assert shape.cache_name() == 'sigmoid' # the function currently works only for sigmoid fits

    if age_range is None:
        age_range = data.age_range
    from_age, to_age = age_range
    bin_edges, bin_size = np.linspace(from_age, to_age, n_bins+1, retstep=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    fits.change_distribution_params = Bunch(
        bin_edges = bin_edges,
        bin_centers = bin_centers,
    )

    for dsname,g,r,fit in iterate_fits(fits, return_keys=True):
        thetas = fit.theta_samples # bootstrap samples of theta values
        n_params, n_samples = thetas.shape
        weights = np.zeros(n_bins)
        for i in xrange(n_samples):
            weights += calc_change_distribution(shape, thetas[:,i], bin_edges)
        weights /= n_samples # now values are in fraction of total change (doesn't have to sum up to 1 if ages don't cover the whole transition range)
        fit.change_distribution_weights = weights
        fit.change_distribution_spread = change_distribution_spread_cumsum(bin_centers, weights)
        fit.change_distribution_mean_std = change_distribution_mean_and_std(bin_centers, weights)

@cache(lambda data, fitter, fits: join(cache_dir(), fit_results_relative_path(data,fitter) + '-dprime-cube.pkl'))
def compute_dprime_measures_for_all_pairs(data, fitter, fits):
    genes = data.gene_names
    regions = data.region_names 
    r2ds = data.region_to_dataset()        
    cube_shape = (len(genes), len(regions), len(regions))
    d_mu = np.empty(cube_shape) # mu2-mu1 for all genes and region pairs
    std = np.empty(cube_shape) # std (combined) for all genes and region pairs
    def get_mu_std(g,r):
        dsfits = fits[r2ds[r]]
        fit = dsfits.get((g,r))
        if fit is None:
            return np.nan, np.nan
        else:
            return fit.change_distribution_mean_std
    for ig,g in enumerate(genes):
        for ir1,r1 in enumerate(regions):
            mu1, std1 = get_mu_std(g,r1)
            for ir2,r2 in enumerate(regions):
                mu2, std2 = get_mu_std(g,r2)
                d_mu[ig,ir1,ir2] = mu2 - mu1
                std[ig,ir1,ir2] = math.sqrt(0.5*(std1*std1 + std2*std2))
    return Bunch(d_mu=d_mu, std=std, genes=genes, regions=regions, age_scaler=data.age_scaler)

@cache(lambda data, fitter, fits: join(cache_dir(), fit_results_relative_path(data,fitter) + '-change-dist.pkl'))
def compute_timing_info_for_all_fits(data, fitter, fits):
    genes = data.gene_names
    regions = data.region_names 
    r2ds = data.region_to_dataset()        
    bin_edges = fits.change_distribution_params.bin_edges
    bin_centers = fits.change_distribution_params.bin_centers

    mu = init_array(np.NaN, len(genes), len(regions))
    std = init_array(np.NaN, len(genes), len(regions))
    weights = init_array(np.NaN, len(genes), len(regions), len(bin_centers))
    for ig,g in enumerate(genes):
        for ir,r in enumerate(regions):
            dsfits = fits[r2ds[r]]
            fit = dsfits.get((g,r))
            if fit is None:
                continue
            mu[ig,ir], std[ig,ir] = fit.change_distribution_mean_std
            weights[ig,ir,:] = fit.change_distribution_weights
            
    return Bunch(
        bin_edges = bin_edges,
        bin_centers = bin_centers,
        weights = weights,
        mu=mu, 
        std=std, 
        genes=genes, 
        regions=regions, 
        age_scaler=data.age_scaler,
    )

def compute_fraction_of_change(weights, bin_edges, x_from, x_to, normalize=False):
    total = 0
    for weight, bin_from, bin_to in zip(weights,bin_edges[:-1],bin_edges[1:]):
        bin_width = bin_to - bin_from
        effective_from = max(x_from, bin_from)
        effective_to = min(x_to, bin_to)
        effective_width = effective_to - effective_from
        if effective_width > 0:
            total += weight * effective_width/bin_width
    if normalize:
        total /= sum(weights)
    return total
