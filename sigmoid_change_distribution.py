import math
from os.path import join
import numpy as np
from sklearn.datasets.base import Bunch
from all_fits import iterate_fits
from project_dirs import cache_dir, fit_results_relative_path
from utils.misc import cache, init_array, save_matfile
from utils.formats import list_of_strings_to_matlab_cell_array
import scalers

def bin_edges_to_centers(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:])/2

def get_bins(data, age_range=None, n_bins=50):
    if age_range is None:
        age_range = data.age_range
    from_age, to_age = age_range
    bin_edges, bin_size = np.linspace(from_age, to_age, n_bins+1, retstep=True)
    bin_centers = bin_edges_to_centers(bin_edges)
    return bin_edges, bin_centers

def calc_change_distribution(shape, theta, bin_edges):
    a,h,mu,_ = theta
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
    assert shape.cache_name() in ['sigmoid','sigslope'] # the function currently works only for sigmoid/sigslope fits

    bin_edges, bin_centers = get_bins(data, age_range, n_bins)
    fits.change_distribution_params = Bunch(
        bin_edges = bin_edges,
        bin_centers = bin_centers,
    )

    for dsname,g,r,fit in iterate_fits(fits, return_keys=True):
        weights = calc_bootstrap_change_distribution(shape, fit.theta_samples, bin_edges)
        fit.change_distribution_weights = weights
        fit.change_distribution_spread = change_distribution_spread_cumsum(bin_centers, weights)
        fit.change_distribution_mean_std = change_distribution_mean_and_std(bin_centers, weights)

def calc_bootstrap_change_distribution(shape, theta_samples, bin_edges):
    bin_centers = bin_edges_to_centers(bin_edges)
    n_params, n_samples = theta_samples.shape
    weights = np.zeros(bin_centers.shape)
    for i in xrange(n_samples):
        weights += calc_change_distribution(shape, theta_samples[:,i], bin_edges)
    weights /= n_samples # now values are in fraction of total change (doesn't have to sum up to 1 if ages don't cover the whole transition range)
    return weights

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

def export_timing_info_for_all_fits(data, fitter, fits):
    change_dist = compute_timing_info_for_all_fits(data, fitter, fits)
    README = """\
mu:
The mean age of the change distribution for given gene and region.
Dimensions: <n-genes> X <n-regions>

std:
The standard deviation of the change distribution for given gene and region.
Dimensions: <n-genes> X <n-regions>

genes: 
Gene names for the genes represented in other arrays

weights:
The change distributions for each gene and region.
Dimensions: <n-genes> X <n-regions> X <n-bins>

bin_centers:
The ages for the center of each bin used in calculating the histogram in "weights".
Dimensions: <n-bins> X 1

bin_edges:
The edges of the bins used in calculating the change histogram.
(centers can be calculated from the bin_edges, but it's convenient to have it pre-calculated)
Dimensions: <n-bins + 1> X 1

regions: 
Region names for the regions represented in other arrays

age_scaler: 
The scaling used for ages (i.e. 'log' means x' = log(x + 38/52))
"""
    mdict = dict(
        README_CHANGE_DISTRIBUTIONS = README,
        genes = list_of_strings_to_matlab_cell_array(change_dist.genes),
        regions = list_of_strings_to_matlab_cell_array(change_dist.regions),
        age_scaler = scalers.unify(change_dist.age_scaler).cache_name(),
        mu = change_dist.mu,
        std = change_dist.std,
        bin_edges = change_dist.bin_edges,
        bin_centers = change_dist.bin_centers,
        weights = change_dist.weights,
    )
    filename = join(cache_dir(), fit_results_relative_path(data,fitter) + '-change-dist.mat')
    save_matfile(mdict, filename)

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
