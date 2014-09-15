import numpy as np
from sklearn.datasets.base import Bunch
from all_fits import iterate_fits

def aggregate_change_distribution(fits, R2_threshold=None, b_normalize=False):
    bin_centers = fits.change_distribution_params.bin_centers
    for i,fit in enumerate(iterate_fits(fits, R2_threshold=R2_threshold)):
        if i == 0:
            weights = fit.change_distribution_weights
        else:
            weights =+ fit.change_distribution_weights
    n_fits = i + 1
    weights /= n_fits
    if b_normalize:
        weights /= sum(weights)
    return bin_centers, weights, n_fits

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
            t = thetas[:,i]
            a,h,mu,w = t
            edge_vals = shape.f(t,bin_edges)
            changes = np.abs(edge_vals[1:] - edge_vals[:-1])
            # ignore change magnitude per gene - take only distribution of change times
            weights += changes / abs(h)
        weights /= n_samples # now values are in fraction of total change (doesn't have to sum up to 1 if ages don't cover the whole transition range)
        fit.change_distribution_weights = weights
    return fits

def change_distribution_width_std(bin_centers, weights):
    weights = np.array(weights, dtype=float) # in case it's a list
    weights /= np.sum(weights) # normalize to make it a PMF
    x0 = np.sum([x*w for x,w in zip(bin_centers,weights)])
    V = np.sum([w*(x-x0)**2 for x,w in zip(bin_centers,weights)])
    return np.sqrt(V)

def change_distribution_width_cumsum(bin_centers, weights, threshold=0.8):
    weights = np.array(weights, dtype=float) # in case it's a list
    weights /= np.sum(weights) # normalize to make it a PMF
    bin_width = bin_centers[1] - bin_centers[0] # we assume uniform bins here
    s = np.cumsum(weights)
    i_from = np.argmax(s > 0.5 - threshold/2.0)
    i_to = np.argmax(s > 0.5 + threshold/2.0)
    width = bin_centers[i_to] - bin_centers[i_from] + bin_width
    return width
