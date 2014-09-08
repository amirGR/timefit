import numpy as np
from sklearn.datasets.base import Bunch
from all_fits import iterate_fits

def aggregate_change_distribution(fits, R2_threshold=None, b_normalize=False):
    for i,fit in enumerate(iterate_fits(fits, R2_threshold=R2_threshold)):
        hist = fit.change_histogram
        if i == 0:
            bin_edges = hist.bin_edges
            change_vals = hist.change_vals
        else:
            assert (hist.bin_edges == bin_edges).all()
            change_vals =+ hist.change_vals
    n_fits = i + 1
    change_vals /= n_fits
    if b_normalize:
        change_vals /= sum(change_vals)
    return bin_edges, change_vals, n_fits

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

    for dsname,g,r,fit in iterate_fits(fits, return_keys=True):
        thetas = fit.theta_samples # bootstrap samples of theta values
        n_params, n_samples = thetas.shape
        bin_edges, bin_size = np.linspace(from_age, to_age, n_bins+1, retstep=True)
        change_vals = np.zeros(n_bins)
        for i in xrange(n_samples):
            t = thetas[:,i]
            a,h,mu,w = t
            edge_vals = shape.f(t,bin_edges)
            changes = np.abs(edge_vals[1:] - edge_vals[:-1])
            # ignore change magnitude per gene - take only distribution of change times
            change_vals += changes / abs(h)
        change_vals /= n_samples # now values are in fraction of total change (doesn't have to sum up to 1 if ages don't cover the whole transition range)
        fit.change_histogram = Bunch(
            bin_edges = bin_edges,
            change_vals = change_vals,
        )
    return fits
