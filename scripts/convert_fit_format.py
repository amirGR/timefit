import setup
from sklearn.datasets.base import Bunch
from all_fits import Bunch, convert_format, _cache_file_from_fields
from fitter import Fitter
from shapes.sigmoid import Sigmoid

def f_convert(fit):
    "added fitter and shape params"
    return Bunch(
        fitter = fit.fitter,
        seed = fit.seed,
        theta = fit.theta,
        sigma = fit.sigma,
        fit_predictions = fit.fit_predictions,
        LOO_predictions = fit.LOO_predictions,
    )

filename = _cache_file_from_fields('serotonin', 'kang2011', False, 'sigmoid-t1-s0')
convert_format(filename, f_convert)
