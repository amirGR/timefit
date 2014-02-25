import setup
from sklearn.datasets.base import Bunch
from all_fits import Bunch, convert_format, _cache_file

def f_convert(fit):
    "P changed to theta,sigma"
    return Bunch(
        seed = fit.seed,
        theta = fit.P[:-1],
        sigma = 1/fit.P[-1],
        fit_predictions = fit.fit_predictions,
        LOO_predictions = fit.LOO_predictions,
    )

filename = _cache_file('serotonin', 'kang2011', 'sigmoid')
convert_format(filename, f_convert)
