import setup
from os.path import join
import project_dirs
from all_fits import Bunch, convert_format

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

filename = join(project_dirs.cache_dir(), 'kang2011', 'fits-serotonin-poly1-t0-s0.pkl')
convert_format(filename, f_convert)
