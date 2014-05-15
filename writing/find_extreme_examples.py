import setup
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits, iterate_fits
from scalers import LogScaler

cfg.verbosity = 1
age_scaler = LogScaler()
pathway = 'serotonin'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
fitter = Fitter(Sigmoid(priors=None))
fits = get_all_fits(data,fitter)

extreme = [(g,r) for dsname,g,r,fit in iterate_fits(fits, R2_threshold=0.5, return_keys=True) if abs(fit.theta[0]) > 100]
