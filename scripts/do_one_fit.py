import setup
import config as cfg
from load_data import GeneData
from all_fits import compute_fit
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from scalers import LogScaler
from plots import plot_one_series

import utils
utils.disable_all_warnings()

data = GeneData.load('kang2011').restrict_pathway('serotonin').restrict_postnatal()
data.scale_ages(LogScaler(cfg.kang_log_scale_x0))
series = data.get_one_series('HTR1E','VFC')
shape = Sigmoid()
fitter = Fitter(shape, use_theta_prior=False, use_sigma_prior=False)
fit = compute_fit(series,fitter)
plot_one_series(series,fit=fit)
