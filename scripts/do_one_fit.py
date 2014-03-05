import setup
import matplotlib.pyplot as plt
from load_data import load_data
from all_fits import compute_fit
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from shapes.poly import Poly
from plots import plot_one_series

import utils
utils.disable_all_warnings()

data = load_data()
series = data.get_one_series('HTR1E','VFC')
x = series.ages
y = series.expression

def do_fit(shape=None, theta_prior=True, sigma_prior=True):
    if shape is None: 
        shape = Sigmoid()
    fitter = Fitter(shape,theta_prior,sigma_prior)
    fit = compute_fit(series,fitter)
    plot_one_series(series,fit=fit)
    P_ttl = fitter.format_params(fit.theta, fit.sigma, latex=True)
    plt.title('theta_prior={}, simga_prior={}\n{}'.format(theta_prior, sigma_prior,P_ttl))

sigmoid = Sigmoid()
do_fit(sigmoid, theta_prior=True, sigma_prior=True)