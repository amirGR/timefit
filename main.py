from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sigmoid_fit import *
from plots import *

def draw_with_fit(series,L=0):
    loo_preds = fit_sigmoid_loo(series.ages, series.expression, L)
    theta = fit_sigmoid_simple(series.ages, series.expression, L)
    fit = sigmoid(theta,series.ages)
    plot_one_series(series, fits = {'Simple fit':fit, 'LOO predictions': loo_preds})

data = load_data()   
series = data.get_one_series(0,0)

#draw_with_fit(series)
