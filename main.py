from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sigmoid_fit import *
from plots import *
import config as cfg

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.seterr(all='ignore') # Ignore numeric overflow/underflow etc. YYY - can/should we handle these warnings?

def draw_with_fit(series, cv=True):
    x = series.ages
    y = series.expression
    P = fit_sigmoid_simple(x,y)
    fit = sigmoid(P[:-1],series.ages)
    fit_label = 'Simple fit ({}={:.3f})'.format(cfg.score_type, cfg.score(y,fit))
    fits = {fit_label : fit}
    if cv:
        preds = fit_sigmoid_loo(x,y)
        loo_label = 'LOO predictions ({}={:.3f})'.format(cfg.score_type, cfg.score(y,preds))
        fits[loo_label] = preds        
    plot_one_series(series, fits)

data = load_data()   
series = data.get_one_series(0,0)
x = series.ages
y = series.expression

draw_with_fit(series)
