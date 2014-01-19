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


def draw_with_fit(series, L=0, cv=True):
    x = series.ages
    y = series.expression
    theta = fit_sigmoid_simple(x,y,L)
    fit = sigmoid(theta,series.ages)
    fit_label = 'Simple fit ({}={:.3f})'.format(cfg.score_type, cfg.score(y,fit))
    fits = {fit_label : fit}
    if cv:
        preds = fit_sigmoid_loo(x,y,L)
        loo_label = 'LOO predictions ({}={:.3f})'.format(cfg.score_type, cfg.score(y,preds))
        fits[loo_label] = preds
    more_title = r'$\lambda={:.3g}$'.format(L)
    plot_one_series(series, fits, more_title)

def find_best_L(series, Ls=None):
    if Ls is None:
        Ls = np.logspace(-4,3,20)
    def score(L):
        print 'Computing score for L={}'.format(L)
        preds = fit_sigmoid_loo(series.ages,series.expression,L)
        return cfg.score(series.expression, preds)
    scores = array([score(L) for L in Ls])
    plot_L_scores(Ls,scores)

data = load_data()   
series = data.get_one_series(0,0)

#cfg.n_optimization_restarts = 100
#find_best_L(series, Ls=np.logspace(-3,-1,20))

#draw_with_fit(series)
