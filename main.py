from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sigmoid_fit import *
from plots import *

def draw_with_fit(series, L=0, cv=True):
    theta = fit_sigmoid_simple(series.ages, series.expression, L)
    fits = {'Simple fit': sigmoid(theta,series.ages)}
    if cv:
        fits['LOO predictions'] = fit_sigmoid_loo(series.ages, series.expression, L)
    plot_one_series(series, fits)

def find_best_L(series):
    Ls = np.linspace(0,1,20)
    def score(L):
        print 'Computing score for L={}'.format(L)
        return get_fit_score(series.ages,series.expression,L)
    scores = array([score(L) for L in Ls])
    plot_L_scores(Ls,scores)
#    idx = scores.argmin()
#    L = L_options[idx]
#    scores = scores[idx]

data = load_data()   
series = data.get_one_series(0,0)

#draw_with_fit(series)
