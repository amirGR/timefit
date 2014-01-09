from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sigmoid_fit import *
from plots import *

def draw_with_fit(series):
    theta = fit_sigmoid(series)
    fit = sigmoid(theta,series.ages)
    plot_one_fit(series,fit)

data = load_data()   
series = data.get_one_series(0,0)

#draw_with_fit(series)