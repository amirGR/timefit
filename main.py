from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sigmoid_fit import *
from plots import *

data = load_data()

def draw_with_fit(series):
    theta = fit_sigmoid(series)
    fit = sigmoid(theta,series.ages)
    plot_one_fit(series,fit)