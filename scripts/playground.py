import setup
from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from plots import *
from all_fits import *
import config as cfg
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from shapes.spline import Spline
from scalers import LogScaler

cfg.verbosity = 1
data = load_data(pathway='serotonin', scaler=LogScaler(cfg.kang_log_scale_x0))   
series = data.get_one_series('HTR1A','MD')
x = series.ages
y = series.expression
fitter = Fitter(Spline())
theta, sigma, LOO_predictions = fitter.fit(x,y)
spline = theta[0]
preds = spline(x)
print preds
