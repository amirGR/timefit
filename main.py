from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sigmoid_fit import *
from plots import *
from all_fits import *
import config as cfg

# setup
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.seterr(all='ignore') # Ignore numeric overflow/underflow etc. YYY - can/should we handle these warnings?

# load some data
data = load_data()   
series = data.get_one_series(0,0)
x = series.ages
y = series.expression
fits = get_all_fits(data)

#draw_with_fit(series)
#plot_gene(data,0)