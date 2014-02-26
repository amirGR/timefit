import setup
from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from plots import *
from all_fits import *
import config as cfg

# load some data
data = load_data()   
series = data.get_one_series('HTR2B','VFC')
x = series.ages
y = series.expression
#fits = get_all_fits(data)

#plot_gene(data,0)