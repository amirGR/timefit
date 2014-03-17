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

# load some data
data = load_data()   
series = data.get_one_series('HTR2B','VFC')
x = series.ages
y = series.expression
fitter = Fitter(Sigmoid(),False,False)
#fits = get_all_fits(data,fitter)

#plot_gene(data,0)