from load_data import load_data
import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt

data = load_data()

def plot_gene(iGene):
    fig = plt.figure()
    for iRegion in range(len(data.region_names)):
        series = data.get_one_series(iGene,iRegion)
        ax = fig.add_subplot(4,4,iRegion+1)
        ax.plot(series.ages,series.expression,'ro')
        ax.plot(series.ages,series.expression,'b-')
        ax.set_title('Region {}'.format(series.gene_name))
        if iRegion % 4 == 0:
            ax.set_ylabel('Expression Level')
        if iRegion / 4 >= 3:
            ax.set_xlabel('Age [years]')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(series.gene_name))
    
def plot_one_fit(series,theta=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.ages,series.expression,'ro')
    ax.plot(series.ages,series.expression,'b-')
    if theta is not None:
        fit = sigmoid(theta,series.ages)
        ax.plot(series.ages,fit,'g-')
    ax.set_title('Region {}'.format(series.gene_name))
    ax.set_ylabel('Expression Level')
    ax.set_xlabel('Age [years]')
    
def sigmoid(theta,x):
    a,h,mu,w = theta
    return a + h/(1+np.exp(-(x-mu)/w))

def f_error(theta,x,y):
    squares = (sigmoid(theta,x) - y)**2
    return 0.5*sum(squares)

def fit_sigmoid(series):
    expr = series.expression
    ages = series.ages
    theta0 = np.array([
        expr.min(), # a
        expr.max(), # h
        (ages.min() + ages.max()) / 2, # mu
        (ages.max() - ages.min()) / 2, # w
    ])
    res = optimize.minimize(f_error, theta0, args=(ages,expr), method='Nelder-Mead')
    return res.x
