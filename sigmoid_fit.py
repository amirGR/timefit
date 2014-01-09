import numpy as np
from scipy.optimize import minimize

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
    res = minimize(f_error, theta0, args=(ages,expr), method='CG')
    return res.x
