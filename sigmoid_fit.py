import numpy as np
from scipy.optimize import minimize

def sigmoid(theta,x):
    a,h,mu,w = theta
    return a + h/(1+np.exp(-(x-mu)/w))
    
def sigmoid_grad(theta,x):
    a,h,mu,w = theta
    e = np.exp(-(x-mu)/w)
    d_a = 1
    d_h = 1/(1+e)
    d_mu = -h/(1+e)**2 * e/w
    d_w = -h/(1+e)**2 * e * (x-mu)/w**2
    return [d_a, d_h, d_mu, d_w]

def f_error(theta,x,y):
    squares = (sigmoid(theta,x) - y)**2
    return 0.5*sum(squares)

def f_error_gradient(theta,x,y):
    diffs = sigmoid(theta,x) - y
    d_a, d_h, d_mu, d_w = sigmoid_grad(theta,x)
    d_a = sum(diffs * d_a)
    d_h = sum(diffs * d_h)
    d_mu = sum(diffs * d_mu)
    d_w = sum(diffs * d_w)
    return np.array([d_a, d_h, d_mu, d_w])

def fit_sigmoid(series):
    expr = series.expression
    ages = series.ages
    theta0 = np.array([
        expr.min(), # a
        expr.max(), # h
        (ages.min() + ages.max()) / 2, # mu
        (ages.max() - ages.min()) / 2, # w
    ])
    res = minimize(f_error, theta0, args=(ages,expr), method='BFGS', jac=f_error_gradient)
    return res.x
