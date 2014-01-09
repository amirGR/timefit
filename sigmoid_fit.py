import numpy as np
from scipy.optimize import minimize
import config as cfg

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

def f_error(theta,x,y,L):
    a,h,mu,w = theta
    squares = (sigmoid(theta,x) - y)**2
    log_p0 = -0.5*((w-cfg.w0)/cfg.w0_sigma)**2;
    return 0.5*sum(squares) - L*log_p0

def f_error_gradient(theta,x,y,L):
    a,h,mu,w = theta
    diffs = sigmoid(theta,x) - y
    d_a, d_h, d_mu, d_w = sigmoid_grad(theta,x)
    d_a = sum(diffs * d_a)
    d_h = sum(diffs * d_h)
    d_mu = sum(diffs * d_mu)
    d_w = sum(diffs * d_w) + L*(w-cfg.w0)/cfg.w0_sigma**2
    return np.array([d_a, d_h, d_mu, d_w])

def fit_sigmoid(series,L):
    expr = series.expression
    ages = series.ages
    theta0 = np.array([
        expr.min(), # a
        expr.max(), # h
        (ages.min() + ages.max()) / 2, # mu
        (ages.max() - ages.min()) / 2, # w
    ])
    res = minimize(f_error, theta0, args=(ages,expr,L), method='BFGS', jac=f_error_gradient)
    return res.x
    
def check_grad(series, n=10):
    for i in xrange(n):
        a,b,c,d = np.random.uniform(size=4)
        theta = [a, a+b, c, d]
        x = series.ages[0:1]
        y = series.expression[0:1]
        L = 3
        diff = sp.optimize.check_grad(f_error, f_error_gradient, theta, x, y, L)
        print '{}) theta={}, diff={}'.format(i,theta,diff)

