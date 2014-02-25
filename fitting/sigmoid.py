import numpy as np
from scipy.optimize import minimize
import config as cfg
from minimization import minimize_with_restarts
from fitting.fitter_base import FitterBase

class Sigmoid(FitterBase):
    def cache_name(self):
        return 'sigmoid'

    def f(self,theta,x):
        a,h,mu,w = theta
        return a + h/(1+np.exp(-(x-mu)/w))

    def fit_simple(self,x,y):
        rng = np.random.RandomState(cfg.random_seed)
        P0_base = np.array([
            y.min(), # a
            y.max()-y.min(), # h
            (x.min() + x.max()) / 2, # mu
            (x.max() - x.min()) / 2, # w
            1, # p
        ])
        def get_P0():
            return P0_base + rng.normal(0,1,size=5)
        def f_minimize(P0):
            return minimize(self._Err, P0, args=(x,y), method='BFGS', jac=self._Err_grad)
        P = minimize_with_restarts(f_minimize, get_P0)
        if P is None:
            return None,None
        assert not np.isnan(P).any()
        theta = P[:-1]
        sigma = 1/P[-1]
        return theta,sigma
    
    def _f_grad(self,theta,x):
        a,h,mu,w = theta
        e = np.exp(-(x-mu)/w)
        d_a = 1
        d_h = 1/(1+e)
        d_mu = -h/(1+e)**2 * e/w
        d_w = -h/(1+e)**2 * e * (x-mu)/w**2
        return [d_a, d_h, d_mu, d_w]
    
    def _Err(self,P,x,y):
        theta,p = P[:-1],P[-1]
        squares = (self.f(theta,x) - y)**2
        n = len(y)
        prior_z = (theta - cfg.theta_prior_mean) / cfg.theta_prior_sigma
        return -n*np.log(p) + 0.5*p**2*sum(squares) + 0.5*sum(prior_z ** 2)
    
    def _Err_grad(self,P,x,y):
        theta,p = P[:-1],P[-1]
        n = len(y)
        diffs = self.f(theta,x) - y
        d_a, d_h, d_mu, d_w = self._f_grad(theta,x)
        d_prior = (theta - cfg.theta_prior_mean) / cfg.theta_prior_sigma**2
        d_a = sum(diffs * d_a)
        d_h = sum(diffs * d_h)
        d_mu = sum(diffs * d_mu)
        d_w = sum(diffs * d_w)
        d_theta = p**2 * np.array([d_a, d_h, d_mu, d_w]) + d_prior
        d_s = -n/p + p*sum(diffs**2)
        return np.r_[d_theta, d_s]

def check_grad(n=100):
    import scipy.optimize
    rng = np.random.RandomState(0)
    fit = Sigmoid()
    def check_one():
        x = np.arange(-10,11)
        y = fit.f([-1,3,2,2],x) + rng.normal(size=x.shape)
        a,b,c,d = rng.uniform(size=4)
        s = np.e
        P = [a, a+b, c, d, s]
#        g1 = f_error_gradient(p,x,y)
#        g2 = scipy.optimize.approx_fprime(P, f_error, 1E-8, x, y)
#        print 'g1 = {}'.format(g1)
#        print 'g2 = {}'.format(g2)
#        print 'diff = {}'.format(np.abs(g1-g2))
        diff = scipy.optimize.check_grad(fit._Err, fit._Err_grad, P, x, y)
        return diff
    max_diff = max([check_one() for _ in xrange(n)])
    print 'Max difference over {} iterations: {}'.format(n,max_diff)
    if max_diff < 1E-4:
        print 'Gradient is OK'
    else:
        print 'Difference is too big. Gradient is NOT OK!'

if __name__ == '__main__':
    check_grad()
