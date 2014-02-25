import numpy as np
import config as cfg
from fitting.fitter_base import FitterBase

class Sigmoid(FitterBase):
    def cache_name(self):
        return 'sigmoid'

    def f(self,theta,x):
        a,h,mu,w = theta
        return a + h/(1+np.exp(-(x-mu)/w))

    def _get_theta_guess(self,x,y):
        return [
            y.min(), # a
            y.max()-y.min(), # h
            (x.min() + x.max()) / 2, # mu
            (x.max() - x.min()) / 2, # w
        ]
    
    def _f_grad(self,theta,x):
        a,h,mu,w = theta
        e = np.exp(-(x-mu)/w)
        d_a = 1
        d_h = 1/(1+e)
        d_mu = -h/(1+e)**2 * e/w
        d_w = -h/(1+e)**2 * e * (x-mu)/w**2
        return [d_a, d_h, d_mu, d_w]
    
    def _log_prob_theta(self, theta):
        z = (theta - cfg.sigmoid_theta_prior_mean) / cfg.sigmoid_theta_prior_sigma
        return -0.5*sum(z ** 2 )
        
    def _d_theta_prior(self, theta):
        return -(theta - cfg.sigmoid_theta_prior_mean) / cfg.sigmoid_theta_prior_sigma**2
    
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
