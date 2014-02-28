import numpy as np
import config as cfg
from shape import Shape

class Poly(Shape):
    def __init__(self,n):
        self.n = n
        
    def cache_name(self):
        return 'poly{}'.format(self.n)

    def format_params(self, theta, latex=False):
        if latex:
            return ', '.join('$a_{}$={:.2f}'.format(j,a) for j,a in enumerate(theta))
        else:
            return ', '.join('a{}={:.2f}'.format(j,a) for j,a in enumerate(theta))

    def f(self,theta,x):
        powers = [x**j for j in xrange(self.n+1)]
        return np.dot(theta,powers)

    def f_grad(self,theta,x):
        return [x**j for j in xrange(self.n+1)]
    
    def get_theta_guess(self,x,y):
        return [y.mean()] + self.n*[0]
    
    def log_prob_theta(self, theta):
        mu = cfg.poly_theta_prior_mean[:self.n+1]
        sigma = cfg.poly_theta_prior_sigma[:self.n+1]
        z = (theta - mu) / sigma
        return -0.5*sum(z ** 2 )
        
    def d_theta_prior(self, theta):
        mu = cfg.poly_theta_prior_mean[:self.n+1]
        sigma = cfg.poly_theta_prior_sigma[:self.n+1]
        return -(theta - mu) / sigma**2

if __name__ == '__main__':
    thresholds = [1E-10, 1E-7, 1E-6, 1E-4]
    for n in xrange(4):
        print 'Testing polynomial of degree {}'.format(n)
        Poly(n).TEST_check_grad(theta_size=n+1, threshold=thresholds[n])
