import numpy as np
from shape import Shape
from priors import NormalPrior

class Poly(Shape):
    def __init__(self,n):
        self.n = n
        priors = [NormalPrior(0,1) for _ in range(n+1)] # placeholder
        Shape.__init__(self, priors)
        
    def param_names(self, latex=False):
        if latex:
            return ['$a_{}$'.format(j) for j in xrange(self.n+1)]
        else:
            return ['a{}'.format(j) for j in xrange(self.n+1)]
        
    def cache_name(self):
        return 'poly{}'.format(self.n)

    def f(self,theta,x):
        powers = [x**j for j in xrange(self.n+1)]
        return np.dot(theta,powers)

    def f_grad(self,theta,x):
        return [x**j for j in xrange(self.n+1)]
    
    def get_theta_guess(self,x,y):
        return [y.mean()] + self.n*[0]

if __name__ == '__main__':
    thresholds = [1E-10, 1E-7, 1E-6, 1E-4]
    for n in xrange(4):
        print 'Testing polynomial of degree {}'.format(n)
        Poly(n).TEST_check_grad(threshold=thresholds[n])
