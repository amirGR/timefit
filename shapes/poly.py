import numpy as np
from shape import Shape

class Poly(Shape):
    def __init__(self,n,priors=None):
        self.n = n
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

    def adjust_for_scaling(self, theta, sx, sy):
        # handle x scaling
        A,B = sx
        t = theta
        theta = np.zeros(t.shape)
        if self.n >= 0:
            theta[0] = t[0]
        if self.n >= 1:
            theta[0] = theta[0] - A*B*t[1]
            theta[1] = theta[1] + A*t[1]
        if self.n >= 2:
            theta[0] = theta[0] + A**2 * B**2 * t[2]
            theta[1] = theta[1] - 2 * A**2 * B * t[2]
            theta[2] = theta[2] + A**2 * t[2]
        if self.n >= 3:
            theta[0] = theta[0] - A**3 * B**3 * t[3]
            theta[1] = theta[1] + 3 * A**3 * B**2 * t[3]
            theta[2] = theta[2] - 3 * A**3 * B * t[3]
            theta[3] = theta[3] + A**3 * t[3]            
        if self.n >= 4:
            raise AssertionError('adjust_for_scaling not supported for polynomials above degree 3')

        # handle y scaling
        C,D = sy
        theta = theta/C
        theta[0] = theta[0] + D

        return theta
    
if __name__ == '__main__':
    thresholds = [1E-10, 1E-7, 1E-6, 1E-4]
    for n in xrange(4):
        print 'Testing polynomial of degree {}'.format(n)
        Poly(n).TEST_check_grad(threshold=thresholds[n])
