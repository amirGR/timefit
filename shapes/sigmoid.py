import numpy as np
from shape import Shape

class Sigmoid(Shape):
    def __init__(self, priors=None):
        Shape.__init__(self, priors)
        
    def param_names(self, latex=False):
        if latex:
            return ['$Baseline$', '$Height$', '$Onset$', '$Width$']
        else:
            return ['baseline', 'height', 'onset', 'width']
                
    def cache_name(self):
        return 'sigmoid'

    def f(self,theta,x):
        a,h,mu,w = theta
        return a + h/(1+np.exp(-(x-mu)/w))

    def f_grad(self,theta,x):
        a,h,mu,w = theta
        e = np.exp(-(x-mu)/w)
        ie = np.exp((x-mu)/w)
        d_a = 1
        d_h = 1/(1+e)
        d_mu = -h/(w*(1+e)*(1+ie))
        d_w = -h*(x-mu)/(w**2 * (1+e) * (1+ie))
        return [d_a, d_h, d_mu, d_w]
    
    def get_theta_guess(self,x,y):
        return [
            y.min(), # a
            y.max()-y.min(), # h
            (x.min() + x.max()) / 2, # mu
            (x.max() - x.min()) / 2, # w
        ]
    
if __name__ == '__main__':
    Sigmoid().TEST_check_grad()
