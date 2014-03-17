import numpy as np
from shape import Shape
from priors import NormalPrior, GammaPrior

class Sigmoid(Shape):
    def __init__(self, priors=None):
        if priors is None: # XXX - this set is fitted specifically for kang2011/serotonin
            priors = [
                GammaPrior(alpha=1.87, beta=1.135, mu=3.65), # a
                NormalPrior(mu=0.28, sigma=0.84), # h
                GammaPrior(alpha=1.41, beta=0.055, mu=-4.31), # mu    
                #GammaPrior(alpha=0.25, beta=0.15, mu=0), # w
                NormalPrior(mu=2.5, sigma=2.5), # w
            ]
        Shape.__init__(self, priors)
        
    def n_params(self):
        return 4
                
    def cache_name(self):
        return 'sigmoid'

    def format_params(self, theta, latex=False):
        a,h,mu,w = theta
        if latex:
            return r'a={a:.2f}, h={h:.2f}, $\mu$={mu:.2f}, w={w:.2f}'.format(**locals())
        else:
            return r'a={a:.2f}, h={h:.2f}, mu={mu:.2f}, w={w:.2f}'.format(**locals())

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
