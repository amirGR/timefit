import numpy as np
from shape import Shape
from priors import NormalPrior, GammaPrior

class Sigmoid(Shape):
    def __init__(self, priors=None):
        if priors is None: # XXX - this set is fitted specifically for kang2011/serotonin
            # backward compatibility - same normal distribution as before the change:
            priors = [
                NormalPrior(mu=5, sigma=5), # a
                NormalPrior(mu=5, sigma=5), # h
                NormalPrior(mu=30, sigma=30), # mu
                NormalPrior(mu=2.5, sigma=2.5), # w
            ]
            
# This set of priors is modeled after the empirical distribution of the parameters:
#            priors = [
#                GammaPrior(alpha=1.87, beta=0.88, mu=3.65), # a
#                NormalPrior(mu=0.27, sigma=0.79), # h
#                GammaPrior(alpha=1.4, beta=18.15, mu=-4.27), # mu
#                GammaPrior(alpha=0.27, beta=6.55, mu=0), # w
#            ]
        
# This is the set we're trying to actually get to work
# The gamma distribution on 'w' is unworkable (gives very small w all the time)
# The gamma distribution on 'mu' doesn't work for some reason, even though the 
#     range is very compatible with the fit in do_one_fit.py, but it doesn't 
#     converge to the right solution. Need to debug.
#            priors = [
#                GammaPrior(alpha=1.87, beta=0.88, mu=3.65), # a
#                NormalPrior(mu=0.27, sigma=0.79), # h
#                
#                #GammaPrior(alpha=1.4, beta=18.15, mu=-4.27), # mu
#                GammaPrior(alpha=1.4, beta=18.15, mu=-4.27), # mu
#                #NormalPrior(mu=0, sigma=80), # mu
#                
#                #GammaPrior(alpha=0.27, beta=6.55, mu=0), w
#                NormalPrior(mu=5, sigma=5), # w
#            ]
        
        Shape.__init__(self, priors)
        
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
    Sigmoid().TEST_check_grad(theta_size=4)
