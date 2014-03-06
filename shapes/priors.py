import numpy as np
from scipy import stats

class NormalPrior(object):
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
        
    def __repr__(self):
        return 'NormalPrior(mu={:.2f}, sigma={:.2f})'.format(self.mu, self.sigma)
        
    @property
    def rv(self):
        return stats.norm(self.mu, self.sigma)

    def generate(self):
        return self.rv.rvs()
        
    def bounds(self):
        return (None,None)

    def log_prob(self, x):
        z = (x - self.mu) / self.sigma
        return -0.5*(z**2)

    def d_log_prob(self, x):
        return -(x - self.mu) / (self.sigma**2)

class GammaPrior(object):
    def __init__(self,alpha,beta,mu):
        self.a = alpha
        self.b = beta
        self.mu = mu
        
    def __repr__(self):
        return 'Gamma(a={:.2f}, b={:.2f}, mu={:.2f})'.format(self.a,self.b,self.mu)

    @property
    def rv(self):
        return stats.gamma(self.a, loc=self.mu, scale=1/self.b)
        
    def generate(self):
        return self.rv.rvs()
        
    def bounds(self):
        return (self.mu,None)

    def log_prob(self, x):
        x = x - self.mu
        return (self.a-1)*np.log(x) - self.b*x
            
    def d_log_prob(self, x):
        x = x - self.mu
        return (self.a-1)/x - self.b
