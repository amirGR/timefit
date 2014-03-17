import numpy as np

class LogScaler(object):
    def __init__(self, x0=0):
        self.x0 = x0
        
    def cache_name(self):
        if self.x0 == 0:
            return 'log'
        else:
            return 'log(x0={:.2g})'.format(self.x0)
    
    def scale(self, x):
        return np.log(x - self.x0)

    def unscale(self, sx):
        return np.exp(sx) + self.x0

class LinearScaler(object):
    def __init__(self,mu,r):
        self.mu = mu
        self.r =  r

    @staticmethod
    def fromData(self,x):
        mu = np.mean(x)
        r =  max(x) - min(x)
        return LinearScaler(mu,r)
        
    def cache_name(self):
        return 'linear'
        
    def scale(self, x):
        return (x-self.mu) / self.r

    def unscale(self, sx):
        return sx*self.r + self.mu
