import numpy as np
import config as cfg

def unify(scaler):
    if scaler is None:
        return NopScaler()
    else:
        return scaler

class NopScaler(object):
    def scale(self, x): return x
    def unscale(self, sx): return sx

class LogScaler(object):
    def __init__(self, x0=None):
        if x0 is None:
            x0 = cfg.log_scale_x0
        self.x0 = x0
        
    def cache_name(self):
        return 'log'
    
    def scale(self, x):
        return np.log(x - self.x0)

    def unscale(self, sx):
        return np.exp(sx) + self.x0

class LinearScaler(object):
    def __init__(self,mu,r):
        self.mu = mu
        self.r =  r

    @staticmethod
    def fromData(x):
        mu = np.mean(x)
        r =  max(x) - min(x)
        return LinearScaler(mu,r)
        
    def cache_name(self):
        return 'linear'
        
    def scale(self, x):
        return (x-self.mu) / self.r

    def unscale(self, sx):
        return sx*self.r + self.mu

##########################################################
# Building scaler from command line input
##########################################################
def allowed_scaler_names():
    return ['linear', 'log']

def build_scaler(name, data):
    if name == 'linear':
        return LinearScaler.fromData(data.ages)
    elif name == 'log':
        return LogScaler()
    else:
        raise Exception('Unknown scaler: {}'.format(name))
