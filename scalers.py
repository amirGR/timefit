import numpy as np
import config as cfg

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
    return ['none', 'linear', 'log']

def build_scaler(name, data):
    if name == 'none':
        return None
    elif name == 'linear':
        return LinearScaler.fromData(data.ages)
    elif name == 'log':
        if data.dataset == 'kang2011':
            x0 = cfg.kang_log_scale_x0
        elif data.dataset == 'colantuoni2011':
            x0 = cfg.colantuoni_log_scale_x0
        else:
            x0 = cfg.log_scale_x0
            print "Don't know where x0 is for this dataset. Using default: x0={}".format(x0)
        return LogScaler(x0)
    else:
        raise Exception('Unknown scaler: {}'.format(name))
