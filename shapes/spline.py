import numpy as np
from scipy.interpolate import UnivariateSpline
import config as cfg
from shape import Shape

class Spline(Shape):
    def __init__(self):
        Shape.__init__(self, priors=None)
        
    def param_names(self, latex=False):
        return ['spline-params']
        
    def format_params(self, theta, latex=False):
        return ''
                
    def cache_name(self):
        return 'spline'

    def f(self,theta,x):
        spline = theta[0]
        return spline(x)

    def fit(self,x,y):
        inds = np.argsort(x)
        x = x[inds]
        y = y[inds]
        
        sig = self._estimate_std(y)
        w = 1.0 / sig
        for i in xrange(3):
            spline = UnivariateSpline(x, y, w=w*np.ones(x.shape), s=None)
            preds = spline(x)
            if not np.isnan(preds).any():
                break
            if cfg.verbosity > 1:
                print 'retrying spline fit with w /= 2'
            w = w/2
        else:
            raise Exception('Failed to fit spline')
        #spline = UnivariateSpline(x, y)

        return [spline]
    
    @staticmethod
    def _estimate_std(y):
        k = 10 # window size for std estimation
        s = [np.std(y[i:i+k]) for i in xrange(len(y)-k+1)]
        return np.mean(s)
