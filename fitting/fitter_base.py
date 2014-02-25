import config as cfg
import numpy as np
from sklearn.cross_validation import LeaveOneOut

class FitterBase(object):
    """Base class for objects that fit different regression models to data.
       In all methods, the following variables are used:
           x - x-values for the data points
           y - y-values for the data points
           theta - array/tuple/list of parameters for f(x)
           sigma - std of normal additive noise around the deterministic f(x)
           preds - predicted y values
    
       Derived classes should implement:
           str = cache_name(self)
           y = f(theta,x)
           theta, sigma = fit_simple(x,y)
    """
    def name(self):
        return self.cache_name()

    def high_res_preds(self, x, theta):
        x_smooth = np.linspace(x.min(),x.max(),cfg.n_curve_points_to_plot)
        y_smooth = self.f(theta, x_smooth)
        return x_smooth,y_smooth

    def fit_loo(self, x, y):
        n = len(y)
        test_preds = np.empty(n)
        for train,test in LeaveOneOut(n):
            theta,sigma = self.fit_simple(x[train],y[train])
            if theta is None:
                test_preds[test] = np.nan
            else:
                test_preds[test] = self.f(theta,x[test])
        return test_preds
