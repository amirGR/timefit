import config as cfg
import numpy as np
from scipy.optimize import minimize
from minimization import minimize_with_restarts
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
        
    def fit_simple(self,x,y):
        rng = np.random.RandomState(cfg.random_seed)
        P0_base = np.array(self._get_theta_guess(x,y) + [1])
        def get_P0():
            return P0_base + rng.normal(0,1,size=P0_base.shape)
        def f_minimize(P0):
            return minimize(self._Err, P0, args=(x,y), method='BFGS', jac=self._Err_grad)
        P = minimize_with_restarts(f_minimize, get_P0)
        if P is None:
            return None,None
        assert not np.isnan(P).any()
        theta = P[:-1]
        sigma = 1/P[-1]
        return theta,sigma    

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

    def _Err(self,P,x,y):
        theta,p = P[:-1],P[-1]
        diffs = self.f(theta,x) - y
        residuals_term = 0.5 * p**2 * sum(diffs**2)
        n = len(y)
        return -n*np.log(p) + residuals_term -self._log_prob_theta(theta)
        
    def _Err_grad(self,P,x,y):
        theta,p = P[:-1],P[-1]
        n = len(y)
        diffs = self.f(theta,x) - y
        d_theta = np.array([p**2 * sum(diffs*d) for d in self._f_grad(theta,x)]) - self._d_theta_prior(theta)
        d_p = -n/p + p*sum(diffs**2)
        return np.r_[d_theta, d_p]
