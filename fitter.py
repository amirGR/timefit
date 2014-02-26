import config as cfg
import numpy as np
from scipy.optimize import minimize
from minimization import minimize_with_restarts
from sklearn.cross_validation import LeaveOneOut

class Fitter(object):
    def __init__(self, shape):
        self.shape = shape
        
    def cache_name(self):
        return self.shape.cache_name()
        
    def fit_simple(self,x,y):
        rng = np.random.RandomState(cfg.random_seed)
        P0_base = np.array(self.shape.get_theta_guess(x,y) + [1])
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
                test_preds[test] = self.predict(theta,x[test])
        return test_preds
        
    def predict(self, theta, x):
        return self.shape.f(theta,x)

    ##########################################################
    # Private methods
    ##########################################################
 
    def _Err(self,P,x,y):
        theta,p = P[:-1],P[-1]
        diffs = self.shape.f(theta,x) - y
        residuals_term = 0.5 * p**2 * sum(diffs**2)
        n = len(y)
        return -n*np.log(p) + residuals_term -self.shape.log_prob_theta(theta)
        
    def _Err_grad(self,P,x,y):
        theta,p = P[:-1],P[-1]
        n = len(y)
        diffs = self.shape.f(theta,x) - y
        d_theta = np.array([p**2 * sum(diffs*d) for d in self.shape.f_grad(theta,x)]) - self.shape.d_theta_prior(theta)
        d_p = -n/p + p*sum(diffs**2)
        return np.r_[d_theta, d_p]

def check_grad(n=100):
    import scipy.optimize
    from shapes.sigmoid import Sigmoid
    rng = np.random.RandomState(0)
    fitter = Fitter(Sigmoid())
    def check_one():
        x = np.arange(-10,11)
        y = fitter.predict([-1,3,2,2],x) + rng.normal(size=x.shape)
        a,b,c,d = rng.uniform(size=4)
        p = np.e
        P = [a, a+b, c, d, p]
        diff = scipy.optimize.check_grad(fitter._Err, fitter._Err_grad, P, x, y)
        return diff
    max_diff = max([check_one() for _ in xrange(n)])
    print 'Max difference over {} iterations: {}'.format(n,max_diff)
    if max_diff < 1E-4:
        print 'Gradient is OK'
    else:
        print 'Difference is too big. Gradient is NOT OK!'

if __name__ == '__main__':
    check_grad()
