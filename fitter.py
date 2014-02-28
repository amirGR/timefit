import config as cfg
import numpy as np
from scipy.optimize import minimize
from minimization import minimize_with_restarts
from sklearn.cross_validation import LeaveOneOut

class Fitter(object):
    def __init__(self, shape, use_theta_prior=True, use_sigma_prior=False):
        self.shape = shape
        self.use_theta_prior = use_theta_prior
        self.use_sigma_prior = use_sigma_prior
        
    def __str__(self):
        return 'Fitter({}, theta_prior={}, sigma_prior={})'.format(self.shape, self.use_theta_prior, self.use_sigma_prior)
        
    def cache_name(self):
        shape = self.shape.cache_name()
        theta = int(self.use_theta_prior)
        sigma = int(self.use_sigma_prior)
        return '{}-t{}-s{}'.format(shape,theta,sigma)
        
    def format_params(self, theta, sigma, latex=False):
        shape_params = self.shape.format_params(theta,latex)
        if latex:
            return r'{}, $\sigma$={:.2f}'.format(shape_params, sigma)
        else:
            return r'{}, sigma={:.2f}'.format(shape_params, sigma)

    def fit_simple(self,x,y):
        rng = np.random.RandomState(cfg.random_seed)
        P0_base = np.array(self.shape.get_theta_guess(x,y) + [1])
        def get_P0():
            return P0_base + rng.normal(0,1,size=P0_base.shape)
        def f_minimize(P0,i):
            if i % 2:
                method = 'BFGS'
            else:
                method = 'CG'
            return minimize(self._Err, P0, args=(x,y), method=method, jac=self._Err_grad)
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
    # Private methods for fitting
    ##########################################################
 
    def _Err(self,P,x,y):
        theta,p = P[:-1],P[-1]
        diffs = self.shape.f(theta,x) - y
        residuals_term = 0.5 * p**2 * sum(diffs**2)
        n = len(y)
        E = -n*np.log(p) + residuals_term
        if self.use_theta_prior:
            E = E - self.shape.log_prob_theta(theta)
        if self.use_sigma_prior:
            # using a normal distribution for priors (should convert to Gamma)
            z = (p - cfg.inv_sigma_prior_mean) / cfg.inv_sigma_prior_sigma
            log_prob_p = -0.5 * z**2
            E = E - log_prob_p
        return E
        
    def _Err_grad(self,P,x,y):
        theta,p = P[:-1],P[-1]
        n = len(y)
        diffs = self.shape.f(theta,x) - y
        d_theta = np.array([p**2 * sum(diffs*d) for d in self.shape.f_grad(theta,x)])
        if self.use_theta_prior:
            d_theta = d_theta - self.shape.d_theta_prior(theta)
        d_p = -n/p + p*sum(diffs**2)
        if self.use_sigma_prior:
            d_p_prior = - (p - cfg.inv_sigma_prior_mean) / cfg.inv_sigma_prior_sigma**2
            d_p = d_p - d_p_prior
        return np.r_[d_theta, d_p]

def check_grad(n=100):
    import utils
    utils.disable_all_warnings()
    import scipy.optimize
    from shapes.sigmoid import Sigmoid
    for use_theta_prior in [True,False]:
        for use_sigma_prior in [True, False]:
            rng = np.random.RandomState(0)
            print 'Checking theta_prior={}, sigma_prior={}'.format(use_theta_prior, use_sigma_prior)
            fitter = Fitter(Sigmoid(), use_theta_prior=use_theta_prior, use_sigma_prior=use_sigma_prior)
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
