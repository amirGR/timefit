from functools import partial
import config as cfg
import numpy as np
from minimization import minimize_with_restarts
from sklearn.cross_validation import LeaveOneOut
from shapes.priors import NormalPrior, GammaPrior

class Fitter(object):
    def __init__(self, shape, use_theta_prior=False, use_sigma_prior=False):
        self.shape = shape
        self.use_theta_prior = use_theta_prior
        self.use_sigma_prior = use_sigma_prior
        self.inv_sigma_prior = GammaPrior(2.61,1.15,0.65)
        
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

    def fit(self, x, y, loo=False):
        P0 = self._fit(x,y)
        t0,s0 = self._unpack_P(P0)
        if loo:            
            n = len(y)
            test_preds = np.empty(n)
            for train,test in LeaveOneOut(n):
                P = self._fit(x[train],y[train],single_init_P0=P0)
                if P is None:
                    test_preds[test] = np.nan
                else:
                    theta,sigma = self._unpack_P(P)
                    test_preds[test] = self.predict(theta,x[test])
        else:
            test_preds = None
        return t0, s0, test_preds
        
    def predict(self, theta, x):
        return self.shape.f(theta,x)

    ##########################################################
    # Private methods for fitting
    ##########################################################

    def _fit(self,x,y,single_init_P0=None):
        n_restarts = cfg.n_optimization_restarts
        if single_init_P0 is not None:
            n_restarts = n_restarts + 1
        rng = np.random.RandomState(cfg.random_seed)
        np.random.seed(cfg.random_seed) # for prior.generate() which doesn't have a way to control rng
        P0_base = np.array(self.shape.get_theta_guess(x,y) + [1])
        def get_P0(i):
            if single_init_P0 is not None and i==0:
                return single_init_P0
            else:
                # if we're using priors, draw from the prior distribution
                P0 = P0_base + rng.normal(0,1,size=P0_base.shape)
                if self.use_theta_prior:
                    P0[:-1] = np.array([pr.generate() for pr in self.shape.priors])
                if self.use_sigma_prior:
                    P0[-1] = self.inv_sigma_prior.generate()
                return P0
        f = partial(self._Err, x=x, y=y)
        f_grad = partial(self._Err_grad, x=x, y=y)
        if self.use_theta_prior:
            theta_bounds = self.shape.bounds()
        else:
            theta_bounds = self.shape.n_params() * [(None,None)]
        if self.use_sigma_prior:
            p_bounds = self.inv_sigma_prior.bounds()
        else:
            p_bounds = (None,None)
        bounds = theta_bounds + [p_bounds]
        P = minimize_with_restarts(f, f_grad, get_P0, bounds, n_restarts)
        return P
        
    def _unpack_P(self, P):
        if P is None:
            return None,None
        assert not np.isnan(P).any()
        theta = P[:-1]
        sigma = 1/P[-1]
        return theta,sigma
 
    def _Err(self,P,x,y):
        theta,p = P[:-1],P[-1]
        diffs = self.shape.f(theta,x) - y
        residuals_term = 0.5 * p**2 * sum(diffs**2)
        n = len(y)
        E = -n*np.log(p) + residuals_term
        if self.use_theta_prior:
            E = E - self.shape.log_prob_theta(theta)
        if self.use_sigma_prior:
            E = E - self.inv_sigma_prior.log_prob(p)
        return E
        
    def _Err_grad(self,P,x,y):
        theta,p = P[:-1],P[-1]
        n = len(y)
        diffs = self.shape.f(theta,x) - y
        d_theta = np.array([p**2 * sum(diffs*d) for d in self.shape.f_grad(theta,x)])
        if self.use_theta_prior:
            d_theta = d_theta - self.shape.d_log_prob_theta(theta)
        d_p = -n/p + p*sum(diffs**2)
        if self.use_sigma_prior:
            d_p = d_p - self.inv_sigma_prior.d_log_prob(p)
        return np.r_[d_theta, d_p]


from scipy.optimize import approx_fprime
_epsilon = np.sqrt(np.finfo(float).eps)
def my_check_grad(func, grad, x0, *args):
    "Copied from scipy.optimize - allows tweaking to see which parameter causes the error"
    g = grad(x0, *args)
    a = approx_fprime(x0, func, _epsilon, *args)
    d = g-a
    return np.sqrt(sum(d**2))

def check_grad(n=100):
    import utils
    utils.disable_all_warnings()
    from shapes.sigmoid import Sigmoid

    def check_one(fitter, use_theta_prior, use_sigma_prior, rng):
        theta = [pr.rv.mean()*rng.normal(1,0.05) for pr in fitter.shape.priors]
        x = np.arange(-10,11)
        y = fitter.predict(theta,x) + rng.normal(size=x.shape)
        theta_guess = fitter.shape.get_theta_guess(x,y)
        p_guess = 1
        P = np.r_[theta_guess, p_guess]
        if use_theta_prior:
            P[:-1] = np.array([pr.generate() for pr in fitter.shape.priors])
        if use_sigma_prior:
            P[-1] = fitter.inv_sigma_prior.generate()
        diff = my_check_grad(fitter._Err, fitter._Err_grad, P, x, y)
        return diff

    def check_variation(shape, use_theta_prior, use_sigma_prior, n):
        np.random.seed(cfg.random_seed) # for prior.generate() which doesn't have a way to control rng     
        rng = np.random.RandomState(cfg.random_seed)
        print '\n',
        80*'='
        print 'Checking theta_prior={}, sigma_prior={}'.format(use_theta_prior, use_sigma_prior)
        fitter = Fitter(shape, use_theta_prior=use_theta_prior, use_sigma_prior=use_sigma_prior)
        max_diff = max([check_one(fitter, use_theta_prior, use_sigma_prior, rng) for _ in xrange(n)])
        print 'Max difference over {} iterations: {}'.format(n,max_diff)
        if max_diff < 1E-3:
            print 'Gradient is OK'
        else:
            print '*** FAILED: Difference is too big. Gradient is NOT OK!'
        return max_diff

#    cfg.random_seed = 46 #608
#    check_variation(Sigmoid(),True,False, 1)
#    return
    
    shape = Sigmoid()
    for tp in [False,True]:
        for sp in [False,True]:
            check_variation(shape,tp, sp, n)

if __name__ == '__main__':
    check_grad()
