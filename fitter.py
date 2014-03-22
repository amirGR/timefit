from functools import partial
import config as cfg
import numpy as np
from minimization import minimize_with_restarts
from sklearn.cross_validation import LeaveOneOut
from shapes.priors import get_prior

class Fitter(object):
    def __init__(self, shape, sigma_prior=None):
        self.shape = shape
        self.inv_sigma_prior_name = sigma_prior
        self.inv_sigma_prior = get_prior(sigma_prior,is_sigma=True)
        if cfg.verbosity > 0:
            if self.shape.priors is not None:
                print 'Fitting using prior {} on theta:'.format(self.shape.priors_name)
                for name,pr in zip(self.shape.param_names(), self.shape.priors):
                    print '\t{}: {}'.format(name,pr)
            if self.inv_sigma_prior is not None:
                print 'Fitting using prior on 1/sigma = {}'.format(self.inv_sigma_prior)
        
    def __str__(self):
        return 'Fitter({}, theta_prior={}, inv_sigma_prior={})'.format(self.shape, self.shape.priors_name, self.inv_sigma_prior_name)
        
    def cache_name(self):
        name = self.shape.cache_name()
        if self.shape.priors is not None:
            name = '{}-theta-{}'.format(name,self.shape.priors_name)
        if self.inv_sigma_prior is not None:
            name = '{}-sigma-{}'.format(name,self.inv_sigma_prior_name)
        return name
        
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
            for i,(train,test) in enumerate(LeaveOneOut(n)):
                if cfg.verbosity >= 2:
                    print 'LOO fit: doing iteration {}/{}'.format(i,n)
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
                if self.shape.priors is not None:
                    P0[:-1] = np.array([pr.generate() for pr in self.shape.priors])
                if self.inv_sigma_prior is not None:
                    P0[-1] = self.inv_sigma_prior.generate()
                return P0
        f = partial(self._Err, x=x, y=y)
        f_grad = partial(self._Err_grad, x=x, y=y)
        if self.shape.priors is not None:
            theta_bounds = self.shape.bounds()
        else:
            theta_bounds = self.shape.n_params() * [(None,None)]
        if self.inv_sigma_prior is not None:
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
        if self.shape.priors is not None:
            E = E - self.shape.log_prob_theta(theta)
        if self.inv_sigma_prior is not None:
            E = E - self.inv_sigma_prior.log_prob(p)
        return E
        
    def _Err_grad(self,P,x,y):
        theta,p = P[:-1],P[-1]
        n = len(y)
        diffs = self.shape.f(theta,x) - y
        d_theta = np.array([p**2 * sum(diffs*d) for d in self.shape.f_grad(theta,x)])
        if self.shape.priors is not None:
            d_theta = d_theta - self.shape.d_log_prob_theta(theta)
        d_p = -n/p + p*sum(diffs**2)
        if self.inv_sigma_prior is not None:
            d_p = d_p - self.inv_sigma_prior.d_log_prob(p)
        return np.r_[d_theta, d_p]

