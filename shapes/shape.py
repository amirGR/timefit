import config as cfg
import numpy as np
from priors import get_prior

class Shape(object):
    """Base class for different shape objects, e.g. sigmoid.    
       Derived classes should implement:
           lst = param_names()
           str = cache_name()
           y = f(theta,x)
       A class that works with Fitter should implement:
           d_theta = f_grad(theta,x)
           theta0 = get_theta_guess(x,y)
           theta = adjust_for_scaling(theta,sx,sy)
       A class that does its own special fitting should implement:
           theta = fit(x,y)
    """
    def __init__(self, priors):
        """Prior function for each parameter should be passed by the derived class.
           NOTE: We are modeling distributions as independent, which may not be good enough later on.
                 If this assumption changes, some code will need to move around.
        """
        priors_name = priors
        dct_priors = get_prior(priors)
        if dct_priors is None:
            priors = None
        else:
            priors = []
            for p in self.param_names():
                val = dct_priors.get(p)
                if val is None:
                    raise Exception("Could not find prior for {} in {}".format(p,priors_name))
                priors.append(val)        
        self.priors_name = priors_name
        self.priors = priors            

    def __str__(self):
        return self.cache_name().capitalize()
        
    def n_params(self):
        return len(self.param_names())

    def format_params(self, theta, latex=False):
        names = self.param_names(latex)
        return ', '.join('{}={:.2g}'.format(name,val) for name,val in zip(names,theta))

    def has_special_fitting(self):
        return hasattr(self,'fit')

    def can_export_params_to_matlab(self):
        return True #override where not possible

    def bounds(self):
        if self.priors is None:
            return self.n_params() * [(None,None)]        
        return [pr.bounds() for pr in self.priors]

    def log_prob_theta(self, theta):
        # NOTE: This assumes the priors for different parameters are independent
        return sum(pr.log_prob(t) for pr,t in zip(self.priors,theta))
        
    def d_log_prob_theta(self, theta):
        # NOTE: This assumes the priors for different parameters are independent
        return np.array([pr.d_log_prob(t) for pr,t in zip(self.priors,theta)])

    def high_res_preds(self, theta, x):
        x_smooth = np.linspace(x.min(),x.max(),cfg.n_curve_points_to_plot)
        y_smooth = self.f(theta, x_smooth)
        return x_smooth,y_smooth

    def TEST_check_grad(self, n=100, threshold=1E-7):
        import scipy.optimize
        rng = np.random.RandomState(0)
        def check_one():
            x = rng.uniform(-10,10)
            theta = rng.uniform(size=self.n_params())
            diff = scipy.optimize.check_grad(self.f, self.f_grad, theta, x)
            return diff
        max_diff = max([check_one() for _ in xrange(n)])
        print 'Max difference over {} iterations: {}'.format(n,max_diff)
        if max_diff < threshold:
            print 'Gradient is OK'
        else:
            print 'Difference is too big. Gradient is NOT OK!'

#####################################################
# Building shape from command line input
#####################################################
def allowed_shape_names():
    return ['sigmoid', 'poly0', 'poly1', 'poly2', 'poly3', 'spline']

def get_shape_by_name(shape_name, priors):
    import re
    if shape_name == 'sigmoid':
        from sigmoid import Sigmoid
        return Sigmoid(priors)
    elif shape_name.startswith('poly'):
        m = re.match('poly(\d)',shape_name)
        assert m, 'Illegal polynomial shape name'
        degree = int(m.group(1))
        from poly import Poly
        return Poly(degree,priors)
    elif shape_name == 'spline':
        from spline import Spline
        return Spline()
    else:
        raise Exception('Unknown shape: {}'.format(shape_name))
