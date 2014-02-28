import config as cfg
import numpy as np

class Shape(object):
    """Base class for different shape objects, e.g. sigmoid.    
       Derived classes should implement:
           str = cache_name()
           str = format_params(theta, latex=False)
           y = f(theta,x)
           
           d_theta = f_grad(theta,x)
           theta0 = get_theta_guess(x,y)
           log_P_theta = log_prob_theta(theta)
           d_theta0 = d_theta_prior(theta)
    """
    def __str__(self):
        return self.cache_name()

    def high_res_preds(self, theta, x):
        x_smooth = np.linspace(x.min(),x.max(),cfg.n_curve_points_to_plot)
        y_smooth = self.f(theta, x_smooth)
        return x_smooth,y_smooth

    def TEST_check_grad(self, theta_size, n=100, threshold=1E-7):
        import scipy.optimize
        rng = np.random.RandomState(0)
        def check_one():
            x = rng.uniform(-10,10)
            theta = rng.uniform(size=theta_size)
            diff = scipy.optimize.check_grad(self.f, self.f_grad, theta, x)
            return diff
        max_diff = max([check_one() for _ in xrange(n)])
        print 'Max difference over {} iterations: {}'.format(n,max_diff)
        if max_diff < threshold:
            print 'Gradient is OK'
        else:
            print 'Difference is too big. Gradient is NOT OK!'
