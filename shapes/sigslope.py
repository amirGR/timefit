import numpy as np
from shape import Shape

class Sigslope(Shape):
    def __init__(self, priors=None):
        Shape.__init__(self, priors)
        
    def param_names(self, latex=False):
        if latex:
            return ['$Baseline$', '$Height$', '$Onset$', '$Slope$']
        else:
            return ['baseline', 'height', 'onset', 'slope']
                
    def format_params(self, theta, x_scaler, latex=False):
        a,h,mu,b = theta
        if x_scaler is None:
            unscaled_mu = mu
            unscaled_w = 1/b
        else:
            unscaled_mu = x_scaler.unscale(mu)
            w_minus = x_scaler.unscale(mu-1/b)
            w_plus = x_scaler.unscale(mu+1/b)
            unscaled_w = 0.5 * (w_plus - w_minus)
        a_name, h_name, mu_name, b_name = self.param_names(latex)
        w_name = '$Width$' if latex else 'width'
        return '{mu_name}={unscaled_mu:.2g} years, {w_name}={unscaled_w:.2g} years'.format(**locals())

    def cache_name(self):
        return 'sigslope'

    def canonical_form(self, theta):
        a,h,mu,b = theta
        if b < 0:
            theta = (a+h,-h,mu,-b) # this is an equivalent sigmoid, with b now positive
        return theta

    def is_positive_transition(self, theta):
        a,h,mu,b = theta
        return h*b > 0

    def f(self,theta,x):
        a,h,mu,b = theta
        return a + h/(1+np.exp(-(x-mu)*b))

    def f_grad(self,theta,x):
        a,h,mu,b = theta
        e = np.exp(-(x-mu)*b)
        ie = np.exp((x-mu)*b)
        d_a = 1
        d_h = 1/(1+e)
        d_mu = -h*b/((1+e)*(1+ie))
        d_b = h*(x-mu)/((1+e)*(1+ie))
        return [d_a, d_h, d_mu, d_b]
    
    def get_theta_guess(self,x,y):
        return [
            y.min(), # a
            y.max()-y.min(), # h
            (x.min() + x.max()) / 2.0, # mu
            2.0 / (x.max() - x.min()), # b = 1/w
        ]
    
    def adjust_for_scaling(self, theta, sx, sy):
        a,h,mu,b = theta
        a = a/sy[0] + sy[1]
        h = h / sy[0]
        mu = mu/sx[0] + sx[1]
        b = b * sx[0]
        return a,h,mu,b

if __name__ == '__main__':
    Sigslope().TEST_check_grad()
