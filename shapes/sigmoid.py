import numpy as np
from shape import Shape

class Sigmoid(Shape):
    def __init__(self, priors=None):
        Shape.__init__(self, priors)
        
    def param_names(self, latex=False):
        if latex:
            return ['$Baseline$', '$Height$', '$Onset$', '$Width$']
        else:
            return ['baseline', 'height', 'onset', 'width']
                
    def format_params(self, theta, x_scaler, latex=False):
        a,h,mu,w = theta
        if x_scaler is None:
            unscaled_mu = mu
            unscaled_w = w
        else:
            unscaled_mu = x_scaler.unscale(mu)
            w_minus = x_scaler.unscale(mu-w)
            w_plus = x_scaler.unscale(mu+w)
            unscaled_w = 0.5 * (w_plus - w_minus)
        theta = (a, h, unscaled_mu, unscaled_w)
        names = self.param_names(latex)
        return ', '.join('{}={:.2g}'.format(name,val) for name,val in zip(names,theta))

    def cache_name(self):
        return 'sigmoid'

    def f(self,theta,x):
        a,h,mu,w = theta
        return a + h/(1+np.exp(-(x-mu)/w))

    def f_grad(self,theta,x):
        a,h,mu,w = theta
        e = np.exp(-(x-mu)/w)
        ie = np.exp((x-mu)/w)
        d_a = 1
        d_h = 1/(1+e)
        d_mu = -h/(w*(1+e)*(1+ie))
        d_w = -h*(x-mu)/(w**2 * (1+e) * (1+ie))
        return [d_a, d_h, d_mu, d_w]
    
    def get_theta_guess(self,x,y):
        return [
            y.min(), # a
            y.max()-y.min(), # h
            (x.min() + x.max()) / 2, # mu
            (x.max() - x.min()) / 2, # w
        ]
    
    def adjust_for_scaling(self, theta, sx, sy):
        a,h,mu,w = theta
        a = a/sy[0] + sy[1]
        h = h / sy[0]
        mu = mu/sx[0] + sx[1]
        w = w / sx[0]
        return a,h,mu,w

if __name__ == '__main__':
    Sigmoid().TEST_check_grad()
