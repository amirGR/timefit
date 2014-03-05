import numpy as np
import config as cfg
from shape import Shape

class Sigmoid(Shape):
    def cache_name(self):
        return 'sigmoid'
        
    def format_params(self, theta, latex=False):
        a,h,mu,w = theta
        if latex:
            return r'a={a:.2f}, h={h:.2f}, $\mu$={mu:.2f}, w={w:.2f}'.format(**locals())
        else:
            return r'a={a:.2f}, h={h:.2f}, mu={mu:.2f}, w={w:.2f}'.format(**locals())

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
    
    def log_prob_theta(self, theta):
        z = (theta - cfg.sigmoid_theta_prior_mean) / cfg.sigmoid_theta_prior_sigma
        return -0.5*sum(z ** 2 )
        
    def d_theta_prior(self, theta):
        return -(theta - cfg.sigmoid_theta_prior_mean) / cfg.sigmoid_theta_prior_sigma**2

if __name__ == '__main__':
    Sigmoid().TEST_check_grad(theta_size=4)
