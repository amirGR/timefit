import numpy as np
from scipy import stats
import config as cfg

class NormalPrior(object):
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
        
    def __repr__(self):
        return 'NormalPrior(mu={:.2f}, sigma={:.2f})'.format(self.mu, self.sigma)
        
    @property
    def rv(self):
        return stats.norm(self.mu, self.sigma)

    def generate(self):
        return self.rv.rvs()
        
    def bounds(self):
        return (None,None)

    def log_prob(self, x):
        z = (x - self.mu) / self.sigma
        return -0.5*(z**2)

    def d_log_prob(self, x):
        return -(x - self.mu) / (self.sigma**2)

class GammaPrior(object):
    def __init__(self,alpha,beta,mu):
        self.a = alpha
        self.b = beta
        self.mu = mu
        
    def __repr__(self):
        return 'Gamma(a={:.2f}, b={:.2f}, mu={:.2f})'.format(self.a,self.b,self.mu)

    @property
    def rv(self):
        return stats.gamma(self.a, loc=self.mu, scale=1/self.b)
        
    def generate(self):
        return self.rv.rvs()
        
    def bounds(self):
        return (self.mu,None)

    def log_prob(self, x):
        x = x - self.mu
        return (self.a-1)*np.log(x) - self.b*x
            
    def d_log_prob(self, x):
        x = x - self.mu
        return (self.a-1)/x - self.b

########################################################
# Load priors
########################################################
from project_dirs import priors_dir
from os.path import join, isfile, basename, splitext
from glob import glob
def get_allowed_priors(is_sigma=False):
    prior_type = 'theta' if not is_sigma else 'sigma'
    pattern = join(priors_dir(), prior_type, '*.txt')
    prior_files = glob(pattern)
    names = [splitext(basename(filename))[0] for filename in prior_files]
    return names
    
def get_prior(name, is_sigma=False):
    if name is None:
        return None
    prior_type = 'theta' if not is_sigma else 'sigma'
    path = join(priors_dir(), prior_type, name+'.txt')
    if isfile(path):
        if cfg.verbosity > 0:
            print 'Reading priors for {} from {}'.format(prior_type, path)
        with open(path) as f:
            priors = eval(f.read())
        return priors
    raise Exception('Could not find prior file at {}'.format(path))
