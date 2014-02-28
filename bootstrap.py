import numpy as np
import config as cfg

def bootstrap(vals, f, n_iter=1000):
    vals = np.array(vals) # in case it's some other sequence
    rng = np.random.RandomState(cfg.random_seed)
    n_vals = len(vals)
    f_samples = np.empty(n_iter)
    for i in xrange(n_iter):
        inds = rng.random_integers(0,n_vals-1,n_vals)
        sample = vals[inds]
        f_samples[i] = f(sample)
    return f_samples.mean(), f_samples.std()

if __name__ == '__main__':
    mu = 3
    sigma = 1
    n = 10000
    vals = np.random.normal(mu,sigma,size=n)
    def f(vals):
        return np.mean(vals)
    bs_mu, bs_se = bootstrap(vals, f, n_iter=10000)
    sem = sigma / np.sqrt(n)
    d_mu = abs(mu - bs_mu)
    d_sem = abs(sem - bs_se)
    print 'd_mu={}, d_sem={}'.format(d_mu,d_sem)
    assert d_mu < 0.015, 'Error in mean is too big'
    assert d_sem < 2E-4, 'Error in sem is too big'
    print 'OK'
