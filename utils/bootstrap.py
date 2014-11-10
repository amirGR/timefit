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
