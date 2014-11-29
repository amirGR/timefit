import numpy as np
import scipy.optimize
import config as cfg

class RecordingCallback(object):
    def __init__(self, f):
        self.f = f
        self.best_val = np.Inf
        self.best_P = None
    def __call__(self, P):
        val = self.f(P)
        if val < self.best_val:
            self.best_val = val
            self.best_P = P

def _minimize(f, f_grad, P0, bounds, cb):
    if bounds is None or bounds == len(bounds)*[(None,None)]:
        scipy.optimize.minimize(f, P0, method='BFGS', jac=f_grad, callback=cb, tol=cfg.minimization_tol)
    else:
        scipy.optimize.minimize(f, P0, method='TNC', jac=f_grad, bounds=bounds, callback=cb, tol=cfg.minimization_tol)
    
def minimize_with_restarts(f, f_grad, f_get_P0, bounds=None, n_restarts=None):
    if n_restarts is None:
        n_restarts = cfg.n_optimization_restarts
    cb = RecordingCallback(f)
    for i in xrange(n_restarts):
        P0 = f_get_P0(i)
        _minimize(f, f_grad, P0, bounds, cb)
    return cb.best_P

def minimize(f, f_grad, P0, bounds=None):
    cb = RecordingCallback(f)
    _minimize(f, f_grad, P0, bounds, cb)
    return cb.best_P
