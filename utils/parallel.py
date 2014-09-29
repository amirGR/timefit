import cPickle as pickle
from sklearn.externals import joblib
from . import misc
import config as cfg

def batches(seq, n):
    """yield batches of size n from seq. e.g.:
       batches(range(10),4) = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """
    from itertools import groupby
    from math import floor
    return [[v for i,v in g] for k,g in groupby(enumerate(seq),lambda ix: floor(ix[0]/n))]

class Parallel(object):
    """A simple wrapper over joblib's parallelization helpers.
       Add common startup code in the child processes that:
         * disables warnings
         * passes the config settings from the parent process in case they changed 
           from their defaults.
    """
    def __init__(self, f, n_jobs=None, verbosity=None):
        if n_jobs is None:
            n_jobs = cfg.parallel_n_jobs
        if verbosity is None:
            verbosity = cfg.verbosity
        job_verbosity = 70 if verbosity >= 1 else 0
        self.pool = joblib.Parallel(n_jobs=n_jobs, verbose=job_verbosity)
        self.f = f
        self.cfg_vars = _get_vars_in_module(cfg)
        
    def delay(self, *a, **kw):        
        df = joblib.delayed(_job_wrapper)
        a = [self.cfg_vars, self.f] + list(a)
        return df(*a,**kw)
        
    def __call__(self, delayed_calls):
        return self.pool(delayed_calls)

def _job_wrapper(_cfg_vars, _func, *a, **kw):
    """Used by Parallel as the entry point for each sub-process.
       This must be a top level function so it can be pickled.
    """
    misc.disable_all_warnings()
    _set_vars_in_module(cfg, _cfg_vars)
    return _func(*a,**kw)

def _get_vars_in_module(module, only_picklable=True, remove_private=True):
    dct = {k:getattr(module,k) for k in dir(module)}
    if remove_private:
        dct = {k:v for k,v in dct.iteritems() if not k.startswith('_')}
    if only_picklable:
        def can_pickle(x):
            try:
                pickle.dumps(x)
                return True
            except:
                return False
        dct = {k:v for k,v in dct.iteritems() if can_pickle(v)}
    return dct

def _set_vars_in_module(module,dct_vars):
    for k,v in dct_vars.iteritems():
        setattr(module,k,v)

