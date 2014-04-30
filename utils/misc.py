import warnings
from math import ceil, sqrt
from contextlib import contextmanager
from functools import wraps
from os import makedirs
import os.path
import numpy as np
import matplotlib.pyplot as plt

def disable_all_warnings():
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    np.seterr(all='ignore') # Ignore numeric overflow/underflow etc. YYY - can/should we handle these warnings?

@contextmanager
def interactive(b):
    b_prev = plt.isinteractive()
    plt.interactive(b)
    try:
        yield
    finally:
        plt.interactive(b_prev)

def ensure_dir(d):
    if not os.path.exists(d):
        makedirs(d)

def init_array(val, *shape):
    a = np.empty(shape)
    a.fill(val)
    return a

def retry(n_max):
    """\
    retry - a decorator that retries a function/method up to N times.
    
    The wrapped function will exit with the return value of the first successful call, or
    with the exception raised in the last attempt, if it failed N times.
    
    >> @retry(3)
    >> def foo(...)
    """
    def deco(f):
        @wraps(f)
        def _wrapped(*a,**kw):
            for i in xrange(n_max):
                try:
                    return f(*a,**kw)
                except:
                    if i == n_max-1:
                        raise
        return _wrapped
    return deco
    
def get_unique(seq):
    s = set(seq)
    if not s:
        raise Exception('get_unique: no items')
    if len(s) > 1:
        raise Exception('get_unique: items are not unique')
    res = s.pop()
    return res

def rect_subplot(nPlots):
    nRows = ceil(sqrt(nPlots))
    nCols = ceil(float(nPlots)/nRows)
    return nRows,nCols