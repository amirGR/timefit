import warnings
from contextlib import contextmanager
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
        
####################################################################
####################################################################
# Matlab conversions        
####################################################################
####################################################################
        
def matlab_cell_array_to_list_of_strings(cell_array):
    def convert_one(x):
        return x[0] if x else None # some data files contain empty names (with some of these have different type)
    return np.array([convert_one(x) for x in cell_array.flat])

def list_of_strings_to_matlab_cell_array(lst_string):
    def convert_one(x):
        return x if x is not None else ''
    lst = [convert_one(x) for x in lst_string]
    return np.array(lst, dtype='object') # tells savemat to produce a cell array
