import pickle

def batches(seq, n):
    """yield batches of size n from seq. e.g.:
       batches(range(10),4) = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """
    from itertools import groupby
    from math import floor
    return [[v for i,v in g] for k,g in groupby(enumerate(seq),lambda ix: floor(ix[0]/n))]

def get_vars_in_module(module, only_picklable=True, remove_private=True):
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

def set_vars_in_module(module,dct_vars):
    for k,v in dct_vars.iteritems():
        setattr(module,k,v)
