"""
Handles sharding, parallalization, saving/loading files from cache, saving checkpoints 
during computation, etc.
The idea is to separate all the boiler plate code that handles the above issues from the code 
that perform the computation you're actually interested in.

This handles embarrasingly parallel computations, where the computation is a big dictionary
from a set of keys to the result of the computation on them.
"""

import pickle
import os
from os.path import dirname, join, isfile
from glob import glob
import config as cfg
from project_dirs import cache_dir
from utils.misc import ensure_dir
from utils import parallel

def proxy(*a,**kw):
    return a,kw

def compute(name, f, arg_mapper, all_keys, k_of_n, base_filename, batch_size=None):
    """ name - appears in print messages if verbosity > 0
        f - pickleable function that is called to do the actual computation on each sub-process
        arg_mapper(key,f_proxy):
            Translates key to arguments for the call to f.
            For convenience, this is done by calling f_proxy with the correct arguments and returning the result.
        k_of_n - None for complete computation. Otherwise (k,n) pair - n = number of parts, k = part number in [1..n]
        base_filename - base file name to use for caching the results
        batch_size - how many iterations to do before saving a checkpoint
    """
    if arg_mapper is None:
        def arg_mapper(key,f_proxy):
            return f_proxy(key)
    if batch_size is None:
        batch_size = cfg.job_batch_size
        
    filename = join(cache_dir(), base_filename + '.pkl')
    ensure_dir(dirname(filename))

    keys = _get_shard(all_keys, k_of_n)
    res = _read_all_cache_files(filename, keys, b_consolidate = (k_of_n is None)) #YYY

    missing_keys = set(k for k in keys if k not in res)
    if cfg.verbosity > 0:
        print 'Still need to compute {}/{} {}'.format(len(missing_keys),len(keys),name)

    # compute the keys that are missing
    batches = parallel.batches(missing_keys, batch_size)
    pool = parallel.Parallel(_job_wrapper)
    for i,batch in enumerate(batches):
        if cfg.verbosity > 0:
            print 'Computing {}: batch {}/{} ({} jobs per batch)'.format(name,i+1,len(batches),batch_size)
        updates = pool(pool.delay(f,key,*arg_mapper(key,proxy)) for key in batch)
        updates = dict(updates) # convert key,value pairs to dictionary
        res.update(updates)
        _save_results(res, filename, k_of_n)
    return res

def _job_wrapper(f,key,a,kw):
    # this must be a top-level function so the parallelization can pickle it
    val = f(*a,**kw)
    return key,val

def _get_shard(all_keys, k_of_n):
    if k_of_n is None:
        return all_keys
    k,n = k_of_n
    return all_keys[k-1::n] # k is one-based, so subtract one

def _save_results(res, filename, k_of_n):
    if k_of_n is not None:
        k,n = k_of_n
        filename = '{}.{}-of-{}'.format(filename,k,n)
    with open(filename,'w') as f:
        pickle.dump(res,f)

def _read_all_cache_files(basefile, keys, b_consolidate):
    # collect results from basefile and all shard files
    res = _read_one_cache_file(basefile)
    partial_files = set(glob(basefile + '*')) - {basefile}
    for filename in partial_files:
        res.update(_read_one_cache_file(filename))

    # reduce to the set we need (especially if we're working on a shard)
    res = {k:v for k,v in res.iteritems() if k in set(keys)}
        
    if b_consolidate:
        _save_results(res,basefile,None)
        for filename in partial_files:
            os.remove(filename)

    return res

def _read_one_cache_file(filename):
    if not isfile(filename):
        if cfg.verbosity > 0:
            print 'No cache file {}'.format(filename)
        return {}
    try:
        if cfg.verbosity > 0:
            print 'Reading cached results from {}'.format(filename)
        with open(filename) as f:
            res = pickle.load(f)
            if cfg.verbosity > 0:
                print 'Found {} cached results in {}'.format(len(res),filename)
    except:
        print 'Failed to read cached results from {}'.format(filename)
        res = {}
    return res
