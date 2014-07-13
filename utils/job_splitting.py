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
import shutil
from os.path import dirname, join, isfile, isdir
from glob import glob
import config as cfg
from project_dirs import cache_dir
from utils.misc import ensure_dir
from utils import parallel

def proxy(*a,**kw):
    return a,kw

def compute(name, f, arg_mapper, all_keys, k_of_n, base_filename, batch_size=None, f_sharding_key=None, all_sharding_keys=None, allow_new_computation=True):
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
        batch_size = cfg.job_batch_size if len(all_keys) < cfg.job_big_key_size else cfg.job_big_batch_size
     
    keys = _get_shard(all_keys, k_of_n, f_sharding_key, all_sharding_keys)
    dct_res, found_keys_not_in_main_file = _read_all_cache_files(base_filename, k_of_n, keys)
    _consolidate(dct_res, base_filename, k_of_n, found_keys_not_in_main_file)

    missing_keys = set(k for k in keys if k not in dct_res)
    if cfg.verbosity > 0:
        print 'Still need to compute {}/{} {}'.format(len(missing_keys),len(keys),name)
    if missing_keys and not allow_new_computation:
        raise Exception('Cache does not contain all results')

    # compute the keys that are missing
    batches = parallel.batches(missing_keys, batch_size)
    pool = parallel.Parallel(_job_wrapper)
    for i,batch in enumerate(batches):
        if cfg.verbosity > 0:
            print 'Computing {}: batch {}/{} ({} jobs per batch)'.format(name,i+1,len(batches),batch_size)
        if cfg.parallel_run_locally:
            updates = [_job_wrapper(f,key,*arg_mapper(key,proxy)) for key in batch]
        else:
            updates = pool(pool.delay(f,key,*arg_mapper(key,proxy)) for key in batch)
        dct_updates = dict(updates) # convert key,value pairs to dictionary
        _save_batch(dct_updates, base_filename, k_of_n, i)
        dct_res.update(dct_updates)

    _consolidate(dct_res, base_filename, k_of_n, bool(missing_keys))
    return dct_res

def _job_wrapper(f,key,a,kw):
    # this must be a top-level function so the parallelization can pickle it
    val = f(*a,**kw)
    return key,val

def _get_shard(all_keys, k_of_n, f_sharding_key, all_sharding_keys):
    if k_of_n is None:
        return all_keys
    k,n = k_of_n
    if f_sharding_key is None:
        return all_keys[k-1::n] # k is one-based, so subtract one
    else:
        if all_sharding_keys is None:
            all_sharding_keys = sorted(set(f_sharding_key(key) for key in all_keys))
        chosen_skeys = set(all_sharding_keys[k-1::n])
        return [key for key in all_keys if f_sharding_key(key) in chosen_skeys]

def _save_batch(dct_updates, base_filename, k_of_n, i):
    filename = _batch_base_filename(base_filename, k_of_n) + str(i)
    ensure_dir(dirname(filename))    

    # if there's already a file by that name, merge its contents
    if isfile(filename):
        dct_existing = _read_one_cache_file(filename, st_keys=None, is_batch=True)
        dct_updates.update(dct_existing)

    with open(filename,'w') as f:
        pickle.dump(dct_updates,f)

def _read_all_cache_files(base_filename, k_of_n, keys):
    st_keys = set(keys)

    # collect results from our file
    main_filename = _cache_filename(base_filename, k_of_n)
    dct_res = _read_one_cache_file(main_filename, st_keys)
    st_keys_in_main_file = set(dct_res.iterkeys())
    
    # collect results from main file
    global_filename = _cache_filename(base_filename, k_of_n=None)
    if k_of_n is not None: # otherwise we just read this file
        dct_global = _read_one_cache_file(global_filename, st_keys)
        dct_res.update(dct_global)
    
    # collect from all shard files
    shard_files = set(glob(global_filename + '*')) - {global_filename, main_filename}
    if k_of_n is not None:
        k,n = k_of_n
        # we know they keys for k/n are mutually exclusive with any other shard with same n
        # the reason to go over other shards at all is if we change n in the middle so there's an overlap
        shard_files = {f for f in shard_files if str(n) not in f}
    for filename in shard_files:
        dct_shard = _read_one_cache_file(filename, st_keys)
        dct_res.update(dct_shard)

    # collect from all batch files
    batchdir = _batch_dir(base_filename)
    batch_files = glob(join(batchdir,'*'))
    for filename in batch_files:
        dct_batch = _read_one_cache_file(filename, st_keys, is_batch=True)
        dct_res.update(dct_batch)

    st_all_keys_found = set(dct_res.iterkeys())
    found_keys_not_in_main_file = st_all_keys_found > st_keys_in_main_file
    return dct_res, found_keys_not_in_main_file

def _read_one_cache_file(filename, st_keys, is_batch=False):
    verbosity_threshold = 2 if is_batch else 1
    if not isfile(filename):
        if cfg.verbosity >= verbosity_threshold:
            print 'No cache file {}'.format(filename)
        return {}
    try:
        if cfg.verbosity >= verbosity_threshold:
            print 'Reading cached results from {}'.format(filename)
        with open(filename) as f:
            dct_res = pickle.load(f)
            if cfg.verbosity >= verbosity_threshold:
                print 'Found {} cached results in {}'.format(len(dct_res),filename)
    except:
        print 'Failed to read cached results from {}'.format(filename)
        dct_res = {}
        
    if st_keys is not None:
        dct_res = {k:v for k,v in dct_res.iteritems() if k in st_keys}
    return dct_res

def _consolidate(dct_res, base_filename, k_of_n, found_keys_not_in_main_file):
    filename = _cache_filename(base_filename, k_of_n)

    # write the updated main file    
    if found_keys_not_in_main_file:
        ensure_dir(dirname(filename))
        with open(filename,'w') as f:
            pickle.dump(dct_res,f)
    
    if k_of_n is None:
        # it's the main file - delete all k_of_n files and the batches dir
        batchdir = _batch_dir(base_filename)
        if isdir(batchdir):
            shutil.rmtree(batchdir)
        partial_files = set(glob(filename + '*')) - {filename}
        for filename in partial_files:
            os.remove(filename)
    else:
        # it's a shard - delete just the batches for that file
        base = _batch_base_filename(base_filename,k_of_n)
        batch_filenames = glob(base + '*')
        for filename in batch_filenames:
            os.remove(filename)

def _batch_dir(base_filename):
    return join(cache_dir(),base_filename + '-batches')

def _batch_base_filename(base_filename, k_of_n):
    if k_of_n is None:
        prefix = 'main'
    else:
        k,n = k_of_n
        prefix = '{}-of-{}'.format(k,n)
    return join(_batch_dir(base_filename), prefix + '-batch-')

def _cache_filename(base_filename, k_of_n):
    filename = join(cache_dir(), base_filename + '.pkl')
    if k_of_n is not None:
        k,n = k_of_n
        filename = '{}.{}-of-{}'.format(filename,k,n)
    return filename