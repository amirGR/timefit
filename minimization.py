import numpy as np
import config as cfg

def minimize_with_restarts(f_minimize, f_get_P0):
    n = cfg.n_optimization_restarts
    n_max = n * cfg.n_max_optimization_attempt_factor

    results = n*[None]
    n_results = 0
    for i in xrange(n_max):
        P0 = f_get_P0()
        res = f_minimize(P0)
        if not res.success or np.isnan(res.fun):
            continue
        results[n_results] = res
        n_results += 1
        if n_results == n:
            if cfg.b_verbose_optmization:
                print 'Found {} results after {} attempts'.format(n,i+1)
            break
    else:
        msg = 'Optimization failed. Got only {}/{} results in {} attempts'.format(n_results,n,n_max)
        assert cfg.b_allow_less_restarts, msg
        print 'Warning: ', msg
    best_res = min(results[:n_results], key=lambda res: res.fun)
    return best_res.x
