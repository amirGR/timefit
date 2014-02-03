import numpy as np
from scipy.optimize import minimize
import config as cfg

def sigmoid(theta,x):
    a,h,mu,w = theta
    return a + h/(1+np.exp(-(x-mu)/w))
    
def sigmoid_grad(theta,x):
    a,h,mu,w = theta
    e = np.exp(-(x-mu)/w)
    d_a = 1
    d_h = 1/(1+e)
    d_mu = -h/(1+e)**2 * e/w
    d_w = -h/(1+e)**2 * e * (x-mu)/w**2
    return [d_a, d_h, d_mu, d_w]

def f_error(P,x,y):
    theta,p = P[:4],P[4]
    squares = (sigmoid(theta,x) - y)**2
    n = len(y)
    prior_z = (theta - cfg.theta_prior_mean) / cfg.theta_prior_sigma
    return -n*np.log(p) + 0.5*p**2*sum(squares) + 0.5*sum(prior_z ** 2)

def f_error_gradient(P,x,y):
    theta,p = P[:4],P[4]
    n = len(y)
    diffs = sigmoid(theta,x) - y
    d_a, d_h, d_mu, d_w = sigmoid_grad(theta,x)
    d_prior = (theta - cfg.theta_prior_mean) / cfg.theta_prior_sigma**2
    d_a = sum(diffs * d_a)
    d_h = sum(diffs * d_h)
    d_mu = sum(diffs * d_mu)
    d_w = sum(diffs * d_w)
    d_theta = p**2 * np.array([d_a, d_h, d_mu, d_w]) + d_prior
    d_s = -n/p + p*sum(diffs**2)
    return np.r_[d_theta, d_s]

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

def fit_sigmoid_simple(x,y):
    rng = np.random.RandomState(cfg.random_seed)
    P0_base = np.array([
        y.min(), # a
        y.max()-y.min(), # h
        (x.min() + x.max()) / 2, # mu
        (x.max() - x.min()) / 2, # w
        1, # p
    ])
    def get_P0():
        return P0_base + rng.normal(0,1,size=5)
    def f_minimize(P0):
        return minimize(f_error, P0, args=(x,y), method='BFGS', jac=f_error_gradient)
    return minimize_with_restarts(f_minimize, get_P0)

def fit_sigmoid_loo(x,y):
    from sklearn.cross_validation import LeaveOneOut
    n = len(y)
    test_preds = np.empty(n)
    for i,(train,test) in enumerate(LeaveOneOut(n)):
        P = fit_sigmoid_simple(x[train],y[train])
        assert not np.isnan(P).any()
        assert y[test] == y[i]  # this should hold for LOO
        theta = P[:-1]
        test_preds[i] = sigmoid(theta,x[test])
    return test_preds

def high_res_preds(x,theta):
    x_smooth = np.linspace(x.min(),x.max(),cfg.n_sigmoid_points_to_plot)
    y_smooth = sigmoid(theta, x_smooth)
    return x_smooth,y_smooth

def check_grad(n=100):
    import scipy.optimize
    rng = np.random.RandomState(0)
    def check_one():
        x = np.arange(-10,11)
        y = sigmoid([-1,3,2,2],x) + rng.normal(size=x.shape)
        a,b,c,d = rng.uniform(size=4)
        s = np.e
        P = [a, a+b, c, d, s]
#        g1 = f_error_gradient(p,x,y)
#        g2 = scipy.optimize.approx_fprime(P, f_error, 1E-8, x, y)
#        print 'g1 = {}'.format(g1)
#        print 'g2 = {}'.format(g2)
#        print 'diff = {}'.format(np.abs(g1-g2))
        diff = scipy.optimize.check_grad(f_error, f_error_gradient, P, x, y)
        return diff
    max_diff = max([check_one() for _ in xrange(n)])
    print 'Max difference over {} iterations: {}'.format(n,max_diff)
    if max_diff < 1E-4:
        print 'Gradient is OK'
    else:
        print 'Difference is too big. Gradient is NOT OK!'

if __name__ == '__main__':
    check_grad()
