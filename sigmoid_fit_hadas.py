import numpy as np
from scipy.optimize import minimize
import config as cfg
from minimization import minimize_with_restarts
from fit_score import loo_score

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

def f_error(theta,x,y,L):
    a,h,mu,w = theta
    squares = (sigmoid(theta,x) - y)**2
    prior_z = (theta - cfg.theta_prior_mean) / cfg.theta_prior_sigma
    regularizer = sum(prior_z ** 2);
    return 0.5*sum(squares) + 0.5*L*regularizer

def f_error_gradient(theta,x,y,L):
    a,h,mu,w = theta
    diffs = sigmoid(theta,x) - y
    d_a, d_h, d_mu, d_w = sigmoid_grad(theta,x)
    d_prior = L * (theta - cfg.theta_prior_mean) / cfg.theta_prior_sigma**2
    d_a = sum(diffs * d_a)
    d_h = sum(diffs * d_h)
    d_mu = sum(diffs * d_mu)
    d_w = sum(diffs * d_w)
    return np.array([d_a, d_h, d_mu, d_w]) + d_prior

def fit_sigmoid_simple(x,y,L):
    rng = np.random.RandomState(cfg.random_seed)
    theta0_base = np.array([
        y.min(), # a
        y.max()-y.min(), # h
        (x.min() + x.max()) / 2, # mu
        (x.max() - x.min()) / 2, # w
    ])
    def get_theta0():
        return theta0_base + rng.normal(0,1,size=4)
    def f_minimize(theta0):
        return minimize(f_error, theta0, args=(x,y,L), method='BFGS', jac=f_error_gradient)
    return minimize_with_restarts(f_minimize, get_theta0)

def fit_sigmoid_loo_with_L(x,y,L):
    from sklearn.cross_validation import KFold
    rng = np.random.RandomState(cfg.random_seed)
    n = len(y)
    k = cfg.n_folds_for_hadas_fit
    test_preds = np.empty(n)
    for train,test in KFold(n,k,shuffle=True,random_state=rng):
        theta = fit_sigmoid_simple(x[train],y[train],L)
        if theta is None:
            test_preds[test] = np.nan
        else:
            assert not np.isnan(theta).any()
            test_preds[test] = sigmoid(theta,x[test])
    return test_preds

def find_best_L(x,y,Ls=None):
    if Ls is None:
        Ls = np.logspace(-2,2,10)
    def score(L):
        preds = fit_sigmoid_loo_with_L(x,y,L)
        return loo_score(y, preds)
    scores = np.array([score(L) for L in Ls])
    return Ls[np.argmax(scores)]

def fit_sigmoid_loo(x,y,Ls=None):
    from sklearn.cross_validation import KFold
    rng = np.random.RandomState(cfg.random_seed)
    n = len(y)
    k = cfg.n_folds_for_hadas_fit
    test_preds = np.empty(n)
    for i,(train,test) in enumerate(KFold(n,k,shuffle=True,random_state=rng)):
        print 'Computing prediction for points {} (batch {}/{})'.format(list(test),i+1,k)        
        L = find_best_L(x[train],y[train],Ls)
        theta = fit_sigmoid_simple(x[train],y[train],L)
        if theta is None:
            test_preds[test] = np.nan
        else:
            assert not np.isnan(theta).any()
            test_preds[test] = sigmoid(theta,x[test])
    return test_preds

def check_grad(n=10):
    import scipy.optimize
    def check_one():
        x = np.arange(-10,11)
        y = sigmoid([-1,3,2,2],x) + np.random.normal(size=x.shape)
        L = np.random.uniform()
        a,b,c,d = np.random.uniform(size=4)
        theta = [a, a+b, c, d]
        diff = scipy.optimize.check_grad(f_error, f_error_gradient, theta, x, y, L)
        return diff
    max_diff = max([check_one() for _ in xrange(n)])
    print 'Max difference over {} iterations: {}'.format(n,max_diff)
    if max_diff < 1E-5:
        print 'Gradient is OK'
    else:
        print 'Difference is too big. Gradient is NOT OK!'

if __name__ == '__main__':
    check_grad()
