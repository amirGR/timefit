import numpy as np

random_seed = 0 # None means initialize using time or /dev/urandom

fontsize = 18

b_verbose_optmization = False
b_minimal_restarts = False
if b_minimal_restarts:
    n_optimization_restarts = 2
    n_max_optimization_attempt_factor = 10
else:
    n_optimization_restarts = 10
    n_max_optimization_attempt_factor = 5

# theta = [a,h,mu,w]
theta_prior_mean = np.array([5, 5, 30, 2.5])
theta_prior_sigma = np.array([5, 5, 30, 2.5])

# these two settings must change together
from sklearn.metrics import r2_score as score
score_type = 'R2'
