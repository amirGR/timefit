import numpy as np

random_seed = 0 # None means initialize using time or /dev/urandom

fontsize = 18

default_figure_size_x = 18.5
default_figure_size_y = 10.5
default_figure_facecolor = 0.85 * np.ones(3)
default_figure_dpi = 100

b_verbose_optmization = False
b_allow_less_restarts = True
b_minimal_restarts = False
if b_minimal_restarts:
    n_optimization_restarts = 2
    n_max_optimization_attempt_factor = 10
else:
    n_optimization_restarts = 10
    n_max_optimization_attempt_factor = 20

# theta = [a,h,mu,w]
theta_prior_mean = np.array([5, 5, 30, 2.5])
theta_prior_sigma = np.array([5, 5, 30, 2.5])

# these two settings must change together
from sklearn.metrics import r2_score as score
score_type = 'R2'
