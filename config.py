import numpy as np

fontsize = 18

n_optimization_attempts = 5

# theta = [a,h,mu,w]
theta_prior = np.array([5, 5, 30, 2.5])
theta_prior_sigma = np.array([5, 5, 30, 2.5])

# these two settings must change together
from sklearn.metrics import r2_score as score
score_type = 'R2'
