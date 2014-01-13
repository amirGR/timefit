import numpy as np

fontsize = 18

n_optimization_attempts = 5

theta_prior = np.array([5, 5, 30, 2.5])
theta_prior_sigma = np.array([5, 5, 30, 2.5])

score_type = 'R2'
from sklearn.metrics import r2_score, explained_variance_score
dct_scores = {'R2' : r2_score, 'Explained Variance' : explained_variance_score}
score = dct_scores[score_type]
