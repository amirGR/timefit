import setup
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import load_data
from all_fits import get_all_fits
from scipy.stats import spearmanr, wilcoxon

import utils
utils.disable_all_warnings()

data = load_data('serotonin', 'kang2011')
fits = get_all_fits(data, b_hadas_fits=False)
fits_hadas = get_all_fits(data, b_hadas_fits=True)

matched_scores = [(fits_hadas[k].LOO_score,fits[k].LOO_score) for k in fits.iterkeys()]
old_scores = np.array([old for old,new in matched_scores])
new_scores = np.array([new for old,new in matched_scores])
score_diff = new_scores - old_scores

r, p_value = spearmanr(old_scores, new_scores)
T, signed_rank_p_value = wilcoxon(old_scores, new_scores)

# as histogram
fig = plt.figure()
plt.hist(score_diff, range=(-1,1), bins=20)
plt.title('Comparison of fit methods on held out data\nWilcoxon signed rank p-value={:.4f}'.format(signed_rank_p_value), fontsize=cfg.fontsize)
plt.xlabel(r'R2(global MLE) - R2($\lambda$ by CV)', fontsize=cfg.fontsize)
plt.ylabel(r'Number of gene/regions', fontsize=cfg.fontsize)

# as scatter plot
fig = plt.figure()
plt.scatter(old_scores, new_scores, alpha=0.5)
plt.plot(np.linspace(-1,1,100),np.linspace(-1,1,100),'b--')
plt.axis([-1,1,-1,1])
plt.title('Comparison of fit methods on held out data\n(Spearman r={:.2f})'.format(r), fontsize=cfg.fontsize)
plt.xlabel(r'R2($\lambda$ by CV)', fontsize=cfg.fontsize)
plt.ylabel(r'R2(global MLE)', fontsize=cfg.fontsize)
