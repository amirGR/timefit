import setup
from scipy.stats import spearmanr
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits
from scalers import LogScaler

cfg.verbosity = 1
age_scaler = LogScaler()

def get_gene_correlation(data, fits, gene, regions):
    def get_onset_time(r):
        fit = fits[data.name][(gene,r)]
        a,h,mu,w = fit.theta
        return mu
    onset_times = [get_onset_time(r) for r in regions]
    r,pval = spearmanr(onset_times, range(len(regions)))
    return r

pathway = 'serotonin'
data = GeneData.load('kang2011').restrict_pathway(pathway).scale_ages(age_scaler)
shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')
fits = get_all_fits(data, fitter, allow_new_computation=False)
# R2_threshold = 0.5 YYY problem - we might be using bad fits.

regions = ['OFC', 'M1C', 'S1C', 'IPC', 'V1C']

scores = {g: get_gene_correlation(data,fits,g,regions) for g in data.gene_names}

