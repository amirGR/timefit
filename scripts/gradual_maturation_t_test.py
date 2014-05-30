import setup
from os.path import join
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import fdrcorrection0 as fdr # not installed on cortex
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits
from scalers import LogScaler
from plots import create_html
from project_dirs import results_dir, fit_results_relative_path
from utils.misc import ensure_dir
from dev_stages import PCW

fontsize = 30

def create_top_genes_html(data, fitter, fits, scores, regions, n_top=None, filename_suffix=''):
    if n_top is None:
        n_top = len(scores)
        
    basedir = join(results_dir(), fit_results_relative_path(data,fitter))
    ensure_dir(basedir)
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'

    def key_func(score):
        g,pval,qval = score
        return pval
    scores.sort(key=key_func)
    top_genes = [g for g,pval,qval in scores[:n_top]]
    top_pvals = {g:pval for g,pval,qval in scores[:n_top]}
    top_qvals = {g:qval for g,pval,qval in scores[:n_top]}
    
    n = len(scores)
    n05 = len([g for g,pval,qval in scores if qval < 0.05])
    n01 = len([g for g,pval,qval in scores if qval < 0.01])
    top_text = """\
one sided t-test: {regions[0]} < {regions[1]}
{n05}/{n} q-values < 0.05
{n01}/{n} q_values < 0.01
""".format(**locals())
    
    def get_onset_time(fit):
        a,h,mu,w = fit.theta
        age = age_scaler.unscale(mu)
        return 'onset = {:.3g} years'.format(age)
        
    def get_onset_dist(fit):
        mu_vals = fit.theta_samples[2,:]
        mu = mu_vals.mean()
        vLow,vHigh = np.percentile(mu_vals, (20,80))
        mu = age_scaler.unscale(mu)
        vLow = age_scaler.unscale(vLow)
        vHigh = age_scaler.unscale(vHigh)
        return 'onset reestimate (mean [20%, 80%]) = {:.3g} [{:.3g},{:.3g}]'.format(mu,vLow,vHigh)
    
    create_html(
        data, fitter, fits, basedir, gene_dir, series_dir,
        gene_names = top_genes, 
        region_names = regions,
        extra_columns = [('p-value',top_pvals), ('q-value',top_qvals)],
        extra_fields_per_fit = [get_onset_time, get_onset_dist],
        b_inline_images = True,
        inline_image_size = '30%',
        b_R2_dist = False, 
        ttl = 'Fit for genes with top t-test scores',
        top_text = top_text,
        filename = 'gradual-maturation-t-test' + filename_suffix,
    )
    
cfg.verbosity = 1
age_scaler = LogScaler()

lst_pathways = [
    'serotonin',
    'dopamine',
]

for pathway in lst_pathways:
    data = GeneData.load('both').restrict_pathway(pathway).restrict_ages('EF3',PCW(10)).scale_ages(age_scaler)
    shape = Sigmoid(priors='sigmoid_wide')
    fitter = Fitter(shape, sigma_prior='normal')
    fits = get_all_fits(data, fitter, allow_new_computation=False)
    ds_fits = fits['kang2011']
    
    for b_reversed in [False,True]:
        regions = ['V1C', 'OFC']
        if b_reversed:
            regions = regions[::-1]
    
        scores = []
        for i,g in enumerate(data.gene_names):
            mu1 = ds_fits[(g,regions[0])].theta_samples[2,:]
            mu2 = ds_fits[(g,regions[1])].theta_samples[2,:]
            t,pval = ttest_ind(mu1,mu2)
            if mu1.mean() < mu2.mean(): # make it one sided: V1C < OFC
                pval = pval/2
            else:
                pval = 1 - pval/2
            scores.append( (g,pval) )
        
        # add FDR correction
        _,qvals = fdr([pval for g,pval in scores])
        scores = [(g,pval,qval) for (g,pval),qval in zip(scores,qvals)]
        
        filename_suffix = '-reversed' if b_reversed else ''
        create_top_genes_html(data,fitter,fits,scores,regions,filename_suffix=filename_suffix)
