import setup
from os.path import join
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits
from scalers import LogScaler
from plots import save_figure, create_html
from project_dirs import results_dir, fit_results_relative_path
from utils.misc import ensure_dir

fontsize = 30

def plot_correlation_histogram(scores,pathway):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lst_r = [r for g,r,pval,lst_R2 in scores]

    ax.hist(lst_r)
    ax.set_title('Gradual maturation for {}'.format(pathway), fontsize=fontsize)
    ax.set_xlabel('Spearman r', fontsize=fontsize)
    ax.set_ylabel('number of genes', fontsize=fontsize)   
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig

def plot_scatter(scores, pathway, fR2):
    lst_r = [abs(r) for g,r,pval,lst_R2 in scores]
    R2 = [fR2(lst_R2) for g,r,pval,lst_R2 in scores]
    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.scatter(R2,lst_r)
    ax.set_xlim(-1,1)
    ax.set_ylim(0,1)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title('Gradual maturation for {}'.format(pathway), fontsize=fontsize)
    ax.set_xlabel('{} $R^2$'.format(fR2.__name__), fontsize=fontsize)
    ax.set_ylabel('abs(r)', fontsize=fontsize)
    return fig

def create_top_correlations_html(data, fitter, fits, scores, regions, n_top=None):
    if n_top is None:
        n_top = len(scores)
        
    basedir = join(results_dir(), fit_results_relative_path(data,fitter))
    ensure_dir(basedir)
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'

    def key_func(score):
        g,r,pval,lst_R2 = score
        return r
    scores.sort(key=key_func)
    top_genes = [g for g,r,pval,lst_R2 in scores[:n_top]]
    top_scores = {g:r for g,r,pval,lst_R2 in scores[:n_top]}
    top_pvals = {g:pval for g,r,pval,lst_R2 in scores[:n_top]}
    
    def get_onset_time(fit):
        a,h,mu,_ = fit.theta
        age = age_scaler.unscale(mu)
        txt = 'onset = {:.3g} years'.format(age)
        cls = ''
        return txt,cls
    
    create_html(
        data, fitter, fits, basedir, gene_dir, series_dir,
        gene_names = top_genes, 
        region_names = regions,
        extra_columns = [('r',top_scores),('p-value',top_pvals)],
        extra_fields_per_fit = [get_onset_time],
        b_inline_images = True,
        b_R2_dist = False, 
        ttl = 'Fit for genes with top Spearman correlations',
        filename = 'top-gradual-maturation',
    )
    
cfg.verbosity = 1
age_scaler = LogScaler()

def get_gene_correlation(fits, gene, regions):
    # XXX fits should be changed to a class which support indexing by (g,r) and hides datasets
    ds_fits = fits['kang2011'] 
    def get_onset_time(r):
        fit = ds_fits[(gene,r)]
        a,h,mu,_ = fit.theta
        return mu,fit.LOO_score
    lst_mu_R2 = [get_onset_time(r) for r in regions]
    onset_times, lst_R2 = zip(*lst_mu_R2)
    r,pval = spearmanr(onset_times, range(len(regions)))
    return r,pval,lst_R2

lst_pathways = [
    'serotonin',
    'dopamine',
]

for pathway in lst_pathways:
    data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)
    shape = Sigmoid(priors='sigmoid_wide')
    fitter = Fitter(shape, sigma_prior='normal')
    fits = get_all_fits(data, fitter, allow_new_computation=False)
    # R2_threshold = 0.5 YYY problem - we might be using bad fits.
    
    regions = ['OFC', 'M1C', 'S1C', 'IPC', 'V1C']
    
    scores = []
    for g in data.gene_names:
        r,pval,lst_R2 = get_gene_correlation(fits,g,regions)
        scores.append( (g,r,pval,lst_R2) )
    
    fig = plot_correlation_histogram(scores,pathway)
    save_figure(fig,'{}/gradual-maturation-hist.png'.format(pathway,pathway), under_results=True, b_close=True)
    
    for fR2 in [np.mean]: #[min,max,np.mean]:
        fig = plot_scatter(scores, pathway, fR2)
        save_figure(fig,'{}/gradual-maturation-scatter-{}.png'.format(pathway,fR2.__name__), under_results=True, b_close=True)
    
    create_top_correlations_html(data,fitter,fits,scores,regions)
