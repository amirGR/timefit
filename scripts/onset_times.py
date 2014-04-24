import setup
from os.path import join
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData, load_17_pathways_breakdown
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits, restrict_genes
from scalers import LogScaler
from dev_stages import dev_stages
from plots import save_figure
import project_dirs
from utils.misc import ensure_dir

cfg.verbosity = 1
age_scaler = LogScaler()

def get_fits():
    data = GeneData.load('both').restrict_pathway('17pathways').scale_ages(age_scaler)
    shape = Sigmoid(priors='sigmoid_wide')
    fitter = Fitter(shape, sigma_prior='normal')
    fits = get_all_fits(data, fitter)
    return fits
    
def get_fit_param(fits, getter, R2_threshold=None):
    lst = []
    for dsfits in fits.itervalues():
        for fit in dsfits.itervalues():
            if R2_threshold is None or fit.LOO_score > R2_threshold:
                val = getter(fit)
                if val is not None:
                    lst.append(val)
    return lst

def plot_onset_times(onset_times, pathway_name, R2_threshold):
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    low = min(stage.from_age for stage in stages)
    high = max(stage.to_age for stage in stages)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(onset_times, 50, range=(low,high))
    ttl = 'Histogram of onset times for {} (R2 threshold={})'.format(pathway_name, R2_threshold)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_xlabel('onset time', fontsize=cfg.fontsize)
    ax.set_ylabel('count', fontsize=cfg.fontsize)    

    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=cfg.xtick_fontsize, fontstretch='condensed', rotation=90)    
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    # save the figure
    dirname = join(project_dirs.results_dir(), 'onset-times')
    if 'unique' in pathway_name:
        dirname = join(dirname,'unique')
    elif '17' not in pathway_name:
        dirname = join(dirname,'overlapping')
    ensure_dir(dirname)
    filename = join(dirname, pathway_name + '.png')
    print 'Saving figure to {}'.format(filename)
    save_figure(fig, filename, b_close=True)

def unique_genes_only(dct_pathways):
    res = {}
    def count(dct,g):
        return sum(1 for pathway_genes in dct.itervalues() if g in pathway_genes)
    for pathway_name,genes in dct_pathways.iteritems():
        dct_counts = {g:count(dct_pathways,g) for g in genes}
        unique_genes = {g for g,c in dct_counts.iteritems() if c == 1}
        res[pathway_name] = unique_genes
    return res

def main():
    R2_threshold = 0.5
    dct_pathways = load_17_pathways_breakdown()
    dct_unique = unique_genes_only (dct_pathways)
    for pathway_name, genes in dct_unique.iteritems():
        dct_pathways[pathway_name + ' (unique)'] = genes
        
    fits = get_fits()
    def get_onset_time(fit):
        if fit.theta is None:
            return None
        a,h,mu,w = fit.theta
        return mu
    dct_pathways['17 pathways'] = None
    for pathway_name, genes in dct_pathways.iteritems():
        print 'Doing {}...'.format(pathway_name)
        pathway_fits = restrict_genes(fits,genes)
        onset_times = get_fit_param(pathway_fits, get_onset_time, R2_threshold=R2_threshold)
        if not onset_times:
            print 'Skipping {}. No fits left'.format(pathway_name)
            continue
        plot_onset_times(onset_times, pathway_name, R2_threshold)

if __name__ == '__main__':
    main()