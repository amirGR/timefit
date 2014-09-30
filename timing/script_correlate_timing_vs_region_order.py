import setup
import argparse
from os.path import join
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import config as cfg
from project_dirs import results_dir
from single_region import SingleRegion
import pathway_lists
from plots import save_figure

def plot_pathway(singles, pathway, order):
    lst_ir = [singles.r2i[r] for r in order]
    pathway_genes = singles.pathways[pathway]
    pathway_ig = [singles.g2i[g] for g in pathway_genes]
    pathway_mu = singles.mu[pathway_ig,:]
    pathway_mu = pathway_mu[:,lst_ir]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for row in pathway_mu:
        ax.plot(range(len(row)),row,'x-')
    ax.set_xlim(-0.5,len(order)-0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_title(pathway)
    return fig

def save_scores(singles, scores, order):
    filename = join(results_dir(), 'pathway-spearman-{}.txt'.format('-'.join(order)))
    print 'Saving ordering results to {}'.format(filename)
    with open(filename,'w') as f:
        print >>f, 'Region Order: {}'.format(' '.join(order))
        header = '{:<60}{:<7}{:<15}{:<10}{:<15}'.format('pathway', 'nGenes', '-log10(pval)', 'pval', 'Spearman rho')
        print >>f, header
        print >>f, '-'*len(header)
        for logpval, pval, sr, pathway in scores:
            pathway_size = len(singles.pathways[pathway])
            if len(pathway) > 55:
                pathway = pathway[:55] + '...'
            print >>f, '{pathway:<60}{pathway_size:<7}{logpval:<15.3g}{pval:<10.3g}{sr:<15.3g}'.format(**locals())

def get_permuted_rows(x,rng):
    y = np.empty(x.shape, dtype=x.dtype)
    for i,row in enumerate(x):
        y[i,:] = rng.permutation(x[i])
    return y

def paired_spearman(x):
    """x is a two dimensional array. We check spearman rho (monotoncity) for each row.
       The function returns the average rho across rows and the two sided p-value of that score. 
       The p-value is computed by permutation test.
    """
    def mean_rho(x):
        return np.mean([spearmanr(row,range(len(row)))[0] for row in x])
    rho = mean_rho(x)
    
    rng = np.random.RandomState(cfg.random_seed)
    min_hits = 3
    for n in [10, 10**2, 10**3, 10**4, 10**5]:
        print '  running {} permutations...'.format(n),
        n_more_extreme = 0
        for _ in xrange(n):
            y = get_permuted_rows(x,rng)
            rho_i = mean_rho(y)
            if abs(rho_i) >= abs(rho):
                n_more_extreme += 1
        pval = float(n_more_extreme)/n
        print '  pval={:.3g}'.format(pval)
        if n_more_extreme >= min_hits:
            break
    else:
        print 'NOTE: APPROXIMATE P-VALUE. Permutation test found less than {} "hits"'.format(min_hits)
    return rho, pval

def timing_vs_region_order(singles, order):
    lst_ir = [singles.r2i[r] for r in order]
    scores = [] # (pval,pathway)
    for i, (pathway, pathway_genes) in enumerate(singles.pathways.iteritems()):
        print '[{}/{}] Computing score for {}'.format(i+1, len(singles.pathways),pathway)
        pathway_ig = [singles.g2i[g] for g in pathway_genes]
        pathway_mu = singles.mu[pathway_ig,:]
        pathway_mu = pathway_mu[:,lst_ir]
        rho, pval = paired_spearman(pathway_mu)
        scores.append( (-np.log10(pval), pval, rho, pathway) )
    scores.sort(reverse=True) 
    save_scores(singles, scores, order)
    
##############################################################
# main
##############################################################
if __name__ == '__main__':
    cfg.verbosity = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='Pathways list name. Default=brain_go_num_genes_min_15', default='brain_go_num_genes_min_15', choices=['all'] + pathway_lists.all_pathway_lists())
    parser.add_argument('--cortex_only', help='Use only cortical regions', action='store_true')
    parser.add_argument('--draw', help='Draw plot for this pathway and exit')
    args = parser.parse_args()

    if args.cortex_only:
        order = 'V1C A1C S1C M1C DFC MFC OFC'.split()
    else:
        order = 'MD STR V1C OFC'.split()

    singles = SingleRegion(args.list)
    if args.draw is None:
        timing_vs_region_order(singles, order)
    else:
        pathway = args.draw
        fig = plot_pathway(singles, pathway, order)
        filename = 'spearman-{}.png'.format(pathway)
        save_figure(fig, filename, under_results=True)
