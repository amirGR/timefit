import setup
import argparse
from os.path import join
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from project_dirs import results_dir
from single_region import SingleRegion
import pathway_lists

def plot_pathway(pathway, order, idx, timing):
    plt.figure()
    ax = plt.gca()
    ax.plot(idx,timing,'bx')
    ax.set_xlim(-0.5,len(order)-0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_title(pathway)    

def save_scores(singles, scores, order):
    filename = join(results_dir(), 'pathway-spearman.txt')
    print 'Saving ordering results to {}'.format(filename)
    with open(filename,'w') as f:
        print >>f, 'Region Order: {}'.format(' '.join(order))
        header = '{:<60}{:<7}{:<15}{:<10}{:<10}'.format('pathway', 'nGenes', '-log10(pval)', 'pval', 'Spearman r')
        print >>f, header
        print >>f, '-'*len(header)
        for logpval, pval, sr, pathway in scores:
            pathway_size = len(singles.pathways[pathway])
            if len(pathway) > 55:
                pathway = pathway[:55] + '...'
            print >>f, '{pathway:<60}{pathway_size:<7}{logpval:<15.3g}{pval:<10.3g}{sr:<10.3g}'.format(**locals())

def timing_vs_region_order(singles, b_cortex_only):
    if b_cortex_only: 
        order = 'V1C S1C M1C DFC OFC'.split()
    else:
        order = 'MD STR V1C OFC'.split()
    pathways_to_plot = [
        'sensory perception of smell',
    ]

    scores = [] # (pval,pathway)
    for pathway, genes in singles.pathways.iteritems():
        idx_timing_pairs = []
        for g in genes:
            ig = singles.g2i[g]            
            for i,r in enumerate(order):
                ir = singles.r2i[r]
                t = singles.mu[ig,ir]
                idx_timing_pairs.append( (i,t) )
        idx,timing = zip(*idx_timing_pairs)
        if pathway in pathways_to_plot:
            plot_pathway(pathway, order, idx, timing)
        sr,pval = spearmanr(idx,timing)
        scores.append( (-np.log10(pval), pval, sr, pathway) )
    scores.sort(reverse=True) 
    save_scores(singles, scores, order)
    
##############################################################
# main
##############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='Pathways list name. Default=brain_go_num_genes_min_15', default='brain_go_num_genes_min_15', choices=['all'] + pathway_lists.all_pathway_lists())
    parser.add_argument('--cortex_only', help='Use only cortical regions', action='store_true')
    args = parser.parse_args()

    singles = SingleRegion(args.list)
    timing_vs_region_order(singles, args.cortex_only)
