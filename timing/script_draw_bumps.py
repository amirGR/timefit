import setup
import argparse
from os.path import join
import matplotlib.pyplot as plt
import config as cfg
from single_region import SingleRegion
import pathway_lists
from plots import save_figure, add_age_ticks

def draw_bumps(singles, pathway, regions):
    if regions is None:
        regions = singles.regions
    pathway_genes = singles.pathways[pathway]
    pathway_ig = [singles.g2i[g] for g in pathway_genes]
    weights = singles.weights[pathway_ig,:,:].mean(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for r in regions:
        ir = singles.r2i[r]
        ax.plot(singles.bin_centers, weights[ir,:], linewidth=3, label=r)
    add_age_ticks(ax, singles.age_scaler)
    ax.set_yticks([])
    ax.set_ylabel('Strength of change', fontsize=cfg.fontsize)
    ax.legend(frameon=False, fontsize=12)
    ax.set_title(pathway, fontsize=cfg.fontsize)
    return fig

##############################################################
# main
##############################################################
if __name__ == '__main__':
    cfg.verbosity = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='Pathways list name. Default=brain_go_num_genes_min_15', default='brain_go_num_genes_min_15', choices=['all'] + pathway_lists.all_pathway_lists())
    parser.add_argument('--regions', help='List  (whitespace separated) of regions to plot. Default=all regions')
    args = parser.parse_args()
    if args.regions is not None:
        args.regions = args.regions.split()
    
    singles = SingleRegion(args.list)
    for pathway in singles.pathways.iterkeys():
        fig = draw_bumps(singles, pathway, args.regions)
        dirname = singles.listname
        if args.regions is not None:
            dirname = '{}-{}'.format(dirname, '-'.join(args.regions))
        filename = join('bumps', dirname, pathway + '.png')
        save_figure(fig, filename, b_close=True, under_results=True)
