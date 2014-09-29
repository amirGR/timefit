import setup
import argparse
from os.path import join
from project_dirs import results_dir
from single_region import SingleRegion
import pathway_lists

def compute_region_ordering(singles):
    timings = singles.region_timings_per_pathway() # pathway -> { r -> mu }
    sorted_timings = {} # pathway -> list of regions (sorted by mu)
    for pathway, dct in timings.iteritems():
        sorted_regions_and_times = sorted((mu,r) for r,mu in dct.iteritems())
        sorted_timings[pathway] = [r for mu,r in sorted_regions_and_times]

    filename = join(results_dir(), 'dprime-region-ordering-{}.txt'.format(singles.listname))
    print 'Saving ordering results to {}'.format(filename)
    with open(filename,'w') as f:
        header = '{:<60}{:<7}{}'.format('pathway', 'nGenes', 'Regions (early to late)')
        print >>f, header
        print >>f, '-'*len(header)
        for pathway, ordered_regions in sorted_timings.iteritems():
            pathway_size = len(singles.pathways[pathway])
            if len(pathway) > 55:
                pathway = pathway[:55] + '...'
            ordered_regions = ' '.join(ordered_regions)
            print >>f, '{pathway:<60}{pathway_size:<7}{ordered_regions}'.format(**locals())

##############################################################
# main
##############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='Pathways list name. Default=brain_go_num_genes_min_15', default='brain_go_num_genes_min_15', choices=['all'] + pathway_lists.all_pathway_lists())
    args = parser.parse_args()
    
    singles = SingleRegion(args.list)
    sorted_timings = compute_region_ordering(singles)
