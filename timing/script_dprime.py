import setup
import argparse
from region_pairs import RegionPairTiming
import pathway_lists

##############################################################
# main
##############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='Pathways list name. Default=brain_go_num_genes_min_15', default='brain_go_num_genes_min_15', choices=['all'] + pathway_lists.all_pathway_lists())
    parser.add_argument('--include', help='whitespace separated list of regions to include (region pair included if at least one of the regions is in the list). Default=all')
    parser.add_argument('--both', help='whitespace separated list of regions to include (region pair included if BOTH of the regions are in the list). Default=all')
    parser.add_argument('--exclude', default='PFC', help='whitespace separated list of regions to exclude. Default=PFC')
    parser.add_argument('-f', '--force', help='Force recomputation of pathway dprime measures', action='store_true')
    parser.add_argument('--mat', help='Export analysis to mat file', action='store_true')
    args = parser.parse_args()
    if args.exclude is not None:
        args.exclude = args.exclude.split()
    if args.include is not None:
        args.include = args.include.split()
    if args.both is not None:
        args.both = args.both.split()
    
    timing = RegionPairTiming(args.list)
    res = timing.analyze_all_pathways(force=args.force).filter_regions(exclude=args.exclude, include=args.include, include_both=args.both)
    d = res.get_by_pathway()
    if args.mat:
        res.save_to_mat()
    res.save_top_results()
