import setup
import re
import argparse
from os.path import join, dirname
import numpy as np
from region_pairs import RegionPairTiming
from utils.misc import ensure_dir
from project_dirs import results_dir
import pathway_lists

def save_file(filename, lines):
    lines.append('')
    txt = '\n'.join(lines)
    ensure_dir(dirname(filename))
    print 'Saving to {}'.format(filename)
    with open(filename,'w') as f:
        f.write(txt)

def export_cytoscape(timing, pval_cutoff):
    res = timing.analyze_all_pathways().filter_regions(exclude=['PFC'])
    def safe_pathway_name(pathway): return re.sub(r'\s+','-',pathway)
    def edge_weight(pval): return min(200, int(-50/np.log10(pval)))
    vals = [(x.r1, safe_pathway_name(x.pathway), x.r2, edge_weight(x.pval)) for x in res.res  if -np.log10(x.pval) > pval_cutoff]
    
    lines = ['{} {} {}'.format(r1,pathway,r2) for r1,pathway,r2,w in vals]
    save_file(join(results_dir(), 'cytoscape', 'regions.sif'), lines)
    lines = ['{} ({}) {} = {}'.format(r1, pathway, r2, w) for r1, pathway, r2, w in vals]
    save_file(join(results_dir(), 'cytoscape', 'edge_weights.attrs'), ['EdgeWeights'] + lines)

##############################################################
# main
##############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='Pathways list name. Default=brain_go_num_genes_min_15', default='brain_go_num_genes_min_15', choices=['all'] + pathway_lists.all_pathway_lists())
    parser.add_argument('--pval_cutoff', help='Only write edges where the -log(p-value) is above this threshold. Default=0 (use all edges)', default='0')
    args = parser.parse_args()
    pval_cutoff = float(args.pval_cutoff)
    timing = RegionPairTiming(args.list)
    export_cytoscape(timing, pval_cutoff)
