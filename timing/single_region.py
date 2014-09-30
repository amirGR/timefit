import setup
from os.path import join
import numpy as np
from project_dirs import cache_dir
from utils.misc import load_pickle
import pathway_lists 

class SingleRegion(object):
    def __init__(self, listname='all'):
        self.listname = listname
        self.pathways = pathway_lists.read_all_pathways(listname)

        self.change_dist = load_pickle(
            filename = join(cache_dir(), 'both', 'fits-log-all-sigmoid-theta-sigmoid_wide-sigma-normal-change-dist.pkl'), 
            name = 'change distribution for all genes and regions'
        )

        self.genes = self.change_dist.genes
        self.regions = self.change_dist.regions
        self.g2i = {g:i for i,g in enumerate(self.genes)}
        self.r2i = {r:i for i,r in enumerate(self.regions)}
        self.age_scaler = self.change_dist.age_scaler
        self.mu = self.change_dist.mu
        self.std = self.change_dist.std
        self.bin_edges = self.change_dist.bin_edges
        self.bin_centers = self.change_dist.bin_centers
        self.weights = self.change_dist.weights

    def region_timings_per_pathway(self):
        def mean_age(pathway_genes, r):
            pathway_ig = [self.g2i[g] for g in pathway_genes]
            ir = self.r2i[r]
            ages = self.mu[pathway_ig,ir]
            weights = 1/self.std[pathway_ig,ir]
            age = np.dot(weights,ages) / sum(weights)
            return self.age_scaler.unscale(age)

        res = {} # pathway -> { r -> mu }
        for pathway in self.pathways.iterkeys():
            pathway_genes = self.pathways[pathway]
            res[pathway] = {r : mean_age(pathway_genes, r) for r in self.regions}
        return res

