import setup
import pickle
from os.path import join
from os import listdir
import numpy as np
from scipy.stats import nanmean
from sklearn.datasets.base import Bunch
from project_dirs import cache_dir, pathways_dir
from utils.misc import z_score_to_p_value, cache

class RegionPairTiming(object):
    def __init__(self):
        self.pathways = self.read_all_pathways()

        info = self.read_timing_info()
        self.genes = info['genes']
        self.regions = info['regions']
        self.g2i = {g:i for i,g in enumerate(self.genes)}
        self.r2i = {r:i for i,r in enumerate(self.regions)}

        self.d_mu = info['d_mu']
        self.std = info['std']
        self.scores = self.d_mu / self.std

        self.baseline = self.baseline_distribution_all_pairs(100, 10000)

    @cache(filename = join(cache_dir(), 'both', 'dprime-all-pathways-and-regions.pkl'))
    def analyze_all_pathways(self):
        res = {} # (pathway,r1,r2) -> timing results
        for pathway in self.pathways.iterkeys():
            print 'Analyzing region pairs for pathway {}'.format(pathway)
            pathway_genes = self.pathways[pathway]
            for r1 in self.regions:
                for r2 in self.regions:
                    res[(pathway,r1,r2)] = self.analyze_pathway_and_region_pair(pathway_genes, r1, r2)
        return res

    def analyze_pathway_and_region_pair(self, pathway_genes, r1, r2):
        ir1, ir2 = self.r2i[r1], self.r2i[r2]
        pathway_ig = [self.g2i[g] for g in pathway_genes]  
        
        all_pathway_scores = self.scores[pathway_ig, ir1, ir2]
        score = nanmean(all_pathway_scores)
        mu, sigma = self.baseline[(r1,r2)]
        sigma = sigma / np.sqrt(len(pathway_ig))
        z = (score - mu) / sigma
        pval = z_score_to_p_value(z)

        pathway_d_mu = [self.d_mu[ig,ir1,ir2] for ig in pathway_ig]
        pathway_std = [self.std[ig,ir1,ir2] for ig in pathway_ig]
        delta = sum(pathway_d_mu)
        weights = [1/sigma for sigma in pathway_std] / sum(pathway_std)
        weighted_delta = sum([w*d_mu_i for w,d_mu_i in zip(weights,pathway_d_mu)])
        return Bunch(
            score = score,
            delta = delta,
            weighted_delta = weighted_delta,
            pval = pval,
        )

    @cache(filename = join(cache_dir(), 'both', 'dprime-baseline.pkl'))
    def baseline_distribution_all_pairs(self, sample_size, n_samples):
        res = {}
        for r1 in self.regions:
            print 'Sampling baseline distribution of {} vs. all other regions'.format(r1)
            for r2 in self.regions:
                if (r2,r1) in res:
                    mu,sigma = res[(r2,r1)]
                    res[(r1,r2)] = -mu,sigma
                else:
                    res[(r1,r2)] = self.baseline_distribution_one_pair(r1, r2, sample_size, n_samples)
        return res

    def baseline_distribution_one_pair(self, r1, r2, sample_size, n_samples):
        ir1, ir2 = self.r2i[r1], self.r2i[r2]
        pair_scores = self.scores[:,ir1,ir2]
        x = np.empty(n_samples)
        for i in xrange(n_samples):
            inds = np.random.random_integers(0, len(pair_scores)-1, sample_size)
            x[i] = nanmean(pair_scores[inds])
        mu = x.mean()
        sigma = x.std() * np.sqrt(sample_size)
        return mu,sigma

    def read_timing_info(self):
        filename = join(cache_dir(), 'both', 'fits-log-all-sigmoid-theta-sigmoid_wide-sigma-normal-dprime.pkl')
        print 'loading timing d-prime info for all genes and region pairs from {}'.format(filename)
        with open(filename) as f:
            info = pickle.load(f)
        return info

    def read_all_pathways(self):
        pathway_names = [f[:-4] for f in listdir(pathways_dir()) if f.endswith('.txt')]
        return {pathway: self.read_gene_names(pathway) for pathway in pathway_names}
    
    def read_gene_names(self, pathway):
        filename = join(pathways_dir(), pathway + '.txt')
        with open(filename) as f:
            lines = f.readlines()
        genes = [x.strip() for x in lines] # remove newlines
        return [x for x in genes if x] # rmeove empty strings

timing = RegionPairTiming()
res = timing.analyze_all_pathways()