import setup
import pickle
from os.path import join
from os import listdir
import numpy as np
from scipy.io import savemat
from scipy.stats import nanmean
from sklearn.datasets.base import Bunch
from project_dirs import cache_dir, pathways_dir, results_dir
from utils.misc import z_score_to_p_value, cache
from utils.formats import list_of_strings_to_matlab_cell_array

class RegionPairTiming(object):
    def __init__(self):
        self.pathways = self.read_all_pathways()

        info = self.read_timing_info()
        self.genes = info['genes']
        self.regions = info['regions']
        self.g2i = {g:i for i,g in enumerate(self.genes)}
        self.r2i = {r:i for i,r in enumerate(self.regions)}
        self.age_scaler = info['age_scaler']
        self.mu = info['mu']
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
                    if r2 <= r1: # keep only results "above the diagonal" (r1 < r2 lexicographically)
                        continue
                    pathway_res = self.analyze_pathway_and_region_pair(pathway_genes, r1, r2)
                    res[(pathway,r1,r2)] = pathway_res
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

        pathway_d_mu = self.d_mu[pathway_ig,ir1,ir2]
        pathway_std = self.std[pathway_ig,ir1,ir2]
        weights = 1/pathway_std
        valid = ~np.isnan(pathway_d_mu) # needed for the PFC region from colantuoni which doesn't contain all genes\
        weighted_delta = np.dot(weights[valid], pathway_d_mu[valid]) / sum(weights[valid])
        delta = nanmean(pathway_d_mu)
        too_many_nans = False
        if not valid.all():
            assert r1 == 'PFC' or r2 == 'PFC', "r1={}, r2={}".format(r1,r2)
            n_genes = len(valid)
            n_non_valid = n_genes - np.count_nonzero(valid)
            if float(n_non_valid)/n_genes > 0.05:
                too_many_nans = True
        return Bunch(
            score = score if not too_many_nans else np.nan,
            delta = delta if not too_many_nans else np.nan,
            weighted_delta = weighted_delta if not too_many_nans else np.nan,
            mu1_years = self.age_scaler.unscale(nanmean(self.mu[pathway_ig,ir1])) if not too_many_nans else np.nan,
            mu2_years = self.age_scaler.unscale(nanmean(self.mu[pathway_ig,ir2])) if not too_many_nans else np.nan,
            pval = pval if not too_many_nans else np.nan,
            pathway_size = len(pathway_genes),
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

class TimingResults(object):
    def __init__(self, res):
        self.res = res
        flat = [
            Bunch(pathway=p, r1=r1, r2=r2, score=v.score, delta=v.delta, weighted_delta=v.weighted_delta, mu1_years=v.mu1_years, mu2_years=v.mu2_years, pval=v.pval, pathway_size=v.pathway_size)
            for (p,r1,r2),v in res.iteritems()
            if not np.isnan(v.score)
        ]
        self.sorted_res = sorted(flat, key=lambda x: -np.log10(x.pval), reverse=True)
        
    def save_to_mat(self, filename):
        mdict = dict(
            pathway = list_of_strings_to_matlab_cell_array([x.pathway for x in self.sorted_res]),
            r1 = list_of_strings_to_matlab_cell_array([x.r1 for x in self.sorted_res]),
            r2 = list_of_strings_to_matlab_cell_array([x.r2 for x in self.sorted_res]),
            score = np.array([x.score for x in self.sorted_res]),
            delta = np.array([x.delta for x in self.sorted_res]),
            weighted_delta = np.array([x.weighted_delta for x in self.sorted_res]),
            mu1_years = np.array([x.mu1_years for x in self.sorted_res]),
            mu2_years = np.array([x.mu2_years for x in self.sorted_res]),
            pval = np.array([x.pval for x in self.sorted_res]),
            pathway_size = np.array([x.pathway_size for x in self.sorted_res]),
        )
        
        print 'Saving results to {}'.format(filename)
        savemat(filename, mdict, oned_as='column')

    def save_top_results(self, n=50):
        filename = join(results_dir(), 'dprime-top-results.txt')
        print 'Saving top {} results to {}'.format(n,filename)
        with open(filename,'w') as f:
            header = '{:<55}{:<5}{:<5}{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('pathway', 'r1', 'r2', '-log10(pval)', 'score', 'delta', 'w-delta', 'mu1 yrs', 'mu2 yrs')
            print >>f, header
            print >>f, '-'*len(header)
            for x in self.sorted_res[:n]:
                logpval = -np.log10(x.pval)
                print >>f, '{x.pathway:<55}{x.r1:<5}{x.r2:<5}{logpval:<15.3g}{x.score:<10.3g}{x.delta:<10.3g}{x.weighted_delta:<10.3g}{x.mu1_years:<10.3g}{x.mu2_years:<10.3g}'.format(**locals())

timing = RegionPairTiming()
res = TimingResults(timing.analyze_all_pathways(force=True))
res.save_to_mat(join(cache_dir(), 'both', 'dprime-all-pathways-and-regions.mat'))
res.save_top_results()

