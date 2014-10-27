import setup
from os.path import join
from collections import defaultdict
import numpy as np
from scipy.io import savemat
from scipy.stats import nanmean
from sklearn.datasets.base import Bunch
from project_dirs import cache_dir, results_dir
from utils.misc import z_score_to_p_value, cache, load_pickle
from utils.formats import list_of_strings_to_matlab_cell_array
from single_region import SingleRegion

##############################################################
# RegionPairTiming
##############################################################
class RegionPairTiming(object):
    cube_filename = join(cache_dir(), 'both', 'fits-log-all-sigslope-theta-sigslope80-sigma-normal-dprime-cube.pkl')
    
    def __init__(self, listname='all'):
        self.listname = listname
        self.single = SingleRegion(listname)
        self.pathways = self.single.pathways
        self.genes = self.single.genes
        self.regions = self.single.regions
        self.g2i = self.single.g2i
        self.r2i = self.single.r2i
        self.age_scaler = self.single.age_scaler
        self.mu = self.single.mu
        self.single_std = self.single.std

        cube = load_pickle(RegionPairTiming.cube_filename, name='timing d-prime info for all genes and region pairs')
        self.d_mu = cube.d_mu
        self.pair_std = cube.std
        self.scores = self.d_mu / self.pair_std

        self.baseline = self.baseline_distribution_all_pairs(100, 10000)

    @cache(lambda self: join(cache_dir(), 'both', 'dprime-all-pathways-and-regions-{}.pkl'.format(self.listname)))
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
        return TimingResults.fromResultsDct(res, self.listname, self.pathways)

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
        pathway_pair_std = self.pair_std[pathway_ig,ir1,ir2]
        weights = 1/pathway_pair_std
        valid = ~np.isnan(pathway_d_mu) # needed for the PFC region from colantuoni which doesn't contain all genes\
        weights, pathway_d_mu = weights[valid], pathway_d_mu[valid]
        weighted_delta = np.dot(weights, pathway_d_mu) / sum(weights)
        delta = np.mean(pathway_d_mu)
        too_many_nans = False
        if not valid.all():
            assert r1 == 'PFC' or r2 == 'PFC', "r1={}, r2={}".format(r1,r2)
            n_genes = len(valid)
            n_non_valid = n_genes - np.count_nonzero(valid)
            if float(n_non_valid)/n_genes > 0.05:
                too_many_nans = True
        def mean_age(ir):
            if too_many_nans:
                return np.NaN            
            ages = self.mu[pathway_ig,ir]
            weights = 1/self.single_std[pathway_ig,ir]
            valid = ~np.isnan(weights)
            weights, ages = weights[valid], ages[valid]
            age = np.dot(weights,ages) / sum(weights)
            return self.age_scaler.unscale(age)
        return Bunch(
            score = score if not too_many_nans else np.nan,
            delta = delta if not too_many_nans else np.nan,
            weighted_delta = weighted_delta if not too_many_nans else np.nan,
            mu1_years = mean_age(ir1),
            mu2_years = mean_age(ir2),
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

##############################################################
# TimingResults
##############################################################
class TimingResults(object):
    @staticmethod
    def fromResultsDct(dct_res, listname, pathways):
        def canonical_form(k,v):
            pathway,r1,r2 = k
            if v.score >= 0:
                return Bunch(pathway=pathway, r1=r1, r2=r2, score=v.score, delta=v.delta, weighted_delta=v.weighted_delta, mu1_years=v.mu1_years, mu2_years=v.mu2_years, pval=v.pval, pathway_size=v.pathway_size)
            else:
                return Bunch(pathway=pathway, r1=r2, r2=r1, score=-v.score, delta=-v.delta, weighted_delta=-v.weighted_delta, mu1_years=v.mu2_years, mu2_years=v.mu1_years, pval=v.pval, pathway_size=v.pathway_size)
        flat = [canonical_form(k,v) for k,v in dct_res.iteritems() if not np.isnan(v.score)]
        res = sorted(flat, key=lambda x: -np.log10(x.pval), reverse=True)
        return TimingResults(res, listname, pathways)
        
    def __init__(self, res, listname, pathways, include=None, include_both=None, exclude=None):
        self.res = res[:] # avoid aliasing
        self.listname = listname
        self.pathways = pathways
        self.include = set(include) if include is not None else None
        self.include_both = set(include_both) if include_both is not None else None
        self.exclude = set(exclude) if exclude is not None else None
        self._apply_region_filter()
        
    def _apply_region_filter(self):
        include = self.include if self.include is not None else self.all_regions
        include_both = self.include_both if self.include_both is not None else self.all_regions
        exclude = self.exclude if self.exclude is not None else set()
        self.res = [x for x in self.res if (x.r1 in include or x.r2 in include) and (x.r1 in include_both and x.r2 in include_both) and not (x.r1 in exclude or x.r2 in exclude)]

    @property
    def all_regions(self):
        return set(x.r1 for x in self.res) | set(x.r2 for x in self.res)

    @property
    def all_pathways(self):
        return set(x.pathway for x in self.res)
        
    @property
    def filename_suffix(self):
        suffix = self.listname
        if self.include is not None:
            suffix += '-only-' + '-'.join(sorted(self.include))
        if self.include_both is not None:
            suffix += '-onlyboth-' + '-'.join(sorted(self.include_both))
        if self.exclude is not None:
            suffix += '-no-' + '-'.join(sorted(self.exclude))
        return suffix
        
    def filter_regions(self, include=None, include_both=None, exclude=None):
        if self.include is not None:
            if include is None:
                include = self.include
            else:
                include = self.include & set(include)
        if self.include_both is not None:
            if include_both is None:
                include_both = self.include_both
            else:
                include_both = self.include_both & set(include_both)
        if self.exclude is not None:
            if exclude is None:
                exclude = self.exclude
            else:
                exclude = self.exclude | set(exclude)
        return TimingResults(self.res, self.listname, self.pathways, include=include, include_both=include_both, exclude=exclude)            

    def get_by_pathway(self):
        d = defaultdict(list)
        for x in self.res:
            d[x.pathway].append(x)
        return d
        
    def save_to_mat(self):
        filename = join(cache_dir(), 'both', 'dprime-all-pathways-and-regions-{}.mat'.format(self._filename_suffix))
        mdict = dict(
            pathway = list_of_strings_to_matlab_cell_array([x.pathway for x in self.res]),
            r1 = list_of_strings_to_matlab_cell_array([x.r1 for x in self.res]),
            r2 = list_of_strings_to_matlab_cell_array([x.r2 for x in self.res]),
            score = np.array([x.score for x in self.res]),
            delta = np.array([x.delta for x in self.res]),
            weighted_delta = np.array([x.weighted_delta for x in self.res]),
            mu1_years = np.array([x.mu1_years for x in self.res]),
            mu2_years = np.array([x.mu2_years for x in self.res]),
            pval = np.array([x.pval for x in self.res]),
            pathway_size = np.array([x.pathway_size for x in self.res]),
        )
        print 'Saving results to {}'.format(filename)
        savemat(filename, mdict, oned_as='column')

    def save_top_results(self, n=50):
        filename = join(results_dir(), 'dprime-top-results-{}.txt'.format(self.filename_suffix))
        print 'Saving top {} results to {}'.format(n,filename)
        with open(filename,'w') as f:
            header = '{:<60}{:<7}{:<5}{:<5}{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('pathway', 'nGenes', 'r1', 'r2', '-log10(pval)', 'score', 'delta', 'w-delta', 'mu1 yrs', 'mu2 yrs')
            print >>f, header
            print >>f, '-'*len(header)
            for x in self.res[:n]:
                logpval = -np.log10(x.pval)
                pathway = x.pathway
                if len(pathway) > 55:
                    pathway = pathway[:55] + '...'
                print >>f, '{pathway:<60}{x.pathway_size:<7}{x.r1:<5}{x.r2:<5}{logpval:<15.3g}{x.score:<10.3g}{x.delta:<10.3g}{x.weighted_delta:<10.3g}{x.mu1_years:<10.3g}{x.mu2_years:<10.3g}'.format(**locals())

