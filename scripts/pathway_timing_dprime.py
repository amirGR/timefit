import setup
import argparse
import cPickle as pickle
from os.path import join
from os import listdir
from collections import defaultdict
import numpy as np
from scipy.io import savemat
from scipy.stats import nanmean
from sklearn.datasets.base import Bunch
from project_dirs import cache_dir, pathways_dir, pathway_lists_dir, results_dir
from utils.misc import z_score_to_p_value, cache
from utils.formats import list_of_strings_to_matlab_cell_array
from load_data import load_kang_tree_distances

class RegionPairTiming(object):
    def __init__(self, listname='all'):
        self.listname = listname
        self.pathways = self.read_all_pathways(listname)

        info = self.read_timing_info()
        self.genes = info['genes']
        self.regions = info['regions']
        self.g2i = {g:i for i,g in enumerate(self.genes)}
        self.r2i = {r:i for i,r in enumerate(self.regions)}
        self.age_scaler = info['age_scaler']
        self.mu = info['mu']
        self.single_std = info['single_std']
        self.d_mu = info['d_mu']
        self.std = info['std']
        self.scores = self.d_mu / self.std

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
        pathway_std = self.std[pathway_ig,ir1,ir2]
        weights = 1/pathway_std
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

    def region_timings_per_pathway(self):
        def mean_age(pathway_genes, r):
            pathway_ig = [self.g2i[g] for g in pathway_genes]
            ir = self.r2i[r]
            ages = self.mu[pathway_ig,ir]
            weights = 1/self.single_std[pathway_ig,ir]
            age = np.dot(weights,ages) / sum(weights)
            return self.age_scaler.unscale(age)

        res = {} # pathway -> { r -> mu }
        for pathway in self.pathways.iterkeys():
            pathway_genes = self.pathways[pathway]
            res[pathway] = {r : mean_age(pathway_genes, r) for r in self.regions}
        return res

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

    @staticmethod
    def pathway_lists():
        return listdir(pathway_lists_dir())

    def read_all_pathways(self, listname='all'):
        if listname == 'all':
            pathway_names = [f[:-4] for f in listdir(pathways_dir()) if f.endswith('.txt')]
        else:
            listfile = join(pathway_lists_dir(),listname)
            with open(listfile) as f:
                lines = f.readlines()
            pathway_names = [x.strip() for x in lines] # remove newlines
            pathway_names = [x for x in pathway_names if x] # rmeove empty strings
        return {pathway: self.read_gene_names(pathway) for pathway in pathway_names}
    
    def read_gene_names(self, pathway):
        filename = join(pathways_dir(), pathway + '.txt')
        with open(filename) as f:
            lines = f.readlines()
        genes = [x.strip() for x in lines] # remove newlines
        return [x for x in genes if x] # rmeove empty strings

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
            header = '{:<55}{:<7}{:<5}{:<5}{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('pathway', 'nGenes', 'r1', 'r2', '-log10(pval)', 'score', 'delta', 'w-delta', 'mu1 yrs', 'mu2 yrs')
            print >>f, header
            print >>f, '-'*len(header)
            for x in self.res[:n]:
                logpval = -np.log10(x.pval)
                print >>f, '{x.pathway:<55}{x.pathway_size:<7}{x.r1:<5}{x.r2:<5}{logpval:<15.3g}{x.score:<10.3g}{x.delta:<10.3g}{x.weighted_delta:<10.3g}{x.mu1_years:<10.3g}{x.mu2_years:<10.3g}'.format(**locals())

class RegionOrdering(object):
    def __init__(self, timing_results):
        self.timing = timing_results
        self.lst_orders = self._region_ordering()
        
    def _region_ordering(self):
        regions = self.timing.all_regions
        pathways = self.timing.all_pathways
        dct_before = defaultdict(dict) # {pathway -> {region -> set of regions with transition before this region}}
        for pathway in pathways: # make sure dictionary includes all pathways and regions
            for r in regions:
                dct_before[pathway][r] = set()
        for x in self.timing.res: # res is already sorted so r1 is before r2
            dct_before[x.pathway][x.r2].add(x.r1)
        lst_orders = []
        for pathway, dct in dct_before.iteritems():
            ranks = [(r,len(s)) for r,s in dct.iteritems()]
            ranks.sort(key=lambda x: x[1])
            #print '{}: {}'.format(pathway,ranks)
            lst_orders.append( (pathway, [r for r,rank in ranks]) )
        lst_orders.sort(key = lambda x: x[1])
        return lst_orders
        
    def save(self):
        filename = join(results_dir(), 'dprime-region-ordering-{}.txt'.format(self.timing.filename_suffix))
        print 'Saving ordering results to {}'.format(filename)
        with open(filename,'w') as f:
            header = '{:<60}{:<7}{}'.format('pathway', 'nGenes', 'Regions (early to late)')
            print >>f, header
            print >>f, '-'*len(header)
            for pathway,ordered_regions in self.lst_orders:
                pathway_size = len(self.timing.pathways[pathway])
                if len(pathway) > 55:
                    pathway = pathway[:55] + '...'
                ordered_regions = ' '.join(ordered_regions)
                print >>f, '{pathway:<60}{pathway_size:<7}{ordered_regions}'.format(**locals())

def timing_against_region_order(timing):
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    #order = 'AMY HIP MD DFC OFC MFC STC VFC IPC STR S1C M1C CBC ITC V1C A1C'.split() # arbitrary order. replace with a real ordering
    #order = 'V1C S1C M1C DFC OFC'.split()
    order = 'MD STR V1C OFC'.split()

    scores = [] # (pval,pathway)
    for pathway, genes in timing.pathways.iteritems():
        pairs = []
        for g in genes:
            ig = timing.g2i[g]            
            for i,r in enumerate(order):
                ir = timing.r2i[r]
                t = timing.mu[ig,ir]
                pairs.append( (i,t) )
        i,t = zip(*pairs)
        if pathway in ['sensory perception of smell']:
            fig = plt.figure()
            ax = plt.gca()
            ax.plot(i,t,'bx')
            ax.set_xlim(-0.5,4.5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
            ax.set_title(pathway)
        sr,pval = spearmanr(i,t)
        scores.append( (-np.log10(pval), pval, sr, pathway) )
    scores.sort(reverse=True) 
    filename = join(results_dir(), 'pathway-spearman.txt')
    print 'Saving ordering results to {}'.format(filename)
    with open(filename,'w') as f:
        header = '{:<60}{:<7}{:<15}{:<10}{:<10}'.format('pathway', 'nGenes', '-log10(pval)', 'pval', 'Spearman r')
        print >>f, header
        print >>f, '-'*len(header)
        for logpval, pval, sr, pathway in scores:
            pathway_size = len(timing.pathways[pathway])
            if len(pathway) > 55:
                pathway = pathway[:55] + '...'
            print >>f, '{pathway:<60}{pathway_size:<7}{logpval:<15.3g}{pval:<10.3g}{sr:<10.3g}'.format(**locals())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='Pathways list name. Default=brain_go_num_genes_min_15', default='brain_go_num_genes_min_15', choices=['all'] + RegionPairTiming.pathway_lists())
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
    timing_against_region_order(timing)
#    dct_timings_per_pathway = timing.region_timings_per_pathway()
#    res = timing.analyze_all_pathways(force=args.force).filter_regions(exclude=args.exclude, include=args.include, include_both=args.both)
#    d = res.get_by_pathway()
#    if args.mat:
#        res.save_to_mat()
#    res.save_top_results()
#
#    region_ordering = RegionOrdering(res)
#    region_ordering.save()

