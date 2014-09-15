import pickle
from os.path import join, splitext, basename, isfile
from collections import defaultdict
import numpy as np
from scipy.io import loadmat
import project_dirs
import config as cfg
from utils.formats import matlab_cell_array_to_list_of_strings, read_strings_from_file
from utils.misc import get_unique

def unique_genes_only(dct_pathways):
    res = {}
    def count(dct,g):
        return sum(1 for pathway_genes in dct.itervalues() if g in pathway_genes)
    for pathway_name,genes in dct_pathways.iteritems():
        dct_counts = {g:count(dct_pathways,g) for g in genes}
        unique_genes = {g for g,c in dct_counts.iteritems() if c == 1}
        res[pathway_name] = unique_genes
    return res

def get_17_pathways_short_names():
    return {
        'Glutamatergic synapse' : 'glutamate', 
        'Amyotrophic lateral sclerosis (ALS)' : 'ALS', 
        "Huntington's disease" : 'huntington', 
        'GABAergic synapse' : 'GABA', 
        'Cocaine addiction' : 'cocaine', 
        'Cholinergic synapse' : 'cholinergic', 
        "Parkinson's disease" : 'parkinsons', 
        'Amphetamine addiction' : 'amphetamine', 
        'Long-term depression' : 'LTD', 
        'Nicotine addiction' : 'nicotine', 
        'Morphine addiction' : 'morphine', 
        'Calcium signaling' : 'calcium', 
        'Dopaminergic synapse' : 'dopamine', 
        'Neuroactive ligand-receptor interaction' : 'ligand-receptor-interaction', 
        'Alcoholism' : 'alcoholism', 
        "Alzheimer's disease" : 'alzheimers', 
        'Serotonergic synapse' : 'serotonin',
    }

def load_17_pathways_breakdown(b_unique=False, short_names=False):
    filename = join(project_dirs.data_dir(),'17pathways-breakdown.pkl')
    with open(filename) as f:
        dct_pathways = pickle.load(f)
    if b_unique:
        dct_pathways = unique_genes_only(dct_pathways)
    if short_names:
        name_mapping = get_17_pathways_short_names()
        dct_pathways = {name_mapping[name]:val for name,val in dct_pathways.iteritems()}
    return dct_pathways

def load_data(dataset='both', pathway=None, remove_prenatal=False, scaler=None):
    """This function is mostly for backward compatibility / syntactic sugar.
    """
    return GeneData.load(dataset).restrict_pathway(pathway).restrict_postnatal(remove_prenatal).scale_ages(scaler)

class SeveralGenesOneRegion(object):
    def __init__(self, expression, ages, gene_names, region_name, original_inds, age_scaler):
        n_ages, n_genes = expression.shape
        assert len(ages) == n_ages
        assert len(gene_names) == n_genes
        self.expression = expression
        self.ages = ages
        self.gene_names = gene_names
        self.region_name = region_name
        self.original_inds = original_inds
        self.age_scaler = age_scaler
        
    @property
    def num_genes(self): return len(self.gene_names)
        
    @property
    def gene_name(self):
        assert self.num_genes == 1
        return self.gene_names[0]
        
    @property
    def single_expression(self):
        assert self.num_genes == 1
        return self.expression[:,0]
        
    def get_single_gene_series(self, g):
        if isinstance(g, basestring):
            g = self._find_gene_index(g)
        return SeveralGenesOneRegion(
            expression = self.expression[:,[g]],
            ages = self.ages,
            gene_names = [self.gene_names[g]],
            region_name = self.region_name,
            original_inds = self.original_inds,
            age_scaler = self.age_scaler,
        )

    def _find_gene_index(self, name, allow_missing=False):
        match_positions = np.where(self.gene_names == name)[0]
        if len(match_positions) > 0:
            return match_positions[0]
        if allow_missing:
            return None
        raise Exception('Gene {} not found'.format(name))


class GeneData(object):
    def __init__(self, datasets, name):
        self.datasets = datasets
        self.name = name
        
    @staticmethod
    def load(dataset):
        name = dataset
        if dataset == 'both':
            dataset_names = ['kang2011', 'colantuoni2011']
        else:
            dataset_names = [dataset]
        datasets = [OneDataset.load(dataset_name) for dataset_name in dataset_names]
        return GeneData(datasets, name)

    @property
    def pathway(self):
        return get_unique(ds.pathway for ds in self.datasets)

    @property
    def age_range(self):
        ds_ranges = [ds.age_range for ds in self.datasets]
        min_age = min(ds_min for ds_min,ds_max in ds_ranges)
        max_age = max(ds_max for ds_min,ds_max in ds_ranges)
        return min_age, max_age
        
    @property
    def age_restriction(self):
        return get_unique(ds.age_restriction for ds in self.datasets)

    @property
    def age_scaler(self):
        return get_unique(ds.age_scaler for ds in self.datasets)

    @property
    def is_shuffled(self):
        return get_unique(ds.is_shuffled for ds in self.datasets)

    @property
    def gene_names(self):
        s = set()
        for ds in self.datasets:
            s.update(ds.gene_names)
        return sorted(s)
    
    @property
    def region_names(self):
        region_names = []
        for ds in self.datasets:
            region_names.extend(cfg.sorted_regions[ds.name])
        return region_names

    def region_to_dataset(self):
        res = {}
        for ds in self.datasets:
            for r in ds.region_names:
                res[r] = ds.name
        return res

    def restrict_pathway(self, pathway, ad_hoc_genes=None, allow_missing_genes=True):
        for ds in self.datasets:
            ds.restrict_pathway(pathway, ad_hoc_genes, allow_missing_genes)
        return self

    def restrict_ages(self, restriction_name, from_age=-10, to_age=1000):
        for ds in self.datasets:
            ds.restrict_ages(restriction_name, from_age, to_age)
        return self    

    def restrict_postnatal(self, b=True):
        for ds in self.datasets:
            ds.restrict_postnatal(b)
        return self    

    def restrict_regions(self, lst_regions):
        for ds in self.datasets:
            ds.restrict_regions(lst_regions)
        return self
        
    def scale_ages(self, scaler):
        for ds in self.datasets:
            ds.scale_ages(scaler)
        return self
        
    def shuffle(self):
        for ds in self.datasets:
            ds.shuffle()
        return self
    
    def get_one_series(self, iGene, iRegion, allow_missing=False):
        for ds in self.datasets:
            series = ds.get_one_series(iGene, iRegion, allow_missing=True)
            if series is not None:
                return series
        if allow_missing:
            return None
        raise Exception('{}@{} not found in the datasets'.format(iGene,iRegion))
    
    def get_several_series(self, genes, iRegion, allow_missing=False):
        for ds in self.datasets:
            series = ds.get_several_series(genes, iRegion, allow_missing=True)
            if series is not None:
                return series
        if allow_missing:
            return None
        raise Exception('{}@{} not found in the datasets'.format(genes,iRegion))
        
    def get_dataset_for_region(self, region_name):
        res = {ds.name for ds in self.datasets if region_name in ds.region_names}
        return get_unique(res)

class OneDataset(object):
    def __init__(self, expression, gene_names, region_names, genders, ages, name):
        n_ages, n_genes, n_regions = expression.shape
        assert len(ages) == n_ages
        assert len(genders) == n_ages
        assert len(gene_names) == n_genes
        assert len(region_names) == n_regions
        self.expression = expression
        self.gene_names = gene_names
        self.region_names = region_names
        self.genders = genders
        self.ages = ages        
        self.name = name
        self.pathway = 'all'
        self.age_restriction = None
        self.age_scaler = None
        self.is_shuffled = False
        
    @property
    def age_range(self):
        min_age = min(self.ages)
        max_age = max(self.ages)
        return min_age, max_age
        
    @staticmethod
    def load(dataset):
        datadir = project_dirs.data_dir()
        filename = '{}_allGenes.mat'.format(dataset)
        path = join(datadir,filename)
        if cfg.verbosity > 0:
            print 'Loading dataset {} from {}'.format(dataset,path)
        mat = loadmat(path)
        ages = np.array(mat['ages'].flat)
        gene_names = np.array(matlab_cell_array_to_list_of_strings(mat['gene_names']))
        region_names = np.array(matlab_cell_array_to_list_of_strings(mat['region_names']))
        genders = np.array(matlab_cell_array_to_list_of_strings(mat['genders']))        
        expression = mat['expression']
        if expression.ndim == 2: # extend shape to represent a single region name
            expression.shape = list(expression.shape)+[1]
            
        # average expression for duplicate genes
        dct = defaultdict(list) # gene_name -> list of indices where it appears
        for i,g in enumerate(gene_names):
            dct[g].append(i)
        new_gene_names = sorted(set(gene_names))
        new_expression = np.empty([len(ages),len(new_gene_names),len(region_names)])
        for i,g in enumerate(new_gene_names):
            idx = dct[g]
            new_expression[:,i,:] = expression[:,idx,:].mean(axis=1)
        gene_names = np.array(new_gene_names)
        expression = new_expression

        # make sure ages are sorted (for colantuoni there are 2 datapoints that aren't)
        inds = np.argsort(ages)
        ages = ages[inds]
        genders = genders[inds]
        expression = expression[inds,:,:]
        
        return OneDataset(
            expression = expression,
            gene_names = gene_names,
            region_names = region_names,
            genders = genders,
            ages = ages,
            name = dataset
        ).restrict_pathway('all')

    def restrict_pathway(self, pathway, ad_hoc_genes=None, allow_missing_genes=True):
        pathway, pathway_genes = _translate_pathway(pathway, ad_hoc_genes)
        if pathway_genes is None:
            # we still need to get rid of gene_names that are None
            inds = [i for i,g in enumerate(self.gene_names) if g is not None]
        else:
            inds = [self._find_gene_index(gene,allow_missing_genes) for gene in pathway_genes]
            missing = [g for g,i in zip(pathway_genes,inds) if i is None]
            if missing and cfg.verbosity > 0:
                print 'Dataset {} is missing {} genes from pathway {}: {}'.format(self.name, len(missing), pathway, missing)
            inds = [x for x in inds if x is not None]
        self.expression = self.expression[:,inds,:]
        self.gene_names = self.gene_names[inds]
        self.pathway = pathway
        return self

    def restrict_ages(self, restriction_name, from_age=-10, to_age=1000):
        if self.age_scaler is not None:
            from_age = self.age_scaler.scale(from_age)
            to_age = self.age_scaler.scale(to_age)
        valid = ((self.ages>=from_age) & (self.ages<=to_age))
        self.ages = self.ages[valid]
        self.genders = self.genders[valid]
        self.expression = self.expression[valid,:,:]
        self.age_restriction = restriction_name
        return self    

    def restrict_postnatal(self, b=True):
        if b:
            self.restrict_ages('postnatal',from_age=0)
        return self    

    def restrict_regions(self, lst_regions):
        inds = [np.where(self.region_names == region)[0][0] for region in lst_regions]
        self.expression = self.expression[:,:,inds]
        self.region_names = self.region_names[inds]
        return self
        
    def scale_ages(self, scaler):
        if scaler is None: # support NOP
            return self
        assert self.age_scaler is None, 'More than one scaling is not supported'
        self.ages = scaler.scale(self.ages)
        self.age_scaler = scaler
        return self
    
    def shuffle(self):
        nPoints, nGenes, nRegions = self.expression.shape
        for ig,g in enumerate(self.gene_names):
            for ir,r in enumerate(self.region_names):
                seed = abs(hash(g) ^ hash(r))
                rng = np.random.RandomState(seed)
                self.expression[:,ig,ir] = rng.permutation(self.expression[:,ig,ir])
        self.is_shuffled = True

    def get_one_series(self, iGene, iRegion, allow_missing=False):
        return self.get_several_series([iGene],iRegion,allow_missing)
        
    def get_several_series(self, genes, iRegion, allow_missing=False):
        if isinstance(iRegion, basestring):
            iRegion = self._find_region_index(iRegion, allow_missing=allow_missing)
        if iRegion is None:
            return None
        genes = [self._find_gene_index(g,allow_missing=allow_missing) if isinstance(g, basestring) else g for g in genes]
        genes = [g for g in genes if g is not None] # remove genes we don't have data for
        if not genes:
            return None
        expression = self.expression[:,genes,iRegion]
        ages = self.ages
        valid = ~np.all(np.isnan(expression),axis=1) # remove subjects where we don't have data for any gene
        ages, expression = ages[valid], expression[valid,:]
        return SeveralGenesOneRegion(
            expression = expression,
            ages = ages,
            gene_names = self.gene_names[genes],
            region_name = self.region_names[iRegion],
            original_inds = valid.nonzero(),
            age_scaler = self.age_scaler,
        )
        
    #####################################################################
    # Private helper methods
    #####################################################################        
        
    def _find_gene_index(self, name, allow_missing=False):
        match_positions = np.where(self.gene_names == name)[0]
        if len(match_positions) > 0:
            return match_positions[0]
        if allow_missing:
            return None
        raise Exception('Gene {} not found'.format(name))

    def _find_region_index(self, name, allow_missing=False):
        match_positions = np.where(self.region_names == name)[0]
        if len(match_positions) > 0:
            return match_positions[0]
        if allow_missing:
            return None
        raise Exception('Region {} not found'.format(name))        


####################################################
# Helpers
####################################################

def _translate_pathway(pathway, ad_hoc_genes):
    if pathway is None or pathway == 'all':
        return 'all',None
    if ad_hoc_genes is not None:
        _, pathway_genes = _translate_gene_list(ad_hoc_genes)
        return pathway, pathway_genes
    if pathway in cfg.pathways:
        _, pathway_genes = _translate_gene_list(cfg.pathways[pathway])
        return pathway, pathway_genes
        
    pathway_name, pathway_genes = _translate_gene_list(pathway)
    if pathway_genes is not None:
        assert pathway_name not in cfg.pathways, 'Changing the meaning (gene list) of a known pathway is not allowed. Pathway: {}'.format(pathway_name)
        return pathway_name, pathway_genes
        
    raise Exception('Unknown pathway: {}'.format(pathway))

def _translate_gene_list(gene_list):
    """gene_list can already be a sequence of strings or it could be a path to
       a file that contains a list of strings. 
       The path can be relative to the data directory or an absolute path.
       The file can be any format supported by read_strings_from_file()
    """
    # if it's not something that could be a filename assume it's a sequence of strings
    if not isinstance(gene_list, basestring):
        return None, gene_list 

    # try to find a matching filename and read the data from it
    path = gene_list
    pathway_name = splitext(basename(gene_list))[0] # use the file's basename (without extension) as the pathway name
    for path in [gene_list, join(project_dirs.data_dir(),gene_list)]:
        if isfile(path):
            if cfg.verbosity > 0:
                print 'Reading gene list from {}'.format(path)
            gene_list = read_strings_from_file(path)
            return pathway_name, gene_list

    # not found
    return None,None 
