from os.path import join, splitext, basename, isfile
import numpy as np
from scipy.io import loadmat
import project_dirs
import config as cfg
from utils.formats import matlab_cell_array_to_list_of_strings, read_strings_from_file
from utils.misc import get_unique

def load_17_pathways_breakdown():
    import pickle
    filename = join(project_dirs.data_dir(),'17pathways-breakdown.pkl')
    with open(filename) as f:
        dct_pathways = pickle.load(f)
    return dct_pathways

def load_data(dataset='both', pathway=None, remove_prenatal=False, scaler=None):
    """This function is mostly for backward compatibility / syntactic sugar.
    """
    return GeneData.load(dataset).restrict_pathway(pathway).restrict_postnatal(remove_prenatal).scale_ages(scaler)

class OneGeneRegion(object):
    def __init__(self, expression, ages, gene_name, region_name, original_inds, age_scaler):
        self.expression = expression
        self.ages = ages
        self.gene_name = gene_name
        self.region_name = region_name
        self.original_inds = original_inds
        self.age_scaler = age_scaler

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
    def postnatal_only(self):
        return get_unique(ds.postnatal_only for ds in self.datasets)

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

class OneDataset(object):
    def __init__(self, expression, gene_names, region_names, genders, ages, name):
        self.expression = expression
        self.gene_names = gene_names
        self.region_names = region_names
        self.genders = genders
        self.ages = ages        
        self.name = name
        self.pathway = 'all'
        self.postnatal_only = False
        self.age_scaler = None
        self.is_shuffled = False
        
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
        expression = mat['expression']
        if expression.ndim == 2: # extend shape to represent a single region name
            expression.shape = list(expression.shape)+[1]             
        return OneDataset(
            expression = expression,
            gene_names = gene_names,
            region_names = np.array(matlab_cell_array_to_list_of_strings(mat['region_names'])),
            genders = np.array(matlab_cell_array_to_list_of_strings(mat['genders'])),
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

    def restrict_postnatal(self, b=True):
        if b:
            assert self.age_scaler is None, 'restrict_postnatal cannot be called after scaling'
            valid = (self.ages>0)
            self.ages = self.ages[valid]
            self.genders = self.genders[valid]
            self.expression = self.expression[valid,:,:]
            self.postnatal_only = True
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
        if isinstance(iGene, basestring):
            iGene = self._find_gene_index(iGene, allow_missing=allow_missing)
        if isinstance(iRegion, basestring):
            iRegion = self._find_region_index(iRegion, allow_missing=allow_missing)
        if iGene is None or iRegion is None:
            return None
        expression = self.expression[:,iGene,iRegion]
        ages = self.ages
        valid = ~np.isnan(expression)
        ages, expression = ages[valid], expression[valid]
        return OneGeneRegion(
            expression = expression,
            ages = ages,
            gene_name = self.gene_names[iGene],
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
    if pathway in cfg.pathways:
        assert ad_hoc_genes is None, 'Specifying ad_hoc_genes for a known pathway is not allowed. Pathway: {}'.format(pathway)
        _, pathway_genes = _translate_gene_list(cfg.pathways[pathway])
        return pathway, pathway_genes
    if ad_hoc_genes is not None:
        _, pathway_genes = _translate_gene_list(ad_hoc_genes)
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
        
