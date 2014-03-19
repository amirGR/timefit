import os.path
import numpy as np
from scipy.io import loadmat
import project_dirs
import config as cfg
from utils import matlab_cell_array_to_list_of_strings

class OneGeneRegion(object):
    def __init__(self, expression, ages, gene_name, region_name):
        self.expression = expression
        self.ages = ages
        self.gene_name = gene_name
        self.region_name = region_name

class GeneData(object):
    def __init__(self, expression, gene_names, region_names, genders, ages, dataset):
        self.expression = expression
        self.gene_names = gene_names
        self.region_names = region_names
        self.genders = genders
        self.ages = ages        
        self.dataset = dataset
        self.pathway = 'all'
        self.postnatal_only = False
        self.age_scaler = None
        
    @staticmethod
    def load(dataset):
        datadir = project_dirs.data_dir()
        filename = '{}_allGenes.mat'.format(dataset)
        path = os.path.join(datadir,filename)
        mat = loadmat(path)
        ages = np.array(mat['ages'].flat)
        gene_names = np.array(matlab_cell_array_to_list_of_strings(mat['gene_names']))
        expression = mat['expression']
        if expression.ndim == 2: # extend shape to represent a single region name
            expression.shape = list(expression.shape)+[1]             
        return GeneData(
            expression = expression,
            gene_names = gene_names,
            region_names = np.array(matlab_cell_array_to_list_of_strings(mat['region_names'])),
            genders = np.array(matlab_cell_array_to_list_of_strings(mat['genders'])),
            ages = ages,
            dataset = dataset
        )

    def restrict_pathway(self, pathway, ad_hoc_genes=None, allow_missing_genes=True):
        if pathway == 'all':
            return self
        if pathway in cfg.pathways:
            assert ad_hoc_genes is None, 'Specifying ad_hoc_genes for a known pathway is not allowed. Pathway: {}'.format(pathway)
            pathway_genes = cfg.pathways[pathway]
        elif ad_hoc_genes is not None:
            pathway_genes = ad_hoc_genes
        else:
            raise Exception('Unknown pathway: {}'.format(pathway))
        inds = [self._find_gene_index(gene,allow_missing_genes) for gene in pathway_genes]
        missing = [g for g,i in zip(pathway_genes,inds) if i is None]
        if missing and cfg.verbosity > 0:
            print 'Dataset {} is missing {} genes from pathway {}: {}'.format(self.dataset, len(missing), pathway, missing)
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
        assert self.age_scaler is None, 'More than one scaling is not supported'
        self.ages = scaler.scale(self.ages)
        self.age_scaler = scaler
        return self
    
    def get_one_series(self, iGene, iRegion):
        if isinstance(iGene, basestring):
            iGene = self._find_gene_index(iGene)
        if isinstance(iRegion, basestring):
            iRegion = self._find_region_index(iRegion)
        expression = self.expression[:,iGene,iRegion]
        ages = self.ages
        valid = ~np.isnan(expression)
        ages, expression = ages[valid], expression[valid]
        # sort by ascending ages (some code later can rely on ages being sorted)
        inds = ages.argsort()
        ages, expression = ages[inds], expression[inds]
        return OneGeneRegion(
            expression = expression,
            ages = ages,
            gene_name = self.gene_names[iGene],
            region_name = self.region_names[iRegion]
        )
        
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
        
def load_data(pathway='serotonin',dataset='kang2011', remove_prenatal=True):
    """This function is mostly for backward compatibility / syntactic sugar.
    """
    return GeneData.load(dataset).restrict_pathway(pathway).restrict_postnatal(remove_prenatal)
