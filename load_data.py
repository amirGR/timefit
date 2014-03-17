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
        
    @staticmethod
    def load(dataset):
        datadir = project_dirs.data_dir()
        filename = '{}_allGenes.mat'.format(dataset)
        path = os.path.join(datadir,filename)
        mat = loadmat(path)
        ages = np.array(mat['ages'].flat)
        gene_names = matlab_cell_array_to_list_of_strings(mat['gene_names'])
        expression = mat['expression']
        if expression.ndim == 2: # extend shape to represent a single region name
            expression.shape = list(expression.shape)+[1]             
        return GeneData(
            expression = expression,
            gene_names = gene_names,
            region_names = matlab_cell_array_to_list_of_strings(mat['region_names']),
            genders = matlab_cell_array_to_list_of_strings(mat['genders']),
            ages = ages,
            dataset = dataset
        )

    def restrict_pathway(self, pathway, ad_hoc_genes=None):
        if pathway == 'all':
            return self
        if pathway in cfg.pathways:
            assert ad_hoc_genes is None, 'Specifying ad_hoc_genes for a known pathway is not allowed. Pathway: {}'.format(pathway)
            pathway_genes = cfg.pathways[pathway]
        elif ad_hoc_genes is not None:
            pathway_genes = ad_hoc_genes
        else:
            raise Exception('Unknown pathway: {}'.format(pathway))
        inds = [np.where(self.gene_names == gene)[0][0] for gene in pathway_genes]
        self.expression = self.expression[:,inds,:]
        self.gene_names = np.array(pathway_genes)
        self.pathway = pathway
        return self
    
    def restrict_postnatal(self, b=True):
        if b:
            valid = (self.ages>0)
            self.ages = self.ages[valid]
            self.expression = self.expression[valid,:,:]
            self.postnatal_only = True
        return self    

    def restrict_regions(self, lst_regions):
        inds = [np.where(self.region_names == region)[0][0] for region in lst_regions]
        self.expression = self.expression[:,:,inds]
        self.region_names = self.region_names[inds]
        return self
    
    def get_one_series(self, iGene, iRegion):
        if isinstance(iGene, basestring):
            iGene = np.where(self.gene_names == iGene)[0][0]
        if isinstance(iRegion, basestring):
            iRegion = np.where(self.region_names == iRegion)[0][0]
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
        
def load_data(pathway='serotonin',dataset='kang2011', remove_prenatal=True):
    """This function is mostly for backward compatibility / syntactic sugar.
    """
    return GeneData.load(dataset).restrict_pathway(pathway).restrict_postnatal(remove_prenatal)

def load_matlab_gene_set(pathway):
    print 'PATHWAY: {}'.format(pathway)
    datadir = project_dirs.data_dir()
    file_names = {'pathway17_seq_sim': 'gene_list_pathways_pairs_seqSim.mat' }
    full_path = os.path.join(datadir,file_names.get(pathway))
    mat = loadmat(full_path)
    pathway_gene_names = matlab_cell_array_to_list_of_strings(mat['geneSymbol_list'])
    return pathway_gene_names