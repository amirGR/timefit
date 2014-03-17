import os.path
from scipy.io import loadmat
import numpy as np
from collections import namedtuple
import project_dirs
import config as cfg
from utils import matlab_cell_array_to_list_of_strings

GeneDataBase = namedtuple('GeneData', [
    'expression', 'gene_names', 'region_names', 'genders', 'ages', 'pathway','dataset', 'postnatal_only',
])

class GeneData(GeneDataBase):
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
            region_name = self.region_names[iRegion],
        )
        
    def restrict_regions(self, lst_regions):
        inds = [np.where(self.region_names == region)[0][0] for region in lst_regions]
        return GeneData(
            expression = self.expression[:,:,inds],
            gene_names = self.gene_names,
            region_names = self.region_names[inds],
            genders = self.genders,
            ages = self.ages,
            pathway = self.pathway,
            dataset = self.dataset,
            postnatal_only = self.postnatal_only,
        )

OneGeneRegion = namedtuple('OneGeneRegion', ['expression', 'ages', 'gene_name', 'region_name'])

def load_data(pathway='serotonin',dataset='kang2011', remove_prenatal=True):
    datadir = project_dirs.data_dir()
    filename = '{}_allGenes.mat'.format(dataset)
    path = os.path.join(datadir,filename)
    mat = loadmat(path)
    ages = np.array(mat['ages'].flat)
    all_gene_names = matlab_cell_array_to_list_of_strings(mat['gene_names'])
    all_expression_levels = mat['expression']
    if all_expression_levels.ndim == 2: # extend shape to represent a single region name
        all_expression_levels.shape = list(all_expression_levels.shape)+[1] 
    if pathway == 'all':
        pathway_expression = all_expression_levels
        gene_names = all_gene_names    
    else:
        assert pathway in cfg.pathways, 'Unknown pathway: {}'.format(pathway)
        pathway_genes = cfg.pathways[pathway]
        inds = [np.where(all_gene_names == gene)[0][0] for gene in pathway_genes]
        pathway_expression = all_expression_levels[:,inds,:]
        gene_names = np.array(pathway_genes)
        
    if remove_prenatal:
        valid = (ages>0)
        ages = ages[valid]
        pathway_expression = pathway_expression[valid,:,:]
        
    data = GeneData(
        expression = pathway_expression,
        gene_names = gene_names,
        region_names = matlab_cell_array_to_list_of_strings(mat['region_names']),
        genders = matlab_cell_array_to_list_of_strings(mat['genders']),
        ages = ages,
        pathway = pathway,
        dataset = dataset,
        postnatal_only = remove_prenatal,
    )
    return data

def load_matlab_gene_set(pathway):
    print 'PATHWAY: {}'.format(pathway)
    datadir = project_dirs.data_dir()
    file_names = {'pathway17_seq_sim': 'gene_list_pathways_pairs_seqSim.mat' }
    full_path = os.path.join(datadir,file_names.get(pathway))
    mat = loadmat(full_path)
    pathway_gene_names = matlab_cell_array_to_list_of_strings(mat['geneSymbol_list'])
    return pathway_gene_names