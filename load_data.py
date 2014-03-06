import os.path
from scipy.io import loadmat
import numpy as np
from collections import namedtuple
import project_dirs
import config as cfg

GeneDataBase = namedtuple('GeneData', [
    'expression', 'gene_names', 'region_names', 'genders', 'ages', 'pathway','dataset',
])

class GeneData(GeneDataBase):
    def get_one_series(self, iGene, iRegion, remove_prenatal=True):
        if isinstance(iGene, basestring):
            iGene = np.where(self.gene_names == iGene)[0][0]
        if isinstance(iRegion, basestring):
            iRegion = np.where(self.region_names == iRegion)[0][0]
        expression = self.expression[:,iGene,iRegion]
        ages = self.ages
        valid = ~np.isnan(expression)
        if remove_prenatal:
            valid = valid & (ages>0)
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
        )

OneGeneRegion = namedtuple('OneGeneRegion', ['expression', 'ages', 'gene_name', 'region_name'])

def load_data(pathway='serotonin',dataset='kang2011'):
    datadir = project_dirs.data_dir()
    filename = '{}_allGenes.mat'.format(dataset)
    path = os.path.join(datadir,filename)
    mat = loadmat(path)
    all_gene_names = convert_matlab_string_cell(mat['gene_names'])
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
    data = GeneData(
        expression = pathway_expression,
        gene_names = gene_names,
        region_names = convert_matlab_string_cell(mat['region_names']),
        genders = convert_matlab_string_cell(mat['genders']),
        ages = np.array(mat['ages'].flat),
        pathway = pathway,
        dataset = dataset,
    )
    return data

def convert_matlab_string_cell(cell_array):
    def convert_one(x):
        return x[0] if x else None # some data files contain empty names (with some of these have different type)
    return np.array([convert_one(x) for x in cell_array.flat])

def load_matlab_gene_set(pathway):
    print 'PATHWAY: {}'.format(pathway)
    datadir = project_dirs.data_dir()
    file_names = {'pathway17_seq_sim': 'gene_list_pathways_pairs_seqSim.mat' }
    full_path = os.path.join(datadir,file_names.get(pathway))
    mat = loadmat(full_path)
    pathway_gene_names = convert_matlab_string_cell(mat['geneSymbol_list'])
    return pathway_gene_names