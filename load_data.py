from scipy.io import loadmat
import numpy as np
from collections import namedtuple
import project_dirs
import config as cfg

GeneDataBase = namedtuple('GeneData', [
    'expression', 'gene_names', 'region_names', 'genders', 'ages', 'pathway'
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

OneGeneRegion = namedtuple('OneGeneRegion', ['expression', 'ages', 'gene_name', 'region_name'])

def load_data(pathway='serotonin'):
    datadir = project_dirs.data_dir()
    filename = 'kang2011_allGenes.mat'
    path = r'{}\{}'.format(datadir,filename)
    mat = loadmat(path)
    all_gene_names = convert_matlab_string_cell(mat['gene_names'])
    all_expression_levels = expression = mat['expression']
    assert pathway in cfg.pathways, 'Unknown pathway: {}'.format(pathway)
    pathway_genes = cfg.pathways[pathway]
    inds = [np.where(all_gene_names == gene)[0][0] for gene in pathway_genes]
    pathway_expression = all_expression_levels[:,inds,:]
    data = GeneData(
        expression = pathway_expression,
        gene_names = np.array(pathway_genes),
        region_names = convert_matlab_string_cell(mat['region_names']),
        genders = convert_matlab_string_cell(mat['genders']),
        ages = mat['ages'][0,:],
        pathway = pathway,
    )
    return data

def convert_matlab_string_cell(cell_array):
    return np.array([x[0] for x in cell_array.flat])
