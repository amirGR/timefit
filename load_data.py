from scipy.io import loadmat
import numpy as np
from collections import namedtuple
import project_dirs

GeneDataBase = namedtuple('GeneData', [
    'expression', 'gene_names', 'region_names', 'genders', 'ages',
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
        return OneGeneRegion(
            expression = expression[valid],
            ages = ages[valid],
            gene_name = self.gene_names[iGene],
            region_name = self.region_names[iRegion],
        )

OneGeneRegion = namedtuple('OneGeneRegion', ['expression', 'ages', 'gene_name', 'region_name'])

def load_data(serotonin_only=True):
    datadir = project_dirs.data_dir()
    if serotonin_only:
        filename = 'kang2011_serotonin.mat'
    else:
        filename = 'kang2011_allGenes.mat'
    path = r'{}\{}'.format(datadir,filename)
    mat = loadmat(path)
    data = GeneData(
        expression = mat['expression'],
        gene_names = convert_matlab_string_cell(mat['gene_names']),
        region_names = convert_matlab_string_cell(mat['region_names']),
        genders = convert_matlab_string_cell(mat['genders']),
        ages = mat['ages'][0,:],
    )
    return data

def convert_matlab_string_cell(cell_array):
    return np.array([x[0] for x in cell_array.flat])
