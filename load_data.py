from scipy.io import loadmat
import numpy as np
from collections import namedtuple
    
GeneData = namedtuple('GeneData', [
    'expression', 'gene_names', 'region_names', 'genders', 'ages',
])
    
def convert_matlab_string_cell(cell_array):
    return np.array([x[0] for x in cell_array.flat])

def load_data(serotonin_only=True):
    datadir = r'C:\data\HTR\data'
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
