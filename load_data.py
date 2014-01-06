from scipy.io import loadmat
import numpy as np

class Data(object): pass
    
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
    data = Data()
    data.expression = mat['expression']
    data.gene_names = convert_matlab_string_cell(mat['gene_names'])
    data.region_names = convert_matlab_string_cell(mat['region_names'])
    data.genders = convert_matlab_string_cell(mat['genders'])
    data.ages = mat['ages'][0,:]
    return data
