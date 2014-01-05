from scipy.io import loadmat
import numpy as np

class Data(object): pass
    
def cell_array_to_list_of_string(cell_array):
    return np.array([x[0] for x in cell_array.flat])

def load_data():
    datadir = r'C:\data\HTR\data'
    filename = 'kang2011_allGenes.mat'
    path = r'{}\{}'.format(datadir,filename)
    mat = loadmat(path)
    data = Data()
    data.expression = mat['expression']
    data.gene_names = cell_array_to_list_of_string(mat['gene_names'])
    data.region_names = cell_array_to_list_of_string(mat['region_names'])
    data.genders = cell_array_to_list_of_string(mat['genders'])
    data.ages = mat['ages'][0,:]
    return data
