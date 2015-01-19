import re
import numpy as np
from scipy.io import loadmat

####################################################################
####################################################################
# Read list of strings from file
####################################################################
####################################################################

def read_strings_from_file(filename):
    if filename.endswith('.mat'):
        return read_strings_from_mat_file(filename)
    else:
        return read_strings_from_text_file(filename)
        
def read_strings_from_text_file(filename):
    with open(filename) as f:
        text = f.read()
    lst = re.split('[\s,]+', text) # split by whitespace or commas
    return [x for x in lst if x] # remove any leading/trailing empty strings which re.split may return

def read_strings_from_mat_file(filename):
    mat = loadmat(filename)
    keys = [k for k in mat.keys() if not k.startswith('__')] # ignore matlab's special fields
    if not keys:
        raise AssertionError('mat file contains no fields')
    if len(keys) > 1:
        raise AssertionError('mat file contains more than one fields. found {} fields: {}'.format(len(keys),keys))
    val = mat[keys[0]]
    return matlab_cell_array_to_list_of_strings(val)

####################################################################
####################################################################
# Matlab conversions        
####################################################################
####################################################################
        
def matlab_cell_array_to_list_of_strings(cell_array):
    def convert_one(x):
        return str(x[0]) if x else None # some data files contain empty names (with some of these have different type)
    return [convert_one(x) for x in cell_array.flat]

def list_of_strings_to_matlab_cell_array(lst_string):
    def convert_one(x):
        return x if x is not None else ''
    lst = [convert_one(x) for x in lst_string]
    return np.array(lst, dtype='object') # tells savemat to produce a cell array
