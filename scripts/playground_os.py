import setup
from load_data import load_data
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
from plots import *
from all_fits import *
import config as cfg
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from utils import matlab_cell_array_to_list_of_strings

def load_matlab_gene_set(pathway):
    print 'PATHWAY: {}'.format(pathway)
    datadir = project_dirs.data_dir()
    file_names = {'pathway17_seq_sim': 'gene_list_pathways_pairs_seqSim.mat' }
    full_path = os.path.join(datadir,file_names.get(pathway))
    mat = loadmat(full_path)
    pathway_gene_names = matlab_cell_array_to_list_of_strings(mat['geneSymbol_list'])
    return pathway_gene_names
    
# load some data
pathway='pathway17_seq_sim'
curr_gene_set = load_matlab_gene_set(pathway)
cfg.pathways[pathway] = curr_gene_set_str;
len(cfg.pathways.get('pathway17_seq_sim'))

#[name for name in curr_gene_set_str if name not in gene_names]
#pathway='serotonin'
dataset = 'kang2011'
remove_prenatal=False

data = load_data(pathway,dataset,remove_prenatal)   
series = data.get_one_series('HTR2B','VFC')
x = series.ages
y = series.expression
fitter = Fitter(Sigmoid())
fits = get_all_fits(data,fitter)
save_as_mat_file(fits, 'try1.mat')
save_fits_and_create_html(data, fitter, project_dirs.results_dir())
