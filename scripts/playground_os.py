import setup
from load_data import load_data
from load_data import load_matlab_gene_set
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from plots import *
from all_fits import *
import config as cfg
from fitter import Fitter
from shapes.sigmoid import Sigmoid

# load some data
pathway='pathway17_seq_sim'
curr_gene_set = load_matlab_gene_set(pathway)
curr_gene_list = curr_gene_set.tolist()
curr_gene_set_str = [str(s) for s in curr_gene_list]
cfg.pathways[pathway] = curr_gene_set_str;
len(cfg.pathways.get('pathway17_seq_sim'))

#[name for name in curr_gene_set_str if name not in gene_names]
#pathway='serotonin'
dataset = 'kang2011'
data = load_data(pathway,dataset)   
series = data.get_one_series('HTR2B','VFC')
x = series.ages
y = series.expression
fitter = Fitter(Sigmoid())
fits = get_all_fits(data,fitter)


filename = join(project_dirs.cache_dir(), dataset, 'fits-{}-{}.pkl'.format(pathway, fitter.cache_name()))  
    # load the cache we have so far
filename = "C:\\work\\HTR\\cache\\kang2011\\fits-serotonin-sigmoid-t1-s0.pkl"
    try:
        with open(filename) as f:
            fits = pickle.load(f)
    except:
        fits = {}
        
