import setup
import pickle
from os.path import join
from os import listdir
import numpy as np
from project_dirs import cache_dir, pathways_dir

def read_gene_names(pathway):
    filename = join(pathways_dir(), pathway + '.txt')
    with open(filename) as f:
        lines = f.readlines()
    genes = [x.strip() for x in lines] # remove newlines
    return [x for x in genes if g] # rmeove empty strings
pathway_names = [f[:-4] for f in listdir(pathways_dir()) if f.endswith('.txt')]
pathways = {name: read_gene_names(name) for name in pathway_names}

filename = join(cache_dir(), 'both', 'fits-log-all-sigmoid-theta-sigmoid_wide-sigma-normal-dprime.pkl')
with open(filename) as f:
    data = pickle.load(f)
g2i = {g:i for i,g in enumerate(data['genes'])}
r2i = {r:i for i,r in enumerate(data['regions'])}
scores = data['d_mu'] / data['std']

def sample_scores(r1, r2, sample_size, n_samples):
    ir1 = r2i[r1]
    ir2 = r2i[r2]
    r12_scores = scores[:,ir1,ir2]
    x = np.empty(n_samples)
    inds = np.random.random_integers(0, )