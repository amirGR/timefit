import setup
import pickle
from collections import defaultdict
from os.path import join
import numpy as np
from scipy.io import loadmat
import config as cfg
import project_dirs

cfg.verbosity = 1

def get_entrez_pathways():
    pathway_numbers = {
        4020 : 'Calcium signaling', # http://www.genome.jp/kegg-bin/show_pathway?map=hsa04020
        4080 : 'Neuroactive ligand-receptor interaction',
        4724 : 'Glutamatergic synapse', 
        4725 : 'Cholinergic synapse', 
        4726 : 'Serotonergic synapse', 
        4727 : 'GABAergic synapse', 
        4728 : 'Dopaminergic synapse', 
        4730 : 'Long-term depression', 
        5010 : "Alzheimer's disease", 
        5012 : "Parkinson's disease", 
        5014 : 'Amyotrophic lateral sclerosis (ALS)', 
        5016 : "Huntington's disease", 
        5030 : 'Cocaine addiction', 
        5031 : 'Amphetamine addiction', 
        5032 : 'Morphine addiction', 
        5033 : 'Nicotine addiction',
        5034 : 'Alcoholism', 
    }
    
    datadir = project_dirs.data_dir()
    filename = 'gene_pathways.mat'
    path = join(datadir,filename)
    mat = loadmat(path)['path_genes_mat']
    
    row_indices, column_indices = np.nonzero(mat)
    pathways = row_indices + 1 # the matrix was created in matlab which is 1 based. python is 0 based
    genes = column_indices + 1
    pathway_genes = zip(pathways,genes)
    
    dct_pathways = defaultdict(set)
    for p,g in pathway_genes:
        if p in pathway_numbers:
            pname = pathway_numbers[p]
            dct_pathways[pname].add(g)
    return dct_pathways
    
def load_entrez_to_symbol_mapping():
    conversion_filename = join(project_dirs.data_dir(),'human_entrez_conversion.txt')
    with open(conversion_filename) as f:
        lines = f.readlines()
    dct_mapping = {}
    for line in lines[1:]: # skip header
        fields = line.split()
        if len(fields) < 3:
            continue
        _,entrez,symbol = fields
        if entrez == 'NA':
            continue
        dct_mapping[int(entrez)] = symbol
    return dct_mapping
    
dct_pathways_entrez = get_entrez_pathways()
dct_mapping = load_entrez_to_symbol_mapping()

dct_pathways = {}
for pathway,entrez_genes in dct_pathways_entrez.iteritems():
    symbols = set(dct_mapping.get(eg) for eg in entrez_genes)
    symbols = set(x for x in symbols if x is not None)
    dct_pathways[pathway] = symbols

outfile = join(project_dirs.data_dir(),'17pathways-breakdown.pkl')
with open(outfile,'w') as f:
    pickle.dump(dct_pathways,f)
    
all_genes = set(g for pwy in dct_pathways.itervalues() for g in pwy)
outfile = join(project_dirs.data_dir(),'17pathways-full.txt')
with open(outfile,'w') as f:
    for g in sorted(all_genes):
        f.write(g + '\n')

