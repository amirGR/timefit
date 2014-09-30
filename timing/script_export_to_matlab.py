import setup
from os.path import join, dirname
import numpy as np
from scipy.io import savemat
from single_region import SingleRegion
from region_pairs import RegionPairTiming
from utils.misc import load_pickle, ensure_dir
from utils.formats import list_of_strings_to_matlab_cell_array
from project_dirs import results_dir
import scalers
import pathway_lists

def save_matfile(mdict, filename):
    ensure_dir(dirname(filename))
    print 'Saving to {}'.format(filename)
    savemat(filename, mdict, oned_as='column')
    
def export_cube():
    cube = load_pickle(RegionPairTiming.cube_filename)
    README = """\
d_mu:
mu(r2)-mu(r1) for every gene and region pair. 
Dimensions: <n-genes> X <n-regions> X <n-regions>

combined_std: 
The combined standard deviation of the two change distributions.
std = sqrt(0.5*(std1^2 + std2^2))
Dimensions: <n-genes> X <n-regions> X <n-regions>

score:
The d' for the two change distributions. Equal to d_mu ./ combined_std.
Dimensions: <n-genes> X <n-regions> X <n-regions>

genes: 
Gene names for the genes represented in other arrays

regions: 
Region names for the regions represented in other arrays

age_scaler: 
The scaling used for ages (i.e. 'log' means x' = log(x + 38/52))
"""
    mdict = dict(
        README_CUBE = README,
        genes = list_of_strings_to_matlab_cell_array(cube.genes),
        regions = list_of_strings_to_matlab_cell_array(cube.regions),
        age_scaler = scalers.unify(cube.age_scaler).cache_name(),
        d_mu = cube.d_mu,
        combined_std = cube.std,
        scores = cube.d_mu / cube.std,
    )
    save_matfile(mdict, join(results_dir(), 'export', 'cube.mat'))

def export_singles():
    change_dist = load_pickle(SingleRegion.change_dist_filename)
    README = """\
mu:
The mean age of the change distribution for given gene and region.
Dimensions: <n-genes> X <n-regions>

std:
The standard deviation of the change distribution for given gene and region.
Dimensions: <n-genes> X <n-regions>

genes: 
Gene names for the genes represented in other arrays

weights:
The change distributions for each gene and region.
Dimensions: <n-genes> X <n-regions> X <n-bins>

bin_centers:
The ages for the center of each bin used in calculating the histogram in "weights".
Dimensions: <n-bins> X 1

bin_edges:
The edges of the bins used in calculating the change histogram.
(centers can be calculated from the bin_edges, but it's convenient to have it pre-calculated)
Dimensions: <n-bins + 1> X 1

regions: 
Region names for the regions represented in other arrays

age_scaler: 
The scaling used for ages (i.e. 'log' means x' = log(x + 38/52))
"""
    mdict = dict(
        README_CHANGE_DISTRIBUTIONS = README,
        genes = list_of_strings_to_matlab_cell_array(change_dist.genes),
        regions = list_of_strings_to_matlab_cell_array(change_dist.regions),
        age_scaler = scalers.unify(change_dist.age_scaler).cache_name(),
        mu = change_dist.mu,
        std = change_dist.std,
        bin_edges = change_dist.bin_edges,
        bin_centers = change_dist.bin_centers,
        weights = change_dist.weights,
    )
    save_matfile(mdict, join(results_dir(), 'export', 'change-distributions.mat'))

def export_pathways():
    change_dist = load_pickle(SingleRegion.change_dist_filename)
    matlab_g2i = {g:(i+1) for i,g in enumerate(change_dist.genes)} # NOTE that matlab is one based
    
    pathways = pathway_lists.read_all_pathways()
    pathway_names = pathways.keys() # make sure the order stays fixed
    pathway_genes_names = np.array([list_of_strings_to_matlab_cell_array(pathways[p]) for p in pathway_names], dtype=object)
    pathway_genes_idx = np.array([np.array([matlab_g2i[g] for g in pathways[p]]) for p in pathway_names], dtype=object)

    matlab_p2i = {p:(i+1) for i,p in enumerate(pathway_names)} # NOTE matlab indexing is one based
    list_names = pathway_lists.all_pathway_lists()
    list_pathway_names = np.empty(len(list_names), dtype=object)
    list_pathway_idx = np.empty(len(list_names), dtype=object)
    for i,listname in enumerate(list_names):
        pathways_in_list = pathway_lists.list_to_pathway_names(listname)
        list_pathway_names[i] = list_of_strings_to_matlab_cell_array(pathways_in_list)
        list_pathway_idx[i] = [matlab_p2i[p] for p in pathways_in_list]
    README = """\
pathway_names:
Cell array of all pathway names. The name in cell number k is the name of the
pathway at position k in "pathway_genes_names" and "pathway_genes_idx".

pathway_genes_names:
Cell array (size <n-pathways>). Each cell contains a cell array of strings which 
are the gene symbols of the genes in that pathway.

pathway_genes_idx:
Same as pathway_genes_names, but each cell in the outer cell array is now an 
array of gene indices corresponding to the gene positions in cube.mat and change-distributions.mat.
Hopefully this should be easier to use in matlab.

list_names:
Names of pathway lists prepared by Noa

list_pathway_names:
Call array. One item per list. Each item is a cell array of strings which are 
the names of the pathways belonging to that list.

list_pathway_idx:
Same as list_pathway_names, but instead of listing the pathways by name, they 
are given as indices into the previous pathway_xxx structures.
"""
    mdict = dict(
        README_PATHWAYS = README,
        pathway_names = list_of_strings_to_matlab_cell_array(pathway_names),
        pathway_genes_names = pathway_genes_names,
        pathway_genes_idx = pathway_genes_idx,
        list_names = list_of_strings_to_matlab_cell_array(list_names),
        list_pathway_names = list_pathway_names,
        list_pathway_idx = list_pathway_idx,
    )
    save_matfile(mdict, join(results_dir(), 'export', 'pathways.mat'))

##############################################################
# main
##############################################################
if __name__ == '__main__':
    export_cube()
    export_singles()
    export_pathways()