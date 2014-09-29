import setup
from os import listdir
from os.path import join
from project_dirs import pathways_dir, pathway_lists_dir


def all_pathway_lists():
    return listdir(pathway_lists_dir())

def read_all_pathways(listname='all'):
    if listname == 'all':
        pathway_names = [f[:-4] for f in listdir(pathways_dir()) if f.endswith('.txt')]
    else:
        listfile = join(pathway_lists_dir(),listname)
        with open(listfile) as f:
            lines = f.readlines()
        pathway_names = [x.strip() for x in lines] # remove newlines
        pathway_names = [x for x in pathway_names if x] # rmeove empty strings
    return {pathway: read_pathway(pathway) for pathway in pathway_names}

def read_pathway(pathway):
    filename = join(pathways_dir(), pathway + '.txt')
    with open(filename) as f:
        lines = f.readlines()
    genes = [x.strip() for x in lines] # remove newlines
    return [x for x in genes if x] # rmeove empty strings
