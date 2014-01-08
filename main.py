from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

data = load_data()

def plot_gene(iGene):
    gene_name = data.gene_names[iGene]
    fig = figure()
    for iRegion in range(len(data.region_names)):
        region_name = data.region_names[iRegion]
        expr = data.expression[:,iGene,iRegion]
        ax = fig.add_subplot(4,4,iRegion)
        ax.plot(data.ages,expr,'ro')
        ax.plot(data.ages,expr,'b-')
        ax.set_title('Region {}'.format(region_name))
        ax.set_xlabel('Age [years]')
        ax.set_ylabel('Expression Level')
    fig.suptitle('Gene {}'.format(gene_name))