from load_data import load_data
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

data = load_data()

def plot_gene(iGene):
    fig = plt.figure()
    for iRegion in range(len(data.region_names)):
        series = data.get_one_series(iGene,iRegion)
        ax = fig.add_subplot(4,4,iRegion+1)
        ax.plot(series.ages,series.expression,'ro')
        ax.plot(series.ages,series.expression,'b-')
        ax.set_title('Region {}'.format(series.gene_name))
        if iRegion % 4 == 0:
            ax.set_ylabel('Expression Level')
        if iRegion / 4 >= 3:
            ax.set_xlabel('Age [years]')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(series.gene_name))
