import matplotlib.pyplot as plt
from sigmoid_fit import sigmoid

def plot_gene(data, iGene):
    fig = plt.figure()
    for iRegion in range(len(data.region_names)):
        series = data.get_one_series(iGene,iRegion)
        ax = fig.add_subplot(4,4,iRegion+1)
        ax.plot(series.ages,series.expression,'ro')
        ax.plot(series.ages,series.expression,'b-')
        ax.set_title('Region {}'.format(series.region_name))
        if iRegion % 4 == 0:
            ax.set_ylabel('Expression Level')
        if iRegion / 4 >= 3:
            ax.set_xlabel('Age [years]')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(series.gene_name))
    
def plot_one_fit(series,fit=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.ages,series.expression,'ro')
    ax.plot(series.ages,series.expression,'b-')
    if fit is not None:
        ax.plot(series.ages,fit,'g-')
    ax.set_title('Region {}'.format(series.gene_name))
    ax.set_ylabel('Expression Level')
    ax.set_xlabel('Age [years]')
 