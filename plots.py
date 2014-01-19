import matplotlib.pyplot as plt
import config as cfg

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
    
def plot_one_series(series, fits=None, more_title=None):
    if fits is None:
        fits = {}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.ages,series.expression,'ro')
    for name,fit in fits.iteritems():
        ax.plot(series.ages, fit ,linewidth=2, label=name)
    ttl = 'Gene: {}, Region: {}'.format(series.gene_name, series.region_name)
    if more_title is not None:
        ttl = '{} ({})'.format(ttl,more_title)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_ylabel('Expression Level', fontsize=cfg.fontsize)
    ax.set_xlabel('Age [years]', fontsize=cfg.fontsize)
    ax.legend()
