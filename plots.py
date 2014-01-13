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
    
def plot_one_series(series, fits=None):
    if fits is None:
        fits = {}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.ages,series.expression,'ro')
    for name,fit in fits.iteritems():
        ax.plot(series.ages, fit ,linewidth=2, label=name)
    ax.set_title('Gene: {}, Region: {}'.format(series.gene_name, series.region_name))
    ax.set_ylabel('Expression Level')
    ax.set_xlabel('Age [years]')
    ax.legend()

def plot_L_scores(Ls, scores, logscale=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Ls, scores, linewidth=2)
    ax.plot(Ls, scores, 'ro')
    if logscale:
        ax.set_xscale('log')
    ax.set_title(r'{} vs. $\lambda$'.format(cfg.score_type), fontsize=cfg.fontsize)
    ax.set_xlabel(r'$\lambda$', fontsize=cfg.fontsize)
    ax.set_ylabel(cfg.score_type, fontsize=cfg.fontsize)

    idx = scores.argmax()
    best_L = Ls[idx]
    best_score = scores[idx]
    ax.plot(best_L,best_score,'gd')
    ax.annotate(
        r'best $\lambda$', 
        xy = (best_L, best_score), 
        xytext = (best_L,best_score*0.6), 
        arrowprops = dict(
            edgecolor = 'black',
            facecolor = 'green',
            shrink = 0.1,
        ),
        horizontalalignment = 'center',
        fontsize = cfg.fontsize
    )
    
