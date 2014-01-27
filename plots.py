import matplotlib.pyplot as plt
import config as cfg
from all_fits import get_all_fits

def save_figure(fig, filename, b_close=False):
    fig.set_size_inches(cfg.default_figure_size_x, cfg.default_figure_size_y)
    fig.savefig(filename, facecolor=cfg.default_figure_facecolor, dpi=cfg.default_figure_dpi)
    if b_close:
        plt.close(fig)
 
def plot_gene(data, iGene, fits=None):
    fig = plt.figure()
    for iRegion in range(len(data.region_names)):
        series = data.get_one_series(iGene,iRegion)
        ax = fig.add_subplot(4,4,iRegion+1)
        ax.plot(series.ages,series.expression,'ro')
        if fits is not None:
            fit = fits[(series.gene_name,series.region_name)]
            ax.plot(series.ages, fit.fit_predictions, 'b-')
        else:
            ax.plot(series.ages,series.expression,'b-')
        ax.set_title('Region {}'.format(series.region_name))
        if iRegion % 4 == 0:
            ax.set_ylabel('Expression Level')
        if iRegion / 4 >= 3:
            ax.set_xlabel('Age [years]')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(series.gene_name))
    return fig
    
def plot_one_series(series, fits=None):
    g,r = series.gene_name, series.region_name
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.ages,series.expression,'ro')
    if fits is not None:
        fit = fits[(g,r)]
        preds = fit.fit_predictions
        label = 'fit ({}={:.3f})'.format(cfg.score_type, cfg.score(series.expression,preds))
        ax.plot(series.ages, preds, linewidth=2, label=label)
        preds = fit.LOO_predictions
        label = 'LOO ({}={:.3f})'.format(cfg.score_type, cfg.score(series.expression,preds))
        ax.plot(series.ages, preds, linewidth=2, label=label)
    a,h,mu,w,p = fit.P
    sigma = 1/p
    P_ttl = r'(a={a:.2f}, h={h:.2f}, $\mu$={mu:.2f}, w={w:.2f}, $\sigma$={sigma:.2f})'.format(**locals())
    ttl = 'Gene: {}, Region: {}\n{}'.format(series.gene_name, series.region_name, P_ttl)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_ylabel('Expression Level', fontsize=cfg.fontsize)
    ax.set_xlabel('Age [years]', fontsize=cfg.fontsize)
    ax.legend()
    return fig
    
def plot_and_save_all_genes(data, dirname):
    from os.path import join
    fits = get_all_fits(data)
    for iGene,gene_name in enumerate(data.gene_names):
        print 'Saving figure for gene {}'.format(gene_name)
        fig = plot_gene(data,iGene,fits)
        filename = join(dirname, 'fits_{}.png'.format(gene_name))
        save_figure(fig, filename, b_close=True)

def plot_and_save_all_series(data, dirname):
    from os.path import join
    fits = get_all_fits(data)
    for iGene,gene_name in enumerate(data.gene_names):
        for iRegion, region_name in enumerate(data.region_names):
            print 'Saving figure for {}@{}'.format(gene_name,region_name)
            series = data.get_one_series(iGene,iRegion)
            fig = plot_one_series(series,fits)
            filename = join(dirname, 'fit-{}-{}.png'.format(gene_name,region_name))
            save_figure(fig, filename, b_close=True)
