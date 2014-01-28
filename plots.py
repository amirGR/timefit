import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from all_fits import get_all_fits
import os.path
from os import makedirs
    
def ensure_dir(d):
    if not os.path.exists(d):
        makedirs(d)

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
        ax.plot(series.ages, preds, 'b-', linewidth=2, label=label)
        preds = fit.LOO_predictions
        for i,(x,y,y_loo) in enumerate(zip(series.ages, series.expression, preds)):
            label = 'LOO ({}={:.3f})'.format(cfg.score_type, cfg.score(series.expression,preds)) if i==0 else None
            ax.plot([x, x], [y, y_loo], 'g-', linewidth=2, label=label)
            ax.plot(x, y_loo, 'gx')
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
    ensure_dir(dirname)
    fits = get_all_fits(data)
    for iGene,gene_name in enumerate(data.gene_names):
        print 'Saving figure for gene {}'.format(gene_name)
        fig = plot_gene(data,iGene,fits)
        filename = os.path.join(dirname, '{}.png'.format(gene_name))
        save_figure(fig, filename, b_close=True)

def plot_and_save_all_series(data, dirname):
    ensure_dir(dirname)
    fits = get_all_fits(data)
    for iGene,gene_name in enumerate(data.gene_names):
        for iRegion, region_name in enumerate(data.region_names):
            print 'Saving figure for {}@{}'.format(gene_name,region_name)
            series = data.get_one_series(iGene,iRegion)
            fig = plot_one_series(series,fits)
            filename = os.path.join(dirname, 'fit-{}-{}.png'.format(gene_name,region_name))
            save_figure(fig, filename, b_close=True)

def plot_score_distribution(fits):
    LOO_R2 = np.array([fit.LOO_score for fit in fits.itervalues()])
    low,high = -1, 1
    n_low = np.count_nonzero(LOO_R2 < low)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(LOO_R2, 50, range=(low,high))
    ax.set_title('LOO R2 score distribution\n(another {} scores below {})'.format(n_low,low), fontsize=cfg.fontsize)
    ax.set_xlabel('R2', fontsize=cfg.fontsize)
    ax.set_ylabel('Count', fontsize=cfg.fontsize)    
    return fig

def create_html(data, basedir, gene_dir, series_dir):
    from os.path import join
    from jinja2 import Template

    fits = get_all_fits(data)    
    html = Template("""
<html>
<head>
    <link rel="stylesheet" type="text/css" href="fits.css">
</head>
<body>
<H1>Fits for every Gene and Region</H1>
<P>
<a href="R2-hist.png">Distribution of LOO R2 scores</a>
<table>
    <th>
        {% for region_name in sorted_regions %}
        <td class="tableHeading">
            <b>{{region_name}}</b>
        </td>
        {% endfor %}
    </th>
    {% for gene_name in data.gene_names %}
    <tr>
        <td>
            <a href="{{gene_dir}}/{{gene_name}}.png"><b>{{gene_name}}</b></a>
        </td>
        {% for region_name in data.region_names %}
        <td>
            <a href="{{series_dir}}/fit-{{gene_name}}-{{region_name}}.png">
               {{fits[(gene_name,region_name)].LOO_score | round(2)}}
            </a>
        </td>
        {% endfor %}
    </tr>
    {% endfor %}
</table>
</P>

</body>
</html>    
""").render(sorted_regions=cfg.sorted_regions,**locals())
    with open(join(basedir,'fits.html'), 'w') as f:
        f.write(html)

def save_fits_and_create_html(data, dirname):
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'
    plot_and_save_all_genes(data, os.path.join(dirname,gene_dir))
    plot_and_save_all_series(data, os.path.join(dirname,series_dir))
    fig = plot_score_distribution(fits)
    save_figure(fig, os.path.join(dirname,'R2-hist.png'), b_close=True)
    create_html(data, dirname, gene_dir, series_dir)