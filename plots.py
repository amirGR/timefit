import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from all_fits import get_all_fits
from sigmoid_fit import high_res_preds
from fit_score import loo_score
import utils
import os.path
from project_dirs import resources_dir
from utils import ensure_dir

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
            x_smooth,y_smooth = high_res_preds(series.ages, fit.P[:-1])
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=2)
        ax.set_title('Region {}'.format(series.region_name))
        if iRegion % 4 == 0:
            ax.set_ylabel('Expression Level')
        if iRegion / 4 >= 3:
            ax.set_xlabel('Age [years]')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(series.gene_name))
    return fig
    
def plot_one_series(series, fits=None, fit=None):
    g,r = series.gene_name, series.region_name
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.ages,series.expression,'ro')
    if fit is None and fits is not None:
        fit = fits[(g,r)]
    if fit is not None:
        preds = fit.fit_predictions
        x_smooth,y_smooth = high_res_preds(series.ages, fit.P[:-1])        
        label = 'fit ({}={:.3f})'.format(cfg.score_type, cfg.score(series.expression,preds))
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=label)
        preds = fit.LOO_predictions
        for i,(x,y,y_loo) in enumerate(zip(series.ages, series.expression, preds)):
            label = 'LOO ({}={:.3f})'.format(cfg.score_type, loo_score(series.expression,preds)) if i==0 else None
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
    with utils.interactive(False):
        for iGene,gene_name in enumerate(data.gene_names):
            print 'Saving figure for gene {}'.format(gene_name)
            fig = plot_gene(data,iGene,fits)
            filename = os.path.join(dirname, '{}.png'.format(gene_name))
            save_figure(fig, filename, b_close=True)

def plot_and_save_all_series(data, dirname):
    ensure_dir(dirname)
    fits = get_all_fits(data)
    with utils.interactive(False):
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
    ttl = 'LOO R2 score distribution'
    if n_low > 0:
        ttl = ttl + '\n(another {} scores below {})'.format(n_low,low)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_xlabel('R2', fontsize=cfg.fontsize)
    ax.set_ylabel('Count', fontsize=cfg.fontsize)    
    return fig

def create_html(data, basedir, gene_dir, series_dir):
    from os.path import join
    from jinja2 import Template
    import shutil

    fits = get_all_fits(data)    
    html = Template("""
<html>
<head>
    <link rel="stylesheet" type="text/css" href="fits.css">
</head>
<body>
<H1>Fits for every Gene and Region</H1>
<P>
<a href="{{data.pathway}}-R2-hist.png">Distribution of LOO R2 scores</a>
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
        {% for region_name in sorted_regions %}
        <td>
            <a href="{{series_dir}}/fit-{{gene_name}}-{{region_name}}.png">
                <div {{'class="highScore"' if  fits[(gene_name,region_name)].LOO_score > highScore}}>
               {{fits[(gene_name,region_name)].LOO_score | round(2)}}
               </div>
            </a>
        </td>
        {% endfor %}
    </tr>
    {% endfor %}
</table>
</P>

</body>
</html>    
""").render(sorted_regions=cfg.sorted_regions, highScore=cfg.html_table_threshold_score, **locals())
    with open(join(basedir,'{}-fits.html'.format(data.pathway)), 'w') as f:
        f.write(html)
    
    shutil.copy(os.path.join(resources_dir(),'fits.css'), basedir)

def save_fits_and_create_html(data, fits, basedir):
    ensure_dir(basedir)
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'
    plot_and_save_all_genes(data, os.path.join(basedir,gene_dir))
    plot_and_save_all_series(data, os.path.join(basedir,series_dir))
    with utils.interactive(False):
        fig = plot_score_distribution(fits)
        save_figure(fig, os.path.join(basedir,'{}-R2-hist.png'.format(data.pathway)), b_close=True)
    create_html(data, basedir, gene_dir, series_dir)