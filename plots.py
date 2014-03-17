import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from all_fits import get_all_fits
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
            if fit.theta is not None:
                x_smooth,y_smooth = fit.fitter.shape.high_res_preds(fit.theta, series.ages)
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
    ttl = 'Gene: {}, Region: {}'.format(series.gene_name, series.region_name)
    if fit is None and fits is not None:
        fit = fits[(g,r)]
    if fit is not None:
        preds = fit.fit_predictions
        if fit.theta is not None:
            x_smooth,y_smooth = fit.fitter.shape.high_res_preds(fit.theta, series.ages)        
            label = 'fit ({}={:.3f})'.format(cfg.score_type, cfg.score(series.expression,preds))
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=label)
        preds = fit.LOO_predictions
        for i,(x,y,y_loo) in enumerate(zip(series.ages, series.expression, preds)):
            if y_loo is None or np.isnan(y_loo):
                continue
            label = 'LOO ({}={:.3f})'.format(cfg.score_type, loo_score(series.expression,preds)) if i==0 else None
            ax.plot([x, x], [y, y_loo], 'g-', linewidth=2, label=label)
            ax.plot(x, y_loo, 'gx')
        if fit.theta is not None:
            P_ttl = fit.fitter.format_params(fit.theta, fit.sigma, latex=True)
        else:
            P_ttl = 'Global fit failed - parameters not extracted'
        ttl = '{}\n{}'.format(ttl,P_ttl)
        ax.legend()
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_ylabel('Expression Level', fontsize=cfg.fontsize)
    ax.set_xlabel('Age [years]', fontsize=cfg.fontsize)
    return fig
    
def plot_and_save_all_genes(data, fitter, dirname):
    ensure_dir(dirname)
    fits = get_all_fits(data, fitter)
    with utils.interactive(False):
        for iGene,gene_name in enumerate(data.gene_names):
            print 'Saving figure for gene {}'.format(gene_name)
            fig = plot_gene(data,iGene,fits)
            filename = os.path.join(dirname, '{}.png'.format(gene_name))
            save_figure(fig, filename, b_close=True)

def plot_and_save_all_series(data, fitter, dirname):
    ensure_dir(dirname)
    fits = get_all_fits(data,fitter)
    with utils.interactive(False):
        for iGene,gene_name in enumerate(data.gene_names):
            for iRegion, region_name in enumerate(data.region_names):
                print 'Saving figure for {}@{}'.format(gene_name,region_name)
                series = data.get_one_series(iGene,iRegion)
                fig = plot_one_series(series,fits)
                filename = os.path.join(dirname, 'fit-{}-{}.png'.format(gene_name,region_name))
                save_figure(fig, filename, b_close=True)

def plot_score_distribution(fits):
    n_failed = len([1 for fit in fits.itervalues() if fit.LOO_score is None])
    LOO_R2 = np.array([fit.LOO_score for fit in fits.itervalues() if fit.LOO_score is not None])
    low,high = -1, 1
    n_low = np.count_nonzero(LOO_R2 < low)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(LOO_R2, 50, range=(low,high))
    ttl = 'LOO R2 score distribution'
    if n_low or n_failed:
        ttl = ttl + '\n(another {} failed fits and {} scores below {})'.format(n_failed, n_low,low)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_xlabel('R2', fontsize=cfg.fontsize)
    ax.set_ylabel('Count', fontsize=cfg.fontsize)    
    return fig

def create_html(data, fitter, basedir, gene_dir, series_dir):
    from os.path import join
    from jinja2 import Template
    import shutil

    fits = get_all_fits(data,fitter)
    n_ranks = 5 # actually we'll have ranks of 0 to n_ranks
    for fit in fits.itervalues():
        if fit.LOO_score is None or fit.LOO_score < 0:
            fit.rank = 0
        else:
            fit.rank = int(np.ceil(n_ranks * fit.LOO_score))
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
        {% for region_name in sorted_regions %}
        <td>
            <a href="{{series_dir}}/fit-{{gene_name}}-{{region_name}}.png">
            {% if fits[(gene_name,region_name)].LOO_score %}
                <div class="score rank{{fits[(gene_name,region_name)].rank}}">
               {{fits[(gene_name,region_name)].LOO_score | round(2)}}
               </div>
            {% else %}
               None
            {% endif %}
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
    with open(join(basedir,'fits.html'), 'w') as f:
        f.write(html)
    
    shutil.copy(os.path.join(resources_dir(),'fits.css'), basedir)

def save_fits_and_create_html(data, fitter, basedir, do_genes=True, do_series=True, do_hist=True, do_html=True):
    ensure_dir(basedir)
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'
    if do_genes:
        plot_and_save_all_genes(data, fitter, os.path.join(basedir,gene_dir))
    if do_series:
        plot_and_save_all_series(data, fitter, os.path.join(basedir,series_dir))
    if do_hist:
        with utils.interactive(False):
            fits = get_all_fits(data,fitter)
            fig = plot_score_distribution(fits)
            save_figure(fig, os.path.join(basedir,'R2-hist.png'), b_close=True)
    if do_html:
        create_html(data, fitter, basedir, gene_dir, series_dir)
