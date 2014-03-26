import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from all_fits import get_all_fits
from fit_score import loo_score
from os.path import join, isfile
from project_dirs import resources_dir, results_dir, fit_results_relative_path
from utils.misc import ensure_dir, interactive
from utils.parallel import Parallel

def save_figure(fig, filename, b_close=False):
    fig.set_size_inches(cfg.default_figure_size_x, cfg.default_figure_size_y)
    fig.savefig(filename, facecolor=cfg.default_figure_facecolor, dpi=cfg.default_figure_dpi)
    if b_close:
        plt.close(fig)

def plot_gene(data, g, fits=None):
    fig = plt.figure()
    for iRegion,r in enumerate(cfg.sorted_regions[data.dataset]):
        series = data.get_one_series(g,r)
        ax = fig.add_subplot(4,4,iRegion+1)
        ax.plot(series.ages,series.expression,'ro')
        if fits is not None:
            fit = fits[(g,r)]
            if fit.theta is not None:
                x_smooth,y_smooth = fit.fitter.shape.high_res_preds(fit.theta, series.ages)
                ax.plot(x_smooth, y_smooth, 'b-', linewidth=2)
        ax.set_title('Region {}'.format(r))
        if iRegion % 4 == 0:
            ax.set_ylabel('Expression Level')
        if iRegion / 4 >= 3:
            ax.set_xlabel('Age')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(g))
    return fig

def plot_one_series(series, shape=None, theta=None, LOO_predictions=None):
    x = series.ages
    y = series.expression    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(series.ages,series.expression,'ro')
    ax.set_ylabel('Expression Level', fontsize=cfg.fontsize)
    ax.set_xlabel('Age', fontsize=cfg.fontsize)
    ttl = 'Gene: {}, Region: {}'.format(series.gene_name, series.region_name)

    if shape is not None and theta is not None:
        more_ttl = shape.format_params(theta,latex=True)
        if more_ttl:
            ttl = '\n'.join([ttl, more_ttl])
        
        score = cfg.score(y,shape.f(theta,x))
        x_smooth,y_smooth = shape.high_res_preds(theta,x)        
        label = 'fit ({}={:.3g})'.format(cfg.score_type, score)
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=label)

        if LOO_predictions is not None:
            score = loo_score(y,LOO_predictions)
            for i,(xi,yi,y_loo) in enumerate(zip(x,y,LOO_predictions)):
                if y_loo is None or np.isnan(y_loo):
                    continue
                label = 'LOO ({}={:.3g})'.format(cfg.score_type, score) if i==0 else None
                ax.plot([xi, xi], [yi, y_loo], 'g-', linewidth=2, label=label)
                ax.plot(xi, y_loo, 'gx')
        ax.legend()
        
    ax.set_title(ttl, fontsize=cfg.fontsize)
    return fig

def plot_and_save_all_genes(data, fitter, dirname):
    ensure_dir(dirname)
    fits = get_all_fits(data, fitter)
    to_plot = []
    for g in data.gene_names:
        filename = join(dirname, '{}.png'.format(g))
        if isfile(filename):
            print 'Figure already exists for gene {}. skipping...'.format(g)
        else:
            to_plot.append((g,filename))    
    pool = Parallel(_plot_genes_job)
    pool(pool.delay(data,fits,g,filename) for g,filename in to_plot)

def _plot_genes_job(data,fits,gene,filename):
    with interactive(False):
        print 'Saving figure for gene {}'.format(gene)
        fig = plot_gene(data,gene,fits)
        save_figure(fig, filename, b_close=True)

def plot_and_save_all_series(data, fitter, dirname, k_of_n=None):
    ensure_dir(dirname)
    fits = get_all_fits(data,fitter,k_of_n)
    to_plot = []
    for g,r in fits.iterkeys():
        filename = join(dirname, 'fit-{}-{}.png'.format(g,r))
        if isfile(filename):
            print 'Figure already exists for {}@{}. skipping...'.format(g,r)
        else:
            to_plot.append((g,r,filename))
    pool = Parallel(_plot_series_job)
    pool(pool.delay(data.get_one_series(g,r),fits[(g,r)],filename) for g,r,filename in to_plot)

def _plot_series_job(series,fit,filename):
    with interactive(False):
        print 'Saving figure for {}@{}'.format(series.gene_name, series.region_name)
        fig = plot_one_series(series, fit.fitter.shape, fit.theta, fit.LOO_predictions)
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
    from jinja2 import Template
    import shutil

    fits = get_all_fits(data,fitter)
    n_ranks = 5 # actually we'll have ranks of 0 to n_ranks
    for fit in fits.itervalues():
        if fit.LOO_score is None or fit.LOO_score < 0:
            fit.rank = 0
        else:
            fit.rank = int(np.ceil(n_ranks * fit.LOO_score))
            
    sorted_regions = cfg.sorted_regions[data.dataset]
    
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
""").render(**locals())
    with open(join(basedir,'fits.html'), 'w') as f:
        f.write(html)
    
    shutil.copy(join(resources_dir(),'fits.css'), basedir)

def save_fits_and_create_html(data, fitter, basedir=None, do_genes=True, do_series=True, do_hist=True, do_html=True, k_of_n=None):
    if basedir is None:
        basedir = join(results_dir(), fit_results_relative_path(data,fitter))
    print 'Writing HTML under {}'.format(basedir)
    ensure_dir(basedir)
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'
    if do_genes and k_of_n is None:
        plot_and_save_all_genes(data, fitter, join(basedir,gene_dir))
    if do_series:
        plot_and_save_all_series(data, fitter, join(basedir,series_dir),k_of_n)
    if do_hist and k_of_n is None:
        with interactive(False):
            fits = get_all_fits(data,fitter)
            fig = plot_score_distribution(fits)
            save_figure(fig, join(basedir,'R2-hist.png'), b_close=True)
    if do_html and k_of_n is None:
        create_html(data, fitter, basedir, gene_dir, series_dir)
