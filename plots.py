import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from all_fits import get_all_fits
from fit_score import loo_score
from os.path import join, isfile
from project_dirs import resources_dir, results_dir, fit_results_relative_path
from utils.misc import ensure_dir, interactive, rect_subplot
from utils.parallel import Parallel
from dev_stages import dev_stages
import scalers

def save_figure(fig, filename, b_close=False, b_square=False, show_frame=True):
    figure_size_x = cfg.default_figure_size_x_square if b_square else cfg.default_figure_size_x
    fig.set_size_inches(figure_size_x, cfg.default_figure_size_y)
    if show_frame:
        facecolor = cfg.default_figure_facecolor
    else:
        facecolor = 'white'
    fig.savefig(filename, facecolor=facecolor, dpi=cfg.default_figure_dpi)
    if b_close:
        plt.close(fig)

def plot_gene(data, g, fits=None):
    region_series_fits = _extract_gene_data(data,g,fits)
    return _plot_gene_inner(g,region_series_fits)

def _extract_gene_data(data, g, fits=None):
    dct_dataset = data.region_to_dataset()
    region_series_fits = []
    for r in data.region_names:
        series = data.get_one_series(g,r,allow_missing=True)
        if series is None:
            continue
        if fits is not None:
            dsname = dct_dataset[r]
            fit = fits[dsname][(g,r)]
        else:
            fit = None
        region_series_fits.append( (r,series,fit) )
    if not region_series_fits:
        raise Exception('Gene not found in the data')
    return region_series_fits

def _plot_gene_inner(g,region_series_fits):
    fig = plt.figure()
    nRows, nCols = rect_subplot(len(region_series_fits))
    for iRegion,(r,series,fit) in enumerate(region_series_fits):
        ax = fig.add_subplot(nRows,nCols,iRegion+1)
        ax.plot(series.ages,series.expression,'ro')
        if fit is not None and fit.theta is not None:
            x_smooth,y_smooth = fit.fitter.shape.high_res_preds(fit.theta, series.ages)
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=2)
        ax.set_title('Region {}'.format(r))
        if iRegion % nCols == 0:
            ax.set_ylabel('expression level')
        if iRegion / nRows == nRows-1:
            ax.set_xlabel('age')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(g))
    return fig

def plot_one_series(series, shape=None, theta=None, LOO_predictions=None):
    x = series.ages
    y = series.expression    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # plot the data points
    ax.plot(series.ages,series.expression, 'ks', markersize=8)
    ax.set_ylabel('expression level', fontsize=cfg.fontsize)
    ax.set_xlabel('age', fontsize=cfg.fontsize)
    ttl = '{}@{}'.format(series.gene_name, series.region_name)
    
    # set the development stages as x labels
    stages = [stage.scaled(series.age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=cfg.xtick_fontsize, fontstretch='condensed', rotation=90)    
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = scalers.unify(series.age_scaler).scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    if shape is not None and theta is not None:
        # add fit parameters to title
        ttl = '{}, {} fit'.format(ttl, shape)
        more_ttl = shape.format_params(theta,latex=True)
        if more_ttl:
            ttl = '\n'.join([ttl, more_ttl])
        
        # draw the overall fit
        score = cfg.score(y,shape.f(theta,x))
        x_smooth,y_smooth = shape.high_res_preds(theta,x)        
        label = 'fit ({}={:.3g})'.format(cfg.score_type, score)
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=3, label=label)

        # draw LOO predictions and residuals
        if LOO_predictions is not None:
            score = loo_score(y,LOO_predictions)
            for i,(xi,yi,y_loo) in enumerate(zip(x,y,LOO_predictions)):
                if y_loo is None or np.isnan(y_loo):
                    continue
                label = 'LOO ({}={:.3g})'.format(cfg.score_type, score) if i==0 else None
                ax.plot([xi, xi], [yi, y_loo], '-', color='0.5', label=label)
                ax.plot(xi, y_loo, 'x', color='0.5', markeredgewidth=2)
        ax.legend(fontsize=cfg.fontsize, frameon=False)
        
    ax.set_title(ttl, fontsize=cfg.fontsize)
    return fig

def plot_and_save_all_genes(data, fitter, fits, dirname):
    ensure_dir(dirname)
    to_plot = []
    genes = set() # use the genes from the fits and not from 'data' to support sharding (k_of_n)
    for ds_fits in fits.itervalues():
        for g,r in ds_fits.iterkeys():
            genes.add(g)
    for g in sorted(genes):
        filename = join(dirname, '{}.png'.format(g))
        if isfile(filename):
            print 'Figure already exists for gene {}. skipping...'.format(g)
            continue
        region_series_fits = _extract_gene_data(data,g,fits)            
        to_plot.append((g,region_series_fits,filename))
    pool = Parallel(_plot_genes_job)
    pool(pool.delay(*args) for args in to_plot)

def _plot_genes_job(gene,region_series_fits,filename):
    with interactive(False):
        print 'Saving figure for gene {}'.format(gene)
        fig = _plot_gene_inner(gene,region_series_fits)
        save_figure(fig, filename, b_close=True)

def plot_and_save_all_series(data, fitter, fits, dirname):
    ensure_dir(dirname)
    to_plot = []
    for dsfits in fits.itervalues():
        for (g,r),fit in dsfits.iteritems():
            filename = join(dirname, 'fit-{}-{}.png'.format(g,r))
            if isfile(filename):
                print 'Figure already exists for {}@{}. skipping...'.format(g,r)
                continue
            series = data.get_one_series(g,r)
            to_plot.append((series,fit,filename))
    pool = Parallel(_plot_series_job)
    pool(pool.delay(*args) for args in to_plot)

def _plot_series_job(series,fit,filename):
    with interactive(False):
        print 'Saving figure for {}@{}'.format(series.gene_name, series.region_name)
        fig = plot_one_series(series, fit.fitter.shape, fit.theta, fit.LOO_predictions)
        save_figure(fig, filename, b_close=True)

def plot_score_distribution(fits):
    def flat_values(fits):
        return (fit for dsfits in fits.itervalues() for fit in dsfits.itervalues())
    n_failed = len([1 for fit in flat_values(fits) if fit.LOO_score is None])
    LOO_R2 = np.array([fit.LOO_score for fit in flat_values(fits) if fit.LOO_score is not None])
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
    ax.set_ylabel('count', fontsize=cfg.fontsize)    
    return fig

def create_html(data, fitter, fits, basedir, gene_dir, series_dir):
    from jinja2 import Template
    import shutil

    n_ranks = 5 # actually we'll have ranks of 0 to n_ranks
    flat_fits = {} # (gene,region) -> fit (may be None)
    for g in data.gene_names:
        for r in data.region_names:
            flat_fits[(g,r)] = None
    for dsfits in fits.itervalues():
        for (g,r),fit in dsfits.iteritems():
            fit.rank = int(np.ceil(n_ranks * fit.LOO_score)) if fit.LOO_score > 0 else 0
            flat_fits[(g,r)] = fit     
            
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
        {% for region_name in data.region_names %}
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
            {% if flat_fits[(gene_name,region_name)] %}
                <a href="{{series_dir}}/fit-{{gene_name}}-{{region_name}}.png">
                {% if flat_fits[(gene_name,region_name)].LOO_score %}
                    <div class="score rank{{flat_fits[(gene_name,region_name)].rank}}">
                   {{flat_fits[(gene_name,region_name)].LOO_score | round(2)}}
                   </div>
                {% else %}
                   No Score
                {% endif %}
                </a>
            {% endif %}
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

def save_fits_and_create_html(data, fitter, fits=None, basedir=None, do_genes=True, do_series=True, do_hist=True, do_html=True, k_of_n=None):
    if fits is None:
        fits = get_all_fits(data,fitter,k_of_n)
    if basedir is None:
        basedir = join(results_dir(), fit_results_relative_path(data,fitter))
    print 'Writing HTML under {}'.format(basedir)
    ensure_dir(basedir)
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'
    if do_genes: # relies on the sharding of the fits respecting gene boundaries
        plot_and_save_all_genes(data, fitter, fits, join(basedir,gene_dir))
    if do_series:
        plot_and_save_all_series(data, fitter, fits, join(basedir,series_dir))
    if do_hist and k_of_n is None:
        with interactive(False):
            fig = plot_score_distribution(fits)
            save_figure(fig, join(basedir,'R2-hist.png'), b_close=True)
    if do_html and k_of_n is None:
        create_html(data, fitter, fits, basedir, gene_dir, series_dir)
