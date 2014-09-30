import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import load_17_pathways_breakdown
from all_fits import get_all_fits, iterate_fits
from fit_score import loo_score
import os.path
from os.path import join, isfile
from sklearn.datasets.base import Bunch
from utils.statsmodels_graphics.correlation import plot_corr
from project_dirs import resources_dir, results_dir, fit_results_relative_path
from utils.misc import ensure_dir, interactive, rect_subplot
from utils.parallel import Parallel
from dev_stages import dev_stages
import scalers

def save_figure(fig, filename, b_close=False, b_square=True, show_frame=False, under_results=False):
    if under_results:
        dirname = results_dir()
        filename = join(dirname,filename)
        ensure_dir(os.path.dirname(filename))
    if cfg.verbosity >= 1:
        print 'Saving figure to {}'.format(filename)
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

def _plot_gene_inner(g, region_series_fits, change_distribution_bin_centers=None):
    fig = plt.figure()
    nRows, nCols = rect_subplot(len(region_series_fits))
    for iRegion,(r,series,fit) in enumerate(region_series_fits):
        ax = fig.add_subplot(nRows,nCols,iRegion+1)
        if change_distribution_bin_centers is None:
            change_distribution = None
        else:
            change_distribution = Bunch(
                centers = change_distribution_bin_centers,
                weights = fit.change_distribution_weights,
            )
        plot_one_series(series, fit.fitter.shape, fit.theta, change_distribution=change_distribution, minimal_annotations=True, ax=ax)
        ax.set_title('Region {}'.format(r))
        if iRegion % nCols == 0:
            ax.set_ylabel('expression level')
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Gene {}'.format(g))
    return fig

def add_age_ticks(ax, age_scaler, fontsize=None):
    if fontsize is None:
        fontsize = cfg.fontsize
        
    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=fontsize, fontstretch='condensed', rotation=90)    
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = scalers.unify(age_scaler).scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')
    
def plot_one_series(series, shape=None, theta=None, LOO_predictions=None, change_distribution=None, minimal_annotations=False, ax=None):
    x = series.ages
    y = series.single_expression
    b_subplot = ax is not None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    fontsize = cfg.minimal_annotation_fontsize if minimal_annotations else cfg.fontsize
    
    # plot the data points
    markersize = 8 if not minimal_annotations else 4
    ax.plot(series.ages, y, 'ks', markersize=markersize)
    if not b_subplot:
        ax.set_ylabel('expression level', fontsize=fontsize)
        ax.set_xlabel('age', fontsize=fontsize)
    ttl = '{}@{}'.format(series.gene_name, series.region_name)
    add_age_ticks(ax, series.age_scaler, fontsize)

    # plot change distribution if provided
    if change_distribution:
        centers = change_distribution.centers
        width = centers[1] - centers[0]
        weights = change_distribution.weights
        weights *= 0.9 * (ymax - ymin) / weights.max()
        ax.bar(centers, weights, width=width, bottom=ymin, color='g', alpha=0.1)

    if shape is not None and theta is not None:
        # add fit parameters to title
        ttl = '{}, {} fit'.format(ttl, shape)
        more_ttl = shape.format_params(theta, series.age_scaler, latex=True)
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
        if not minimal_annotations:
            ax.legend(fontsize=fontsize, frameon=False)
        
    if not minimal_annotations:
        ax.tick_params(axis='y', labelsize=fontsize)
        if not b_subplot:
            ax.set_title(ttl, fontsize=fontsize)
    return ax.figure

def plot_series(series, shape=None, theta=None, LOO_predictions=None):
    if series.num_genes == 1:
        return plot_one_series(series, shape, theta, LOO_predictions)
    fig = plt.figure()
    nRows, nCols = rect_subplot(series.num_genes)
    for iGene,g in enumerate(series.gene_names):
        ax = fig.add_subplot(nRows,nCols,iGene+1)
        theta_i = theta[iGene] if theta is not None else None
        LOO_i = LOO_predictions[:,iGene] if LOO_predictions is not None else None
        plot_one_series(series.get_single_gene_series(iGene), shape, theta_i, LOO_i, ax=ax)
        ax.set_title('Gene {}'.format(g), fontsize=cfg.fontsize)
    fig.tight_layout(h_pad=0,w_pad=0)
    fig.suptitle('Region {}'.format(series.region_name), fontsize=cfg.fontsize)
    return fig

def plot_and_save_all_genes(data, fitter, fits, dirname, show_change_distributions):
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
        if show_change_distributions:
            bin_centers = fits.change_distribution_params.bin_centers
        else:
            bin_centers = None
        to_plot.append((g,region_series_fits,filename, bin_centers))
    pool = Parallel(_plot_genes_job)
    pool(pool.delay(*args) for args in to_plot)

def _plot_genes_job(gene, region_series_fits, filename, bin_centers):
    with interactive(False):
        print 'Saving figure for gene {}'.format(gene)
        fig = _plot_gene_inner(gene, region_series_fits, change_distribution_bin_centers=bin_centers)
        save_figure(fig, filename, b_close=True)

def plot_and_save_all_series(data, fitter, fits, dirname, use_correlations, show_change_distributions):
    ensure_dir(dirname)
    to_plot = []
    for dsfits in fits.itervalues():
        for (g,r),fit in dsfits.iteritems():
            filename = join(dirname, 'fit-{}-{}.png'.format(g,r))
            if isfile(filename):
                print 'Figure already exists for {}@{}. skipping...'.format(g,r)
                continue
            series = data.get_one_series(g,r)
            if show_change_distributions:
                change_distribution = Bunch(
                    centers = fits.change_distribution_params.bin_centers,
                    weights = fit.change_distribution_weights,
                )
            else:
                change_distribution = None
            to_plot.append((series,fit,filename,use_correlations, change_distribution))
    if cfg.parallel_run_locally:
        for args in to_plot:
            _plot_series_job(*args)
    else:
        pool = Parallel(_plot_series_job)
        pool(pool.delay(*args) for args in to_plot)

def _plot_series_job(series, fit, filename, use_correlations, change_distribution):
    with interactive(False):
        print 'Saving figure for {}@{}'.format(series.gene_name, series.region_name)
        if use_correlations:
            preds = fit.with_correlations.LOO_predictions
        else:
            preds = fit.LOO_predictions
        fig = plot_one_series(series, fit.fitter.shape, fit.theta, preds, change_distribution=change_distribution)
        save_figure(fig, filename, b_close=True)

def get_scores_from_fits(fits, use_correlations):
    if use_correlations:
        R2_pairs = [(fit.LOO_score,fit.with_correlations.LOO_score) for fit in iterate_fits(fits)]
        R2_pairs = [(s1,s2) for s1,s2 in R2_pairs if s1>-1 and s2>-1]
        basic = np.array([b for b,m in R2_pairs])
        multi = np.array([m for b,m in R2_pairs])
    else:
        basic = np.array([fit.LOO_score for fit in iterate_fits(fits) if fit.LOO_score>-1])
        multi = None
    return basic,multi

def plot_score_distribution(fits, use_correlations):
    basic, multi = get_scores_from_fits(fits, use_correlations=use_correlations)    
    low,high = -1, 1
    def do_hist(scores):
        counts,bin_edges = np.histogram(scores,50,range=(low,high))
        probs = counts / float(sum(counts))
        width = bin_edges[1] - bin_edges[0]
        return bin_edges[:-1],probs,width
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos, probs, width = do_hist(basic)
    ax.bar(pos, probs, width=width, color='b', label='Single Gene')
    if use_correlations:
        pos, probs, width = do_hist(multi)
        ax.bar(pos, probs, width=width, color='g', alpha=0.5, label='Using Correlations')
        ax.legend(loc='best', fontsize=cfg.fontsize, frameon=False)
    ax.set_xlabel('test set $R^2$', fontsize=cfg.fontsize)
    ax.set_ylabel('probability', fontsize=cfg.fontsize)   
    ax.tick_params(axis='both', labelsize=cfg.fontsize)
    return fig

def plot_score_comparison_scatter_for_correlations(fits):
    basic, multi = get_scores_from_fits(fits, use_correlations=True)    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.8])
    ax.scatter(basic, multi, alpha=0.8, label='data')
    ax.plot(np.mean(basic), np.mean(multi), 'rx', markersize=8, markeredgewidth=2, label='mean')
    ax.plot([-1, 1], [-1, 1],'k--')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ticks = [-1,1]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=cfg.fontsize)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=cfg.fontsize)
    ax.set_xlabel('$R^2$ for single gene fits', fontsize=cfg.fontsize)
    ax.set_ylabel('multi gene $R^2$', fontsize=cfg.fontsize)
    ax.set_title('$R^2$ change using correlations', fontsize=cfg.fontsize)
    ax.legend(fontsize=cfg.fontsize, frameon=False, loc='upper left')
    return fig

def plot_score_improvement_histogram_for_correlations(fits):
    basic, multi = get_scores_from_fits(fits, use_correlations=True)    
    delta = multi - basic    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.72])
    ax.hist(delta, bins=20)
    ttl1 = r'$\Delta R^2$ using correlations'
    ttl2 = r'$\Delta R^2 = {:.2g} \pm {:.2g}$'.format(np.mean(delta), np.std(delta))
    ttl = '{}\n{}\n'.format(ttl1,ttl2)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_xlabel('$\Delta R^2$', fontsize=cfg.fontsize)
    ax.set_ylabel('number of gene-regions', fontsize=cfg.fontsize)   
    ax.tick_params(axis='both', labelsize=cfg.fontsize)
    return fig
            

def create_score_distribution_html(fits, use_correlations, dirname):
    ensure_dir(dirname)
    with interactive(False):
        hist_filename = 'R2-hist.png'
        fig = plot_score_distribution(fits,use_correlations)
        save_figure(fig, join(dirname,hist_filename), b_close=True)
        
        if use_correlations:
            scatter_filename = 'comparison-scatter.png'
            fig = plot_score_comparison_scatter_for_correlations(fits)
            save_figure(fig, join(dirname,scatter_filename), b_close=True)
            
            delta_hist_filename = 'R2-delta-hist.png'
            fig = plot_score_improvement_histogram_for_correlations(fits)
            save_figure(fig, join(dirname,delta_hist_filename), b_close=True)
                        
    image_size = "50%"
    from jinja2 import Template
    import shutil
    html = Template("""
<html>
<head>
    <link rel="stylesheet" type="text/css" href="score-distribution.css">
</head>
<body>
<center><H1>R2 Score Distribution</H1></center>

<a href="{{hist_filename}}"> <img src="{{hist_filename}}" height="{{image_size}}"></a>

{% if use_correlations %}
    <a href="{{delta_hist_filename}}"> <img src="{{delta_hist_filename}}" height="{{image_size}}"></a>
    <a href="{{scatter_filename}}"> <img src="{{scatter_filename}}" height="{{image_size}}"></a>
{% endif %}

</body>
</html>    
""").render(**locals())
    filename = join(dirname,'scores.html')
    with open(filename, 'w') as f:
        f.write(html)
    shutil.copy(join(resources_dir(),'score-distribution.css'), dirname)

def plot_and_save_all_gene_correlations(data, correlations, dirname):
    ensure_dir(dirname)
    for region in data.region_names:
        sigma = correlations[region]        
        ttl = 'Gene expression correlations at {}'.format(region)
        fig = plot_corr(sigma, xnames=data.gene_names, normcolor=True, title=ttl)
        save_figure(fig, join(dirname,'{}.png'.format(region)), b_close=True)

def create_html(data, fitter, fits, 
                basedir, gene_dir, series_dir, scores_dir,
                correlations_dir = None,
                use_correlations = False,
                link_to_correlation_plots = False,
                b_pathways = False, 
                show_R2 = True,
                gene_names=None, region_names=None, 
                extra_columns=None, extra_fields_per_fit=None,
                extra_top_links=None,
                b_inline_images=False, inline_image_size=None,
                b_R2_dist=True, ttl=None, top_text=None,
                filename=None):
    from jinja2 import Template
    import shutil

    if gene_names is None:
        gene_names = data.gene_names
    if region_names is None:
        region_names = data.region_names
    if extra_columns is None:
        extra_columns = []
    if extra_fields_per_fit is None:
        extra_fields_per_fit = []
    if extra_top_links is None:
        extra_top_links = []
    if inline_image_size is None:
        inline_image_size = '20%'
    if ttl is None:
        ttl = 'Fits for every Gene and Region'
    if filename is None:
        filename = 'fits'
    
    if b_pathways:
        create_pathway_index_html(data, fitter, fits, basedir, gene_dir, series_dir, use_correlations=use_correlations, b_unique=True)
        create_pathway_index_html(data, fitter, fits, basedir, gene_dir, series_dir, use_correlations=use_correlations, b_unique=False)

    n_ranks = 5 # actually we'll have ranks of 0 to n_ranks
    flat_fits = {} # (gene,region) -> fit (may be None)
    for g in gene_names:
        for r in region_names:
            flat_fits[(g,r)] = None
    for dsfits in fits.itervalues():
        for (g,r),fit in dsfits.iteritems():
            if use_correlations:
                score = fit.with_correlations.LOO_score
            else:
                score = fit.LOO_score
            fit.score = score
            fit.rank = int(np.ceil(n_ranks * score)) if score > 0 else 0
            flat_fits[(g,r)] = fit     
            
    extra_fields_per_fit = list(enumerate(extra_fields_per_fit))
    html = Template("""
<html>
<head>
    <link rel="stylesheet" type="text/css" href="fits.css">
</head>
<body>
<H1>{{ttl}}</H1>
{% if b_R2_dist %}
    <P>
    <a href="{{scores_dir}}/scores.html">Distribution of LOO R2 scores</a>
    </P>
{% endif %}
{% if b_pathways %}
    <P>
    <a href="pathway-fits-unique.html">Breakdown of fits for 17 pathways (unique)</a><br/>
    <a href="pathway-fits.html">Breakdown of fits for 17 pathways (overlapping)</a>
    </P>
{% endif %}
{% for href,txt in extra_top_links %}
    <a href={{href}}>{{txt}}</a>
{% endfor %}
{% if top_text %}
{{ top_text }}
{% endif %}
<P>
Column/row headings and/or cell entries may link to more details for the corresponding entity.
</P>
<P>
<table>
    <th>
        {% for column_name,dct_vals in extra_columns %}
        <td class="tableExtraColumnHeading">
            <b>{{column_name}}</b>
        </td>
        {% endfor %}
        {% for region_name in region_names %}
        <td class="tableHeading">
            {% if link_to_correlation_plots %}
                <a class=noColorChange href="{{correlations_dir}}/{{region_name}}.png"><b>{{region_name}}</b></a>
            {% else %}
                <b>{{region_name}}</b>
            {% endif %}
        </td>
        {% endfor %}
    </th>
    {% for gene_name in gene_names %}
    <tr>
        <td>
            <a href="{{gene_dir}}/{{gene_name}}.png"><b>{{gene_name}}</b></a>
        </td>
        {% for column_name,dct_vals in extra_columns %}
            <td>
                {{ '%.2g' | format(dct_vals[gene_name]) }}
            </td>
        {% endfor %}
        {% for region_name in region_names %}
        <td>
            {% if flat_fits[(gene_name,region_name)] %}
                <a class=noColorChange href="{{series_dir}}/fit-{{gene_name}}-{{region_name}}.png">
                {% if flat_fits[(gene_name,region_name)].score %}
                    {% if b_inline_images %}
                        R2 &nbsp; = &nbsp;
                    {% endif %}
                    {% if show_R2 %}
                        <div class="score rank{{flat_fits[(gene_name,region_name)].rank}}">
                            {{flat_fits[(gene_name,region_name)].score | round(2)}}
                        </div>
                    {% endif %}
                    {% for i,f in extra_fields_per_fit %}
                        {% set txt,cls = f(flat_fits[(gene_name,region_name)]) %}
                        {% if i>0 or show_R2 %}
                            <br/>
                        {% endif %}
                        <div class="fitField {{cls}}">
                            <b>{{txt}}</b>
                        </div>
                    {% endfor %}
                {% else %}
                   No Score
                {% endif %}
                {% if b_inline_images %}
                    <br/>
                    <img src="{{series_dir}}/fit-{{gene_name}}-{{region_name}}.png" height="{{inline_image_size}}">
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
    filename = join(basedir,'{}.html'.format(filename))
    print 'Saving HTML to {}'.format(filename)
    with open(filename, 'w') as f:
        f.write(html)
    shutil.copy(join(resources_dir(),'fits.css'), basedir)

def create_pathway_index_html(data, fitter, fits, basedir, gene_dir, series_dir, use_correlations, b_unique):
    from jinja2 import Template

    dct_pathways = load_17_pathways_breakdown(b_unique)

    n_ranks = 5 # actually we'll have ranks of 0 to n_ranks
    flat_fits = {} # (gene,region) -> fit (may be None)
    for g in data.gene_names:
        for r in data.region_names:
            flat_fits[(g,r)] = None
    for dsfits in fits.itervalues():
        for (g,r),fit in dsfits.iteritems():
            if use_correlations:
                score = fit.with_correlations.LOO_score
            else:
                score = fit.LOO_score
            fit.score = score
            fit.rank = int(np.ceil(n_ranks * score)) if score > 0 else 0
            flat_fits[(g,r)] = fit     
            
    html = Template("""
<html>
<head>
    <link rel="stylesheet" type="text/css" href="fits.css">
</head>
<body>
<H1>Fits broken down by pathway {% if b_unique %} (unique genes only) {% endif %} </H1>
{% for pathway_name, pathway_genes in dct_pathways.iteritems() %}
<P>
<H2>{{pathway_name}}</H2>
<table>
    <th>
        {% for region_name in data.region_names %}
        <td class="tableHeading">
            <b>{{region_name}}</b>
        </td>
        {% endfor %}
    </th>
    {% for gene_name in pathway_genes %}
    <tr>
        <td>
            <a href="{{gene_dir}}/{{gene_name}}.png"><b>{{gene_name}}</b></a>
        </td>
        {% for region_name in data.region_names %}
        <td>
            {% if flat_fits[(gene_name,region_name)] %}
                <a href="{{series_dir}}/fit-{{gene_name}}-{{region_name}}.png">
                {% if flat_fits[(gene_name,region_name)].score %}
                    <div class="score rank{{flat_fits[(gene_name,region_name)].rank}}">
                   {{flat_fits[(gene_name,region_name)].score | round(2)}}
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
{% endfor %} {# dct_pathways #}

</body>
</html>    
""").render(**locals())
    str_unique = '-unique' if b_unique else ''
    filename = 'pathway-fits{}.html'.format(str_unique)
    with open(join(basedir,filename), 'w') as f:
        f.write(html)
    
def save_fits_and_create_html(data, fitter, fits=None, basedir=None, 
                              do_genes=True, do_series=True, do_hist=True, do_html=True, only_main_html=False,
                              k_of_n=None, 
                              use_correlations=False, correlations=None,
                              show_change_distributions=False,
                              html_kw=None):
    if fits is None:
        fits = get_all_fits(data,fitter,k_of_n)
    if basedir is None:
        basedir = join(results_dir(), fit_results_relative_path(data,fitter))
        if use_correlations:
            basedir = join(basedir,'with-correlations')
    print 'Writing HTML under {}'.format(basedir)
    ensure_dir(basedir)
    gene_dir = 'gene-subplot'
    series_dir = 'gene-region-fits'
    correlations_dir = 'gene-correlations'
    scores_dir = 'score_distributions'
    if do_genes and not only_main_html: # relies on the sharding of the fits respecting gene boundaries
        plot_and_save_all_genes(data, fitter, fits, join(basedir,gene_dir), show_change_distributions)
    if do_series and not only_main_html:
        plot_and_save_all_series(data, fitter, fits, join(basedir,series_dir), use_correlations, show_change_distributions)
    if do_hist and k_of_n is None and not only_main_html:
        create_score_distribution_html(fits, use_correlations, join(basedir,scores_dir))
    if do_html and k_of_n is None:
        link_to_correlation_plots = use_correlations and correlations is not None
        if link_to_correlation_plots and not only_main_html:
            plot_and_save_all_gene_correlations(data, correlations, join(basedir,correlations_dir))
        dct_pathways = load_17_pathways_breakdown()
        pathway_genes = set.union(*dct_pathways.values())
        data_genes = set(data.gene_names)
        missing = pathway_genes - data_genes
        b_pathways = len(missing) < len(pathway_genes)/2 # simple heuristic to create pathways only if we have most of the genes (currently 61 genes are missing)
        if html_kw is None:
            html_kw = {}
        create_html(
            data, fitter, fits, basedir, gene_dir, series_dir, scores_dir, correlations_dir=correlations_dir,
            use_correlations=use_correlations, link_to_correlation_plots=link_to_correlation_plots, 
            b_pathways=b_pathways, **html_kw
        )
