import setup
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from project_dirs import resources_dir, results_dir
from utils.misc import disable_all_warnings
from command_line import get_common_parser, process_common_inputs
from all_fits import get_all_fits
from fit_score import loo_score
from plots import save_figure

def plot_comparison_scatter(R2_pairs, pathway, region_name):
    basic = np.array([b for b,m in R2_pairs])
    multi = np.array([m for b,m in R2_pairs])
    
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
    ax.set_title('$R^2$ gain for {}@{}'.format(pathway,region_name), fontsize=cfg.fontsize)
    ax.legend(fontsize=cfg.fontsize, frameon=False, loc='upper left')
    return fig

def plot_improvement_histogram(R2_pairs, pathway, region_name):
    basic = np.array([b for b,m in R2_pairs])
    multi = np.array([m for b,m in R2_pairs])
    delta = multi - basic
    
    fig = plt.figure()
    ax = fig.add_axes([0.12,0.12,0.8,0.72])
    ax.hist(delta, bins=20)
    ttl1 = r'$\Delta R^2$ distribution for {}@{}'.format(pathway,region_name)
    ttl2 = r'$\Delta R^2 = {:.2g} \pm {:.2g}$'.format(np.mean(basic), np.std(basic))
    ttl = '{}\n{}\n'.format(ttl1,ttl2)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_xlabel('$\Delta R^2$', fontsize=cfg.fontsize)
    ax.set_ylabel('number of gene-regions', fontsize=cfg.fontsize)   
    ax.tick_params(axis='both', labelsize=cfg.fontsize)
    return fig

def do_plots(R2_pairs, pathway, region):
    if region is None:
        region = 'all'

    # scatter
    fig = plot_comparison_scatter(R2_pairs,pathway,region)
    save_figure(fig, scatter_filename(pathway,region), b_close=True, under_results=True)
    
    # histogram
    fig = plot_improvement_histogram(R2_pairs,pathway,region)
    save_figure(fig, histogram_filename(pathway,region), b_close=True, under_results=True)

def scatter_filename(pathway,region):
    return 'multi-gene-prediction-improvement-{}-{}.png'.format(pathway,region)

def histogram_filename(pathway,region):
    return 'multi-gene-improvement-historgram{}-{}.png'.format(pathway,region)

def create_html(pathway, region_names):
    region_names = ['all'] + region_names
    rows = [(region, scatter_filename(pathway,region), histogram_filename(pathway,region)) for region in region_names]
    inline_image_size = '40%'
    
    from jinja2 import Template
    import shutil
    html = Template("""
<html>
<head>
    <link rel="stylesheet" type="text/css" href="multi-gene.css">
</head>
<body>
<H1>Multi-gene regression performance - {{pathway}} pathway</H1>
<P>
<table>
    <th>
        <td class="tableHeading">
            <b>R2 before/after</b>
        </td>
        <td class="tableHeading">
            <b>R2 difference</b>
        </td>
    </th>
    {% for region, scatter, histogram in rows %}
    <tr>
        <td>
            <b>{{region}}</b>
        </td>
        <td>
            <a href="{{scatter}}">
                <img src="{{scatter}}" height="{{inline_image_size}}">
            </a>
        </td>
        <td>
            <a href="{{histogram}}">
                <img src="{{histogram}}" height="{{inline_image_size}}">
            </a>
        </td>
    </tr>
    {% endfor %}
</table>
</P>

</body>
</html>    
""").render(**locals())
    with open(join(results_dir(),'multi-gene.html'), 'w') as f:
        f.write(html)
    
    shutil.copy(join(resources_dir(),'multi-gene.css'), results_dir())

def analyze_one_region(data, fitter, fits, pathway, region):
    print 'Analyzing region {}...'.format(region)
    series = data.get_several_series(data.gene_names,region)
    ds_fits = fits[data.get_dataset_for_region(region)]
    
    def cache(iy,ix):
        g = series.gene_names[iy]
        fit = ds_fits[(g,region)]
        if ix is None:
            return fit.theta
        else:
            theta,sigma = fit.LOO_fits[ix]
            return theta    
    x = series.ages
    y = series.expression
    multi_gene_preds,_ = fitter.fit_multiple_series_with_cache(x,y,cache)
    
    R2_pairs = []
    for i,g in enumerate(series.gene_names):
        y_real = y[:,i]
        y_basic = ds_fits[(g,region)].LOO_predictions
        y_multi_gene = multi_gene_preds[:,i]        
        basic_R2 = loo_score(y_real,y_basic)
        multi_gene_R2 = loo_score(y_real,y_multi_gene)
        R2_pairs.append( (basic_R2, multi_gene_R2) )
    return R2_pairs

if __name__ == '__main__':
    disable_all_warnings()
    parser = get_common_parser()
    args = parser.parse_args()
    pathway = args.pathway
    data, fitter = process_common_inputs(args)
    fits = get_all_fits(data, fitter, allow_new_computation=False)
    
    regions = data.region_names
    all_pairs = []
    for region in regions:
        region_pairs = analyze_one_region(data, fitter, fits, pathway, region)
        do_plots(region_pairs,pathway,region)
        all_pairs += region_pairs
    do_plots(all_pairs,pathway,region=None)
    create_html(pathway,regions)
