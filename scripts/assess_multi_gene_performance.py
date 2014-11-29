import setup
from os.path import join
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from project_dirs import resources_dir, results_dir
from utils.misc import disable_all_warnings, ensure_dir
from command_line import get_common_parser, process_common_inputs
from load_data import load_17_pathways_breakdown, get_17_pathways_short_names
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
    ttl2 = r'$\Delta R^2 = {:.2g} \pm {:.2g}$'.format(np.mean(delta), np.std(delta))
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
    filename = join(pathway,scatter_filename(pathway,region))
    save_figure(fig, filename, b_close=True, under_results=True)
    
    # histogram
    fig = plot_improvement_histogram(R2_pairs,pathway,region)
    filename = join(pathway,histogram_filename(pathway,region))
    save_figure(fig, filename, b_close=True, under_results=True)

def scatter_filename(pathway,region):
    return 'multi-gene-prediction-improvement-{}-{}.png'.format(pathway,region)

def histogram_filename(pathway,region):
    return 'multi-gene-improvement-historgram{}-{}.png'.format(pathway,region)

def create_html(pathway, region_names, gene_names):
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
<P style="font-size:x-large">
This pathway contains <b>{{gene_names|length}}</b> genes
</P>
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
    dirname = join(results_dir(),pathway)
    ensure_dir(dirname)
    with open(join(dirname,'multi-gene.html'), 'w') as f:
        f.write(html)
    
    shutil.copy(join(resources_dir(),'multi-gene.css'), join(results_dir(),pathway))

def analyze_one_region(data, fitter, fits, region):
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
    _,_,multi_gene_preds,_ = fitter.fit_multiple_series_with_cache(x,y,cache)
    
    R2_pairs = []
    for i,g in enumerate(series.gene_names):
        y_real = y[:,i]
        y_basic = ds_fits[(g,region)].LOO_predictions
        y_multi_gene = multi_gene_preds[:,i]        
        basic_R2 = loo_score(y_real,y_basic)
        multi_gene_R2 = loo_score(y_real,y_multi_gene)
        if basic_R2 < -1 or multi_gene_R2 < -1:
            continue
        R2_pairs.append( (basic_R2, multi_gene_R2) )
    return R2_pairs

def analyze_pathway(pathway, gene_names, regions, data, fitter, fits, html_only=False):
    print 80 * '='
    print 'Analyzing pathway {}'.format(pathway)
    print 80 * '='
    if not gene_names:
        print 'No genes in pathway {}. Skipping this pathway.'.format(pathway)
        return
    create_html(pathway,regions,gene_names)
    if html_only:
        return
    pathway_data = deepcopy(data)
    pathway_data.restrict_pathway(pathway,ad_hoc_genes=gene_names)
    all_pairs = []
    for region in regions:
        try:
            region_pairs = analyze_one_region(pathway_data, fitter, fits, region)
        except Exception, e:
            print 'ERROR: Could not analyze region {}. error={}'.format(region,e)
            continue
        do_plots(region_pairs,pathway,region)
        all_pairs += region_pairs
    do_plots(all_pairs,pathway,region=None)
    
if __name__ == '__main__':
    disable_all_warnings()
    
    # load data and cache for 17full, then analyze a subset
    parser = get_common_parser()
    parser.add_argument('--html_only', help='Just create the html file(s) without doing the fits', action='store_true')
    #args = parser.parse_args(['@compute_fits.args', '--pathway', 'cannabinoids'])
    args = parser.parse_args()
    data, fitter = process_common_inputs(args)
    if args.html_only:
        fits = None # don't waste time loading the fits. we're not going to use them anyway.
    else:
        fits = get_all_fits(data, fitter, allow_new_computation=False)
    
    regions = data.region_names
    #regions = ['DFC'] # debugging

    if args.pathway == '17full':
        dct_pathways = load_17_pathways_breakdown(b_unique=True, short_names=True)
        pathways = dct_pathways.keys()
        #pathways = ['serotonin'] # debugging
        for pathway in pathways:
            gene_names = dct_pathways[pathway]
            analyze_pathway(pathway, gene_names, regions, data, fitter, fits, args.html_only)
    else:
        analyze_pathway(args.pathway, data.gene_names, regions, data, fitter, fits, args.html_only)
     
