import setup
from os.path import join
import matplotlib.pyplot as plt
from project_dirs import results_dir
from utils.misc import ensure_dir, disable_all_warnings
from command_line import get_common_parser, process_common_inputs
from plots import plot_one_series, plot_gene, save_figure

def do_one_fit(series, fitter, loo, filename, b_show):
    if fitter is not None:
        theta, sigma, LOO_predictions,_ = fitter.fit(series.ages, series.single_expression, loo=loo)
        fig = plot_one_series(series, fitter.shape, theta, LOO_predictions)
    else:
        fig = plot_one_series(series)
    if filename is None:
        ensure_dir(results_dir())
        filename = join(results_dir(), 'fit.png')
    save_figure(fig, filename, print_filename=True)
    if b_show:
        plt.show(block=True)

def do_gene_fits(data, gene, fitter, filename, b_show):
    fig = plot_gene(data,gene)
    if filename is None:
        ensure_dir(results_dir())
        filename = join(results_dir(), 'fit.png')
    print 'Saving figure to {}'.format(filename)
    save_figure(fig, filename)
    if b_show:
        plt.show(block=True)

if __name__ == '__main__':
    disable_all_warnings()   
    parser = get_common_parser(include_pathway=False)
    parser.add_argument('-g', '--gene', default='HTR1E')
    parser.add_argument('-r', '--region', default='VFC')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--loo', help='Show LOO predictions', action='store_true')
    group.add_argument('--nofit', help='Only show the data points', action='store_true')
    parser.add_argument('--filename', help='Where to save the figure. Default: results/fit.png')
    parser.add_argument('--show', help='Show figure and wait before exiting', action='store_true')
    args = parser.parse_args()
    data, fitter = process_common_inputs(args)
    if args.nofit:
        fitter = None
    if args.region == 'all':
        do_gene_fits(data, args.gene, fitter,args.filename,args.show)
    else:
        series = data.get_one_series(args.gene,args.region)
        do_one_fit(series, fitter, args.loo, args.filename, args.show)
