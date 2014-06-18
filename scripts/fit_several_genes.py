import setup
from os.path import join
import matplotlib.pyplot as plt
from project_dirs import results_dir
from utils.misc import ensure_dir, disable_all_warnings
from command_line import get_common_parser, process_common_inputs
from plots import plot_several_series, save_figure

def fit_serveral_genes(series, fitter, loo, filename, b_show):
    if fitter is not None:
        theta, sigma, LOO_predictions = fitter.fit(series.ages, series.expression, loo=loo)
        fig = plot_several_series(series, fitter.shape, theta, LOO_predictions)
    else:
        fig = plot_several_series(series)
    if filename is None:
        ensure_dir(results_dir())
        filename = join(results_dir(), 'fits.png')
    print 'Saving figure to {}'.format(filename)
    save_figure(fig, filename)
    if b_show:
        plt.show(block=True)

if __name__ == '__main__':
    disable_all_warnings()   
    parser = get_common_parser(include_pathway=False)
    parser.add_argument('-g', '--genes', default='HTR1E HTR1A')
    parser.add_argument('-r', '--region', default='VFC')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--loo', help='Show LOO predictions', action='store_true')
    group.add_argument('--nofit', help='Only show the data points', action='store_true')
    parser.add_argument('--filename', help='Where to save the figure. Default: results/fit.png')
    parser.add_argument('--show', help='Show figure and wait before exiting', action='store_true')
    args = parser.parse_args()
    data, fitter = process_common_inputs(args)
    genes = args.genes.split()
    if args.nofit:
        fitter = None
    series = data.get_several_series(genes,args.region)
    #fit_serveral_genes(series, fitter, args.loo, args.filename, args.show)
