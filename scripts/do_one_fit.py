import setup
from os.path import join
from project_dirs import results_dir
import utils
from command_line import get_common_parser, process_common_inputs
from plots import plot_one_series, save_figure

def do_one_fit(series, fitter, loo, filename):
    theta, sigma, LOO_predictions = fitter.fit(series.ages, series.expression, loo=loo)
    fig = plot_one_series(series, fitter.shape, theta, LOO_predictions)
    if filename is None:
        utils.ensure_dir(results_dir())
        filename = join(results_dir(), 'fit.png')
    print 'Saving figure to {}'.format(filename)
    save_figure(fig, filename, b_close=True)

if __name__ == '__main__':
    utils.disable_all_warnings()   
    parser = get_common_parser(include_pathway=False)
    parser.add_argument('-g', '--gene', default='HTR1E')
    parser.add_argument('-r', '--region', default='VFC')
    parser.add_argument('--loo', help='Show LOO predictions', action='store_true')
    parser.add_argument('--filename', help='Where to save the figure')
    args = parser.parse_args()
    data, fitter = process_common_inputs(args)    
    series = data.get_one_series(args.gene,args.region)
    do_one_fit(series, fitter, args.loo, args.filename)
