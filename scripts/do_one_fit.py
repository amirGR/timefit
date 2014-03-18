import setup
import argparse
from os.path import join
import config as cfg
from project_dirs import results_dir
import utils
from load_data import GeneData
from shapes.shape import get_shape_by_name
from fitter import Fitter
from scalers import LogScaler
from plots import plot_one_series, save_figure

def do_one_fit(series, shape, use_priors, loo, filename):
    fitter = Fitter(shape, use_theta_prior=use_priors, use_sigma_prior=use_priors)
    theta, sigma, LOO_predictions = fitter.fit(series.ages, series.expression, loo=loo)
    fig = plot_one_series(series, shape, theta, LOO_predictions)
    if filename is None:
        utils.ensure_dir(results_dir())
        filename = join(results_dir(), 'fit.png')
    print 'Saving figure to {}'.format(filename)
    save_figure(fig, filename, b_close=True)

if __name__ == '__main__':
    utils.disable_all_warnings()   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='kang2011', choices=['kang2011', 'colantuoni2011'])  
    parser.add_argument('-g', '--gene', default='HTR1E')
    parser.add_argument('-r', '--region', default='VFC')
    parser.add_argument('-s', '--shape', default='sigmoid')
    parser.add_argument('--priors', help='Use priors on theta and sigma', action='store_true')
    parser.add_argument('--postnatal', help='Use only postnatal data points', action='store_true')
    parser.add_argument('--nologx', dest='logx', help='Use linear scale for ages', action='store_false')
    parser.add_argument('--loo', help='Show LOO predictions', action='store_true')
    parser.add_argument('--filename', help='Where to save the figure')
    args = parser.parse_args()
    
    shape = get_shape_by_name(args.shape)
    data = GeneData.load(args.dataset)
    if args.postnatal:
        data.restrict_postnatal()
    if args.logx:
        data.scale_ages(LogScaler(cfg.kang_log_scale_x0))

    series = data.get_one_series(args.gene,args.region)
    do_one_fit(series, shape, args.priors, args.loo, args.filename)
