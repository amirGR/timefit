import argparse
import config as cfg
from load_data import GeneData
from scalers import allowed_scaler_names, build_scaler
from shapes.shape import get_shape_by_name, allowed_shape_names
from fitter import Fitter
from shapes.standard_priors import sigma_priors, theta_priors

def get_common_parser(include_pathway=True):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="count", default=0, help="increase output verbosity")    
    group.add_argument("-q", "--quiet", action="count", default=0, help="decrease output verbosity")    
    parser.add_argument('--dataset', default='kang2011', help='Default: kang2011', choices=['kang2011', 'colantuoni2011'])  
    if include_pathway:
        parser.add_argument('--pathway', default='serotonin', help='Default: serotonin', choices=['all'] + cfg.pathways.keys())
    parser.add_argument('--postnatal', help='Use only postnatal data points', action='store_true')
    parser.add_argument('--scaling', help='What scaling to use for ages. Default: none', choices=allowed_scaler_names())
    parser.add_argument('-s', '--shape', help='The shape to use for fitting. Default: sigmoid', default='sigmoid', choices=allowed_shape_names())
    parser.add_argument('--sigma_prior', metavar='PRIOR', help='Which priors to use for 1/sigma when fitting. Default: None. Use filename or one of {}'.format(sorted(sigma_priors.keys())))
    parser.add_argument('--priors', metavar='PRIORS', help='Which priors to use for theta when fitting. Default: None. Use filename or one of {}'.format(sorted(theta_priors.keys())))
    return parser
    
def process_common_inputs(args):
    cfg.verbosity = args.verbose - args.quiet
    
    data = GeneData.load(args.dataset)
    data.restrict_pathway(getattr(args, 'pathway', 'all'))
    if args.postnatal:
        data.restrict_postnatal()
    if args.scaling is not None:
        scaler = build_scaler(args.scaling,data)
        data.scale_ages(scaler)

    shape = get_shape_by_name(args.shape, args.priors)
    fitter = Fitter(shape, sigma_prior=args.sigma_prior)

    return data,fitter
