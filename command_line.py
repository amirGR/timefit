import argparse
import config as cfg
from load_data import GeneData
from scalers import allowed_scaler_names, build_scaler
from shapes.shape import get_shape_by_name, allowed_shape_names
from fitter import Fitter

def get_common_parser(include_pathway=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", action="count", help="increase output verbosity")    
    parser.add_argument('--dataset', default='kang2011', choices=['kang2011', 'colantuoni2011'])  
    if include_pathway:
        parser.add_argument('--pathway', default='serotonin', choices=cfg.pathways.keys())
    parser.add_argument('--postnatal', help='Use only postnatal data points', action='store_true')
    parser.add_argument('--scaling', help='What scaling to use for ages', default='log', choices=allowed_scaler_names())
    parser.add_argument('-s', '--shape', help='The shape to use for fitting', default='sigmoid', choices=allowed_shape_names())
    parser.add_argument('--priors', help='Use priors on theta and sigma', action='store_true')
    return parser
    
def process_common_inputs(args):
    cfg.verbosity = args.verbosity
    
    data = GeneData.load(args.dataset)
    data.restrict_pathway(getattr(args, 'pathway', 'all'))
    if args.postnatal:
        data.restrict_postnatal()
    if args.scaling is not None:
        scaler = build_scaler(args.scaling,data)
        data.scale_ages(scaler)

    shape = get_shape_by_name(args.shape)
    fitter = Fitter(shape, use_theta_prior=args.priors, use_sigma_prior=args.priors)

    return data,fitter
