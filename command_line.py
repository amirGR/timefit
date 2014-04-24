import argparse
import config as cfg
from load_data import GeneData
from scalers import allowed_scaler_names, build_scaler
from shapes.shape import get_shape_by_name, allowed_shape_names
from shapes.priors import get_allowed_priors
from fitter import Fitter

class MyParser(argparse.ArgumentParser):  
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith('#'): # support comments
            return
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

def get_common_parser(include_pathway=True):
    parser = MyParser(fromfile_prefix_chars='@')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="count", default=0, help="increase output verbosity")    
    group.add_argument("-q", "--quiet", action="count", default=0, help="decrease output verbosity")    
    parser.add_argument('--dataset', default='both', help='Default: both', choices=['kang2011', 'colantuoni2011', 'both'])  
    if include_pathway:
        parser.add_argument('--pathway', default='serotonin', help='Default: serotonin', choices=['all'] + cfg.pathways.keys())
    parser.add_argument('--postnatal', help='Use only postnatal data points', action='store_true')
    parser.add_argument('--scaling', help='What scaling to use for ages. Default: none', choices=allowed_scaler_names())
    parser.add_argument('-s', '--shape', help='The shape to use for fitting. Default: sigmoid', default='sigmoid', choices=allowed_shape_names())
    parser.add_argument('--sigma_prior', help='Prior to use for 1/sigma when fitting. Default: None', choices=get_allowed_priors(is_sigma=True))
    parser.add_argument('--priors', help='Priors to use for theta when fitting. Default: None', choices=get_allowed_priors())
    return parser

def process_common_inputs(args):
    cfg.verbosity = args.verbose - args.quiet
    pathway = getattr(args, 'pathway', None)
    data = get_data_from_args(args.dataset, pathway, args.postnatal, args.scaling)
    fitter = get_fitter_from_args(args.shape, args.priors, args.sigma_prior)
    return data,fitter

def get_data_from_args(dataset, pathway, postnatal, scaling):
    data = GeneData.load(dataset).restrict_pathway(pathway).restrict_postnatal(postnatal)
    if scaling is not None:
        scaler = build_scaler(scaling,data)
        data.scale_ages(scaler)
    return data

def get_fitter_from_args(shape, priors, sigma_prior):
    shape = get_shape_by_name(shape, priors)
    fitter = Fitter(shape, sigma_prior=sigma_prior)
    return fitter
    