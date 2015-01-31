import argparse
import config as cfg
from load_data import GeneData
from scalers import allowed_scaler_names, build_scaler
from shapes.shape import get_shape_by_name, allowed_shape_names
from shapes.priors import get_allowed_priors
from fitter import Fitter
from dev_stages import PCW

class MyParser(argparse.ArgumentParser):  
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith('#'): # support comments
            return
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

dct_ages = {
    'postnatal' : 0,
    'EF3' : PCW(10),
}

known_datasets = ['kang2011', 'colantuoni2011', 'both', 'harris2009', 'somel2010-human', 'brainspan2014']

def get_common_parser(include_pathway=True):
    parser = MyParser(fromfile_prefix_chars='@')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="count", default=0, help="increase output verbosity")    
    group.add_argument("-q", "--quiet", action="count", default=0, help="decrease output verbosity")    
    parser.add_argument('--dataset', default='both', help='Default: both', choices=known_datasets)  
    if include_pathway:
        parser.add_argument('--pathway', default='serotonin', help='Default: serotonin') #, choices=['all'] + cfg.pathways.keys())
    parser.add_argument('--from_age', help='Use only data points with larger ages than this. Default: all ages', choices=dct_ages.keys())
    parser.add_argument('--scaling', help='What scaling to use for ages. Default: log', default='log', choices=allowed_scaler_names())
    parser.add_argument('--shuffle', help='Shuffle the y-values of the data', action='store_true')
    parser.add_argument('-s', '--shape', help='The shape to use for fitting. Default: sigslope', default='sigslope', choices=allowed_shape_names())
    parser.add_argument('--sigma_prior', help='Prior to use for 1/sigma when fitting. Default: normal', default = 'normal', choices=get_allowed_priors(is_sigma=True))
    parser.add_argument('--priors', help='Priors to use for theta when fitting. Default: sigslope80', default = 'sigslope80', choices=get_allowed_priors())
    return parser

def process_common_inputs(args):
    cfg.verbosity = args.verbose - args.quiet
    pathway = getattr(args, 'pathway', None)
    data = get_data_from_args(args.dataset, pathway, args.from_age, args.scaling, args.shuffle)
    fitter = get_fitter_from_args(args.shape, args.priors, args.sigma_prior)
    return data,fitter

def get_data_from_args(dataset, pathway, from_age, scaling, shuffle):
    data = GeneData.load(dataset).restrict_pathway(pathway)
    if from_age is not None:
        restriction_name = from_age
        from_age = dct_ages[from_age]
        data.restrict_ages(restriction_name,from_age=from_age)
    scaler = build_scaler(scaling,data)
    if scaler is not None:
        data.scale_ages(scaler)
    if shuffle:
        data.shuffle()
    return data

def get_fitter_from_args(shape, priors, sigma_prior):
    shape = get_shape_by_name(shape, priors)
    fitter = Fitter(shape, sigma_prior=sigma_prior)
    return fitter
    