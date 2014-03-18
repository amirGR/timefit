import setup
import argparse
from os.path import join
import config as cfg
import utils
from load_data import load_data
from all_fits import get_all_fits
from fitter import Fitter
from shapes.shape import get_shape_by_name
from plots import save_fits_and_create_html

def do_fits(data, shape, use_priors, html_dir):
    fitter = Fitter(shape, use_theta_prior=use_priors, use_sigma_prior=use_priors)
    print """
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
""".format(fitter)
    get_all_fits(data,fitter)    
    if html_dir is not None:
        basedir = join(html_dir, shape.cache_name()) 
        print """
==============================================================================================
==============================================================================================
==== Writing HTML to {}
==============================================================================================
==============================================================================================
""".format(basedir)
        save_fits_and_create_html(data, fitter, basedir)

if __name__ == '__main__':
    utils.disable_all_warnings()
    parser = argparse.ArgumentParser()
    parser.add_argument('--priors', help='Use priors on theta and sigma', action='store_true')
    parser.add_argument('--dataset', default='kang2011', choices=['kang2011', 'colantuoni2011'])  
    parser.add_argument('--pathway', default='serotonin', choices=cfg.pathways.keys())
    parser.add_argument('--html', metavar='HTMLDIR', help='Create html for the fits under HTMLDIR')
    parser.add_argument("shape", nargs='+')
    args = parser.parse_args()
    
    shapes = [get_shape_by_name(name) for name in args.shape]
    data = load_data(pathway=args.pathway, dataset=args.dataset)
    for shape in shapes:
        do_fits(data, shape, args.priors, args.html)
