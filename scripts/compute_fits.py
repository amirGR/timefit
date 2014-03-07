import setup
from load_data import load_data
from all_fits import get_all_fits
from fitter import Fitter
from shapes.shape import get_shape_by_name
import sys

def do_fits(data, shape, use_priors):
    fitter = Fitter(shape, use_theta_prior=use_priors, use_sigma_prior=use_priors)
    print """\
    
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
    """.format(fitter)
    get_all_fits(data,fitter)

def main():
    if len(sys.argv) == 1:
        print 'Usage: {} [use-priors] shape1 shape2...'.format(sys.argv[0])
        return
    shape_names = sys.argv[1:]
    if shape_names[0].lower() == 'use-priors':
        use_priors = True
        shape_names = shape_names[1:]
    else:
        use_priors = False
    shapes = [get_shape_by_name(name) for name in shape_names]
    data = load_data()
    for shape in shapes:
        do_fits(data, shape, use_priors)
    
if __name__ == '__main__':
    main()