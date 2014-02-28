import setup
from load_data import load_data
from all_fits import get_all_fits
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from shapes.poly import Poly
import sys
import re

def get_shape(shape_name):
    if shape_name == 'sigmoid':
        return Sigmoid()
    elif shape_name.startswith('poly'):
        m = re.match('poly(\d)',shape_name)
        assert m, 'Illegal polynomial shape name'
        degree = int(m.group(1))
        return Poly(degree)
    else:
        raise Exception('Unknown shape: {}'.format(shape_name))

def do_fits(data, shape, theta, sigma):
    fitter = Fitter(shape, use_theta_prior=theta, use_sigma_prior=sigma)
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
        print 'Usage: {} shape TT TF ...'.format(sys.argv[0])
        return
    shape = get_shape(sys.argv[1])
    data = load_data()
    for flags in sys.argv[2:]:
        assert len(flags) == 2
        theta = flags[0].upper() == 'T'
        sigma = flags[1].upper() == 'T'
        do_fits(data, shape, theta, sigma)
    
if __name__ == '__main__':
    main()