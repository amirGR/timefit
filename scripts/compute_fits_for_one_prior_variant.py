import setup
from load_data import load_data
from all_fits import get_all_fits
from fitter import Fitter
from shapes.sigmoid import Sigmoid
import sys

def do_fits(theta,sigma):
    data = load_data()
    shape = Sigmoid()
    fitter = Fitter(shape, use_theta_prior=theta, use_sigma_prior=sigma)
    print """\
    
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
    """.format(fitter)
    get_all_fits(data,fitter)

if __name__ == '__main__':
    flags = sys.argv[1]
    assert len(flags) == 2
    theta = flags[0].upper() == 'T'
    sigma = flags[1].upper() == 'T'
    do_fits(theta, sigma)
            
