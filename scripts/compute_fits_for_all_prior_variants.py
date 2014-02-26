import setup
from load_data import load_data
from all_fits import get_all_fits
from fitter import Fitter
from shapes.sigmoid import Sigmoid

if __name__ == '__main__':
    data = load_data()
    shape = Sigmoid()
    for theta_prior in [True,False]:
        for sigma_prior in [True,False]:
            fitter = Fitter(shape, use_theta_prior=theta_prior, use_sigma_prior=sigma_prior)
            print """\
            
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
            """.format(fitter)
            fits = get_all_fits(data,fitter)
    
