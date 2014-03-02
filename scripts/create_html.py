import setup
from os.path import join
from load_data import load_data
from plots import save_fits_and_create_html
from fitter import Fitter
from shapes.sigmoid import Sigmoid

data = load_data()
for theta_prior in [False,True]:
    fitter = Fitter(Sigmoid(),theta_prior,False)
    basedir = join(r'C:\temp\HTR', 'theta-{}'.format(theta_prior)) 
    save_fits_and_create_html(data, fitter, basedir)
