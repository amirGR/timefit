import setup
from os.path import join
from load_data import load_data
from plots import save_fits_and_create_html
from fitter import Fitter
from shapes.sigmoid import Sigmoid
from shapes.poly import Poly

data = load_data()
for n in [1,2,3]:
    shape = Poly(n)
    fitter = Fitter(shape,False,False)
    basedir = join(r'C:\temp\HTR', shape.cache_name()) 
    save_fits_and_create_html(data, fitter, basedir)
