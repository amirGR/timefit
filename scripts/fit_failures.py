import setup
import numpy as np
from scipy.optimize import minimize
from load_data import load_data
from fitter import Fitter
from shapes.sigmoid import Sigmoid

data = load_data()
shape = Sigmoid()
fitter = Fitter(shape,False,False)
series = data.get_one_series('MAOB','DFC')
x = series.ages
y = series.expression

#def E(P):
#    return fitter._Err(P,x,y)
#def E_grad(P):
#    return fitter._Err_grad(P,x,y)
#
#P0 = np.array(shape.get_theta_guess(x,y) + [1])
#
#res = minimize(E, P0, method='BFGS', jac=E_grad, tol=1E-5, callback=cb)
#print 'Final result={}'.format(res.fun)
