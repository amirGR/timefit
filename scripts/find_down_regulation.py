import setup
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits, iterate_fits
from scalers import LogScaler

cfg.verbosity = 1
age_scaler = LogScaler()

def get_fits():
    data = GeneData.load('both').restrict_pathway('17pathways').scale_ages(age_scaler)
    shape = Sigmoid(priors='sigmoid_wide')
    fitter = Fitter(shape, sigma_prior='normal')
    fits = get_all_fits(data, fitter)
    return fits
    
def main():
    fits = get_fits()
    def cond(fit):
        a,h,mu,w = fit.theta
        if h*w > 0:
            return False
        return abs(w) < 0.5
    return [(g,r) for dsname,g,r,fit in iterate_fits(fits,R2_threshold=0.5,return_keys=True) if cond(fit)]

if __name__ == '__main__':
    res = main()