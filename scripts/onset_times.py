import setup
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from all_fits import get_all_fits
from scalers import LogScaler
from dev_stages import dev_stages

cfg.verbosity = 1
age_scaler = LogScaler()

def get_fits():
    data = GeneData.load('both').restrict_pathway('17pathways').scale_ages(age_scaler)
    shape = Sigmoid(priors='sigmoid_wide')
    fitter = Fitter(shape, sigma_prior='normal')
    fits = get_all_fits(data, fitter)
    return fits
    
def get_fit_param(fits, getter, R2_threshold=None):
    lst = []
    for dsfits in fits.itervalues():
        for fit in dsfits.itervalues():
            if R2_threshold is None or fit.LOO_score > R2_threshold:
                val = getter(fit)
                if val is not None:
                    lst.append(val)
    return lst

def plot_onset_times(onset_times, R2_threshold):
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    low = min(stage.from_age for stage in stages)
    high = max(stage.to_age for stage in stages)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(onset_times, 50, range=(low,high))
    ttl = 'Histogram of onset times for 17 pathways (R2 threshold={})'.format(R2_threshold)
    ax.set_title(ttl, fontsize=cfg.fontsize)
    ax.set_xlabel('onset time', fontsize=cfg.fontsize)
    ax.set_ylabel('count', fontsize=cfg.fontsize)    

    # set the development stages as x labels
    stages = [stage.scaled(age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=cfg.xtick_fontsize, fontstretch='condensed', rotation=90)    
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

if __name__ == '__main__':
    R2_threshold = 0.5
    fits = get_fits()
    def get_onset_time(fit):
        if fit.theta is None:
            return None
        a,h,mu,w = fit.theta
        return mu
    onset_times = get_fit_param(fits, get_onset_time, R2_threshold=R2_threshold)
    plot_onset_times(onset_times,R2_threshold=R2_threshold)