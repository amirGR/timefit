import setup
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from shapes.spline import Spline
from shapes.poly import Poly
from fitter import Fitter
from scalers import LogScaler
from dev_stages import dev_stages
from plots import save_figure

fontsize = 30
xtick_fontsize = 30

def plot_one_series(series, fitters, thetas, labels, yrange=None, show_title=False):
    x = series.ages
    y = series.expression    
    xmin, xmax = min(x), max(x)
    xmin = max(xmin,-2)

    fig = plt.figure()
    ax = fig.add_axes([0.12,0.15,0.85,0.8])

    # plot the data points
    ax.plot(x,y, 'ks', markersize=8)
    if yrange is None:
        ymin, ymax = ax.get_ylim()
    else:
        ymin, ymax = yrange

    # mark birth time with a vertical line
    birth_age = series.age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    # draw the fits
    for fitter,theta,label in zip(fitters,thetas,labels):
        x_smooth,y_smooth = fitter.shape.high_res_preds(theta,x)
        ax.plot(x_smooth, y_smooth, '-', linewidth=3, label=label)
    ax.legend(fontsize=fontsize, frameon=False, loc='best')  
        
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    # set the development stages as x labels
    ax.set_xlabel('age', fontsize=fontsize)
    stages = [stage.scaled(series.age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=xtick_fontsize, fontstretch='condensed', rotation=90)    

    # set y ticks (first and last only)
    ax.set_ylabel('expression level', fontsize=fontsize)
    ticks = ax.get_yticks()
    ticks = np.array([ticks[0], ticks[-1]])
    ax.set_yticks(ticks)
    ax.set_yticklabels(['{:g}'.format(t) for t in ticks], fontsize=fontsize)
    
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()
data = GeneData.load('both').scale_ages(age_scaler)

GRs = [
    ('HTR5A','S1C', (6, 10)), 
    ('HTR1A','V1C', (4, 10)), 
]
fitters = [
    Fitter(Sigmoid()), 
    Fitter(Sigmoid('sigmoid_wide'),sigma_prior='normal')
]
labels = ['no priors', 'semi-informative']

for g,r,yrange in GRs:
    print 'Doing {}@{}...'.format(g,r)
    series = data.get_one_series(g,r)
    thetas = []
    for fitter in fitters:
        theta,_,_ = fitter.fit(series.ages, series.expression)
        thetas.append(theta)
    fig = plot_one_series(series,fitters,thetas,labels,yrange)
    save_figure(fig,'RP/fit-examples-{}-{}.png'.format(g,r), under_results=True)            
    for theta,label in zip(thetas,labels):
        print '{}: {}'.format(label,theta)
        
