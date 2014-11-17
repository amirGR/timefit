import setup
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigslope import Sigslope
from fitter import Fitter
from scalers import LogScaler
from dev_stages import dev_stages
from plots import save_figure
from sigmoid_change_distribution import get_bins, calc_change_distribution

fontsize = 30
xtick_fontsize = 30
ytick_fontsize = 30
equation_fontsize = 36

def plot_one_series(series, shape, theta, bin_centers, weights, yrange=None, show_title=False):
    x = series.ages
    y = series.single_expression    
    xmin, xmax = min(x), max(x)
    xmin = max(xmin,-2)

    fig = plt.figure()
    ax = fig.add_axes([0.08,0.15,0.85,0.8])

    # plot the data points
    ax.plot(x,y, 'ks', markersize=8)
    if yrange is None:
        ymin, ymax = ax.get_ylim()
    else:
        ymin, ymax = yrange

    # plot change distribution if provided
    width = bin_centers[1] - bin_centers[0]
    weights *= 0.9 * (ymax - ymin) / weights.max()
    ax.bar(bin_centers, weights, width=width, bottom=ymin, color='g', alpha=0.1)

    # mark birth time with a vertical line
    birth_age = series.age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    # draw the fit
    x_smooth,y_smooth = shape.high_res_preds(theta,x)
    ax.plot(x_smooth, y_smooth, '-', linewidth=3)
        
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    # title
    if show_title:
        ttl = '{}@{}, {} fit'.format(series.gene_name, series.region_name, shape)
        ax.set_title(ttl, fontsize=fontsize)

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
shape = Sigslope('sigslope80')
fitter = Fitter(shape, sigma_prior='normal')
bin_edges, bin_centers = get_bins(data)

GRs = [
    ('GLRA2','STC', (5, 12)), 
    ('ADCY6','IPC', (6, 8)), 
]

for g,r,yrange in GRs:
    print 'Doing {}@{}...'.format(g,r)
    series = data.get_one_series(g,r)
    theta,_,_,_ = fitter.fit(series.ages, series.single_expression)
    weights = calc_change_distribution(shape, theta, bin_edges)
    fig = plot_one_series(series, shape, theta, bin_centers, weights, yrange)
    save_figure(fig,'RP/fit-examples-{}-{}.png'.format(g,r), under_results=True)
