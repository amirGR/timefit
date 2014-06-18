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
ytick_fontsize = 30
equation_fontsize = 36

def plot_one_series(series, shapes, thetas, yrange=None, show_title=False):
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

    # mark birth time with a vertical line
    birth_age = series.age_scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    # draw the fits
    for shape,theta in zip(shapes,thetas):
        x_smooth,y_smooth = shape.high_res_preds(theta,x)
        ax.plot(x_smooth, y_smooth, '-', linewidth=3, label=shape.cache_name())
    ax.legend(fontsize=fontsize, frameon=False, loc='best')  
        
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

shapes = [Sigmoid('sigmoid_wide'), Poly(1,'poly1'), Poly(3,'poly3'), Spline()]
GRs = [
    ('ADRB1','A1C', (5, 8)), 
    ('GLRA2','STC', (5, 12)), 
    ('TUBA1A','V1C', (10, 14)),
]

for g,r,yrange in GRs:
    print 'Doing {}@{}...'.format(g,r)
    thetas = []
    for shape in shapes:
        series = data.get_one_series(g,r)
        sigma_prior = 'normal' if not isinstance(shape,Spline) else None
        fitter = Fitter(shape, sigma_prior=sigma_prior)
        theta,_,_ = fitter.fit(series.ages, series.single_expression)
        thetas.append(theta)
    fig = plot_one_series(series,shapes,thetas,yrange)
    save_figure(fig,'RP/fit-examples-{}-{}.png'.format(g,r), under_results=True)
