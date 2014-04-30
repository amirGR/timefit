import setup
from os.path import join
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
from project_dirs import results_dir
from utils.misc import ensure_dir

fontsize = 24
xtick_fontsize = 24
default_figure_size_x = 18.5
default_figure_size_y = 10.5
default_figure_facecolor = 0.85 * np.ones(3)
default_figure_dpi = 100

def save_figure(fig, filename, b_close=False):
    dirname = join(results_dir(),'RP')
    ensure_dir(dirname)
    filename = join(dirname,filename)
    print 'Saving figure to {}'.format(filename)
    fig.set_size_inches(default_figure_size_x, default_figure_size_y)
    fig.savefig(filename, facecolor=default_figure_facecolor, dpi=default_figure_dpi)
    if b_close:
        plt.close(fig)

def plot_one_series(series, shapes, thetas):
    x = series.ages
    y = series.expression    
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.15,0.8,0.75])

    # plot the data points
    ax.plot(x,y, 'ks', markersize=8)

    ax.set_ylabel('expression level (log scale)', fontsize=fontsize)
    ax.set_xlabel('age', fontsize=fontsize)
    ttl = '{}@{}'.format(series.gene_name, series.region_name)
    ax.set_title(ttl, fontsize=fontsize)

    # set the development stages as x labels
    stages = [stage.scaled(series.age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=xtick_fontsize, fontstretch='condensed', rotation=90)    

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
    
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()

data = GeneData.load('both').scale_ages(age_scaler)

shapes = [Sigmoid('sigmoid_wide'), Poly(1,'poly1'), Poly(3,'poly3'), Spline()]
GRs = [
    ('ADRB1','A1C'), 
    ('ADRB2','S1C'), 
    ('CREBBP','PFC'), 
    ('GLRA2','STC'), 
    ('TUBA1A','V1C'),
]

for g,r in GRs:
    print 'Doing {}@{}...'.format(g,r)
    thetas = []
    for shape in shapes:
        series = data.get_one_series(g,r)
        sigma_prior = 'normal' if not isinstance(shape,Spline) else None
        fitter = Fitter(shape, sigma_prior=sigma_prior)
        theta,_,_ = fitter.fit(series.ages, series.expression)
        thetas.append(theta)
    fig = plot_one_series(series,shapes,thetas)
    save_figure(fig,'fit-examples-{}-{}.png'.format(g,r))
