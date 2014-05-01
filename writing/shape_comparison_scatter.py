import setup
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from shapes.poly import Poly
from shapes.spline import Spline
from fitter import Fitter
from all_fits import get_all_fits, iterate_fits
from scalers import LogScaler
from project_dirs import results_dir
from utils.misc import ensure_dir

fontsize = 30
xtick_fontsize = 30
ytick_fontsize = 30
equation_fontsize = 36
default_figure_size_x = 12.5
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

def plot_comparison_scatter(data, shape1, fits1, shape2, fits2):
    pairs = [(f1.LOO_score, f2.LOO_score) for dsname,g,r,f1,f2 in iterate_fits(fits1,fits2)]
    scores1,scores2 = zip(*pairs)
    
    fig = plt.figure()
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.scatter(scores1, scores2, alpha=0.3)
    ax.plot([-1, 1], [-1, 1],'k--')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ticks = [-1,1]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=fontsize)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=fontsize)
    ax.set_xlabel('R2 for {}'.format(shape1), fontsize=fontsize)
    ax.set_ylabel('R2 for {}'.format(shape2), fontsize=fontsize)
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()
pathway = '17full'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)

sigmoid = Sigmoid(priors='sigmoid_wide')
spline = Spline()
poly1 = Poly(1,priors='poly1')
poly2 = Poly(2,priors='poly2')
poly3 = Poly(3,priors='poly3')
shapes = [sigmoid, spline, poly1, poly2, poly3]

fitters = [Fitter(shape, sigma_prior='normal' if not shape.has_special_fitting() else None) for shape in shapes]
fits = [get_all_fits(data,fitter,allow_new_computation=False) for fitter in fitters]

for i in xrange(1,len(shapes)):
    fig = plot_comparison_scatter(data,shapes[0],fits[0],shapes[i],fits[i])
    save_figure(fig,'scatter-{}-{}.png'.format(shapes[0],shapes[i]))
