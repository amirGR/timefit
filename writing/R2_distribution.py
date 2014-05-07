import setup
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
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
default_figure_size_x = 18.5
default_figure_size_x_square = 12.5
default_figure_size_y = 10.5
default_figure_facecolor = 0.85 * np.ones(3)
default_figure_dpi = 100

def save_figure(fig, filename, b_square=False, b_close=False, show_frame=True):
    dirname = join(results_dir(),'RP')
    ensure_dir(dirname)
    filename = join(dirname,filename)
    print 'Saving figure to {}'.format(filename)
    figure_size_x = default_figure_size_x_square if b_square else default_figure_size_x
    fig.set_size_inches(figure_size_x, default_figure_size_y)
    if show_frame:
        facecolor = default_figure_facecolor
    else:
        facecolor = 'white'
    fig.savefig(filename, facecolor=facecolor, dpi=default_figure_dpi)
    if b_close:
        plt.close(fig)

def plot_score_distribution(fits):
    LOO_R2 = np.array([fit.LOO_score for dsname,g,r,fit in iterate_fits(fits)])
    low,high = -1, 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(LOO_R2, 50, range=(low,high), normed=True)
    ax.set_xlabel('test set $R^2$', fontsize=fontsize)
    ax.set_ylabel('probability density', fontsize=fontsize)   
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()
pathway = '17pathways'
data = GeneData.load('both').restrict_pathway(pathway).scale_ages(age_scaler)

fitter = Fitter(Spline())
fits = get_all_fits(data,fitter)

fig = plot_score_distribution(fits)
save_figure(fig,'R2-distribution-{}.png'.format(data.pathway), show_frame=False)
