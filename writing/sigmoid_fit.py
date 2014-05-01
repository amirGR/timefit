import setup
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from scalers import LogScaler
from dev_stages import dev_stages
from fit_score import loo_score
from project_dirs import results_dir
from utils.misc import ensure_dir

fontsize = 30
xtick_fontsize = 30
ytick_fontsize = 30
equation_fontsize = 36
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

def plot_one_series(series, shape, theta, b_annotate=False, train_mask=None, test_preds=None):
    x = series.ages
    y = series.expression    
    xmin, xmax = min(x), max(x)
    xmin = max(xmin,-2)
    ymin, ymax = min(y), max(y)

    if train_mask is None:
        train_mask = ~np.isnan(x)
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.2,0.8,0.7])

    # plot the data points
    if not b_annotate:
        ax.plot(x[train_mask],y[train_mask], 'ks', markersize=8)

    if not b_annotate:
        # mark birth time with a vertical line
        birth_age = series.age_scaler.scale(0)
        ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')

    if theta is not None:
        # draw the overall fit
        x_smooth,y_smooth = shape.high_res_preds(theta,x)
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=3)
    
        # plot left out points and prediction error
        for xi,yi in zip(x[~train_mask],y[~train_mask]):
            y_loo = shape.f(theta,xi)
            ax.plot(xi,yi, 'rs', markersize=8)
            ax.plot([xi, xi], [yi, y_loo], '-', color='0.5')
            ax.plot(xi, y_loo, 'rx', markeredgewidth=2)
    
    if test_preds is not None:
        for xi,yi,y_loo in zip(x,y,test_preds):
            ax.plot([xi, xi], [yi, y_loo], '-', color='0.5')
            ax.plot(xi, y_loo, 'x', color='0.5', markeredgewidth=2)
        score = loo_score(y,test_preds)
        txt = "$R^2 = {:.2g}$".format(score)
        ax.text(0.02,0.8,txt,fontsize=equation_fontsize,transform=ax.transAxes)

    if b_annotate:        
        # annotate sigmoid parameters
        arrow_color = 'green'
        a,h,mu,w = theta
        
        # onset
        y_onset = shape.f(theta, mu)
        ax.plot([mu,mu],[ymin,y_onset],'g--',linewidth=2)
        ax.text(mu+0.05,y_onset-0.5,'onset', fontsize=fontsize, horizontalalignment='left')
    
        # baseline
        ax.plot([xmin,xmax],[a, a],'g--',linewidth=2)
        ax.text(mu+1.5,a+0.05,'baseline', fontsize=fontsize, verticalalignment='bottom')
        
        # slope
        dx = 0.5
        dy = dx*h/(4*w) # that's df/dx at x=mu
        ax.plot([mu-dx,mu+dx],[y_onset-dy+0.05, y_onset+dy+0.05],'g--',linewidth=2)
        ax.text(mu-0.5,y_onset+1,'slope', fontsize=fontsize, horizontalalignment='right')
        ax.arrow(mu-0.45,y_onset+0.95,0.65,-0.65, length_includes_head=True, width=0.005, facecolor=arrow_color)
        
        #height
        xpos = mu + 4*w
        ax.text(xpos+0.05,y_onset,'height', fontsize=fontsize, verticalalignment='center')
        ax.arrow(xpos,y_onset,0,h*0.45, length_includes_head=True, width=0.005, facecolor=arrow_color)
        ax.arrow(xpos,y_onset,0,-h*0.45, length_includes_head=True, width=0.005, facecolor=arrow_color)

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    
    # title
    ttl = '{}@{}, {} fit'.format(series.gene_name, series.region_name, shape)
    ax.set_title(ttl, fontsize=fontsize)

    # set the development stages as x labels
    ax.set_xlabel('age', fontsize=fontsize)
    stages = [stage.scaled(series.age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=xtick_fontsize, fontstretch='condensed', rotation=90)    

    # set y ticks (first and last only)
    ax.set_ylabel('expression level (log scale)', fontsize=fontsize)
    ticks = ax.get_yticks()
    ticks = np.array([ticks[0], ticks[-1]])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=fontsize)
            
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()

data = GeneData.load('both').scale_ages(age_scaler)
series = data.get_one_series('HTR1D','STR')
x = series.ages
y = series.expression

shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')

def basic_fit():
    print 'Drawing basic fit...'
    theta,_,_ = fitter.fit(x,y)
    fig = plot_one_series(series,shape,theta)
    save_figure(fig,'methods-1-basic-fit.png')

def annotate_parameters():
    print 'Drawing fit with parameters...'
    theta,_,_ = fitter.fit(x,y)
    fig = plot_one_series(series,shape,theta, b_annotate=True)
    save_figure(fig,'methods-2-sigmoid-params.png')

def show_loo_prediction():
    print 'Drawing LOO prediction and error...'
    iLOO = 18
    train_mask = np.arange(len(x)) != iLOO
    x_train = x[train_mask]
    y_train = y[train_mask]
    theta,_,_ = fitter.fit(x_train,y_train)
    fig = plot_one_series(series,shape,theta,train_mask=train_mask)
    save_figure(fig,'methods-3-LOO-prediction.png')
  
def show_loo_score():
    print 'Drawing LOO prediction for all points and R2 score...'
    theta,_,test_preds = fitter.fit(x,y,loo=True)
    fig = plot_one_series(series,shape,theta=None,test_preds=test_preds)
    save_figure(fig,'methods-4-R2-score.png')

#basic_fit()
annotate_parameters()
#show_loo_prediction()
#show_loo_score()