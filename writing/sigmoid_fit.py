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

fontsize = 24
xtick_fontsize = 18
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
    ymin, ymax = min(y), max(y)

    if train_mask is None:
        train_mask = ~np.isnan(x)
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.15,0.8,0.75])

    # plot the data points
    if not b_annotate:
        ax.plot(x[train_mask],y[train_mask], 'ks', markersize=8)

    ax.set_ylabel('expression level (log scale)', fontsize=fontsize)
    ax.set_xlabel('age', fontsize=fontsize)
    ttl = '{}@{}, {} fit'.format(series.gene_name, series.region_name, shape)
    ax.set_title(ttl, fontsize=fontsize)

    # remove y ticks
    ax.set_yticks([])    
    
    # set the development stages as x labels
    stages = [stage.scaled(series.age_scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=xtick_fontsize, fontstretch='condensed', rotation=90)    

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
        latex_r2 = r"1 - \frac{\sum_{i=1}^n{(y_i - \hat y_i)^2}}{\sum_{i=1}^n{(y_i-\bar y)^2} }"
        txt = "$R^2 = {} = {:.2g}$".format(latex_r2,score)
        ax.text(0.02,0.8,txt,fontsize=fontsize,transform=ax.transAxes)

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
        
        # width
        ax.text(mu,a+h+0.1,'width', fontsize=fontsize, horizontalalignment='center')
        ax.arrow(mu,a+h,w/2,0, length_includes_head=True, width=0.005, facecolor=arrow_color)
        ax.arrow(mu,a+h,-w/2,0, length_includes_head=True, width=0.005, facecolor=arrow_color)
        
        #height
        xpos = mu + 4*w
        ax.text(xpos+0.05,y_onset,'height', fontsize=fontsize, verticalalignment='center')
        ax.arrow(xpos,y_onset,0,h*0.45, length_includes_head=True, width=0.005, facecolor=arrow_color)
        ax.arrow(xpos,y_onset,0,-h*0.45, length_includes_head=True, width=0.005, facecolor=arrow_color)
        
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    
    return fig

cfg.verbosity = 1
age_scaler = LogScaler()

data = GeneData.load('both').scale_ages(age_scaler)
series = data.get_one_series('HTR1D','STR')
x = series.ages
y = series.expression

shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')

theta,_,test_preds = fitter.fit(x,y,loo=True)

# the basic fit
fig = plot_one_series(series,shape,theta)
save_figure(fig,'methods-1-basic-fit.png')

# annotate sigmoid parameters
fig = plot_one_series(series,shape,theta, b_annotate=True)
save_figure(fig,'methods-2-sigmoid-params.png')

# show LOO prediction and error for one point
iLOO = 18
train_mask = np.arange(len(x)) != iLOO
x_train = x[train_mask]
y_train = y[train_mask]
theta,_,_ = fitter.fit(x_train,y_train)
fig = plot_one_series(series,shape,theta,train_mask=train_mask)
save_figure(fig,'methods-3-LOO-prediction.png')

# show LOO predictions and errors for all points
fig = plot_one_series(series,shape,theta=None,test_preds=test_preds)
save_figure(fig,'methods-4-R2-score.png')