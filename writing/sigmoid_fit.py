import setup
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from load_data import GeneData
from shapes.sigmoid import Sigmoid
from fitter import Fitter
from scalers import LogScaler
from dev_stages import dev_stages
from fit_score import loo_score
from plots import save_figure

fontsize = 30
xtick_fontsize = 30
ytick_fontsize = 30
equation_fontsize = 36

def plot_one_series(series, shape, theta, yrange=None, b_annotate=False, train_mask=None, test_preds=None, show_title=False):
    x = series.ages
    y = series.expression    
    xmin, xmax = min(x), max(x)
    xmin = max(xmin,-2)

    if train_mask is None:
        train_mask = ~np.isnan(x)
    
    fig = plt.figure()
    ax = fig.add_axes([0.08,0.15,0.85,0.8])

    # plot the data points
    if not b_annotate:
        ax.plot(x[train_mask],y[train_mask], 'ks', markersize=8)
    if yrange is None:
        ymin, ymax = ax.get_ylim()
    else:
        ymin, ymax = yrange

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
series = data.get_one_series('HTR1D','STR')
x = series.ages
y = series.expression
yrange = (3,8)

shape = Sigmoid(priors='sigmoid_wide')
fitter = Fitter(shape, sigma_prior='normal')

def basic_fit():
    print 'Drawing basic fit...'
    theta,_,_ = fitter.fit(x,y)
    fig = plot_one_series(series,shape,theta,yrange)
    save_figure(fig,'RP/methods-1-basic-fit.png', under_results=True)

def annotate_parameters():
    print 'Drawing fit with parameters...'
    theta,_,_ = fitter.fit(x,y)
    fig = plot_one_series(series,shape,theta, yrange, b_annotate=True)
    save_figure(fig,'RP/methods-2-sigmoid-params.png', under_results=True)

def show_loo_prediction():
    print 'Drawing LOO prediction and error...'
    iLOO = 18
    train_mask = np.arange(len(x)) != iLOO
    x_train = x[train_mask]
    y_train = y[train_mask]
    theta,_,_ = fitter.fit(x_train,y_train)
    fig = plot_one_series(series,shape,theta,yrange,train_mask=train_mask)
    save_figure(fig,'RP/methods-3-LOO-prediction.png', under_results=True)
  
def show_loo_score():
    print 'Drawing LOO prediction for all points and R2 score...'
    theta,_,test_preds = fitter.fit(x,y,loo=True)
    fig = plot_one_series(series,shape,theta=None,yrange=yrange,test_preds=test_preds)
    save_figure(fig,'RP/methods-4-R2-score.png', under_results=True)

basic_fit()
annotate_parameters()
show_loo_prediction()
show_loo_score()