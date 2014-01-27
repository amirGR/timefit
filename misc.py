# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:08:14 2014

@author: ronnie
"""

def draw_with_fit(series, cv=True):
    x = series.ages
    y = series.expression
    P = fit_sigmoid_simple(x,y)
    fit = sigmoid(P[:-1],series.ages)
    fit_label = 'Simple fit ({}={:.3f})'.format(cfg.score_type, cfg.score(y,fit))
    fits = {fit_label : fit}
    if cv:
        preds = fit_sigmoid_loo(x,y)
        loo_label = 'LOO predictions ({}={:.3f})'.format(cfg.score_type, cfg.score(y,preds))
        fits[loo_label] = preds        
    plot_one_series(series, fits)
