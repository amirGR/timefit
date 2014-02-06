# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:50:18 2014

@author: MineAllMine
"""
import setup
from all_fits import compute_fit
from load_data import load_data
from utils import interactive,ensure_dir
from plots import plot_one_series,save_figure

data = load_data(serotonin_only=False)
data.expression.shape
gene_symbols = ['CNR1','CNR2']
# comment
with interactive(False):
    for gene_name in gene_symbols:
        dirname = os.path.join(r'C:\work\HTR\results','{}'.format(gene_name)) 
        ensure_dir(dirname)
        for region_name in data.region_names:
            series = data.get_one_series(gene_name,region_name)
            fit = compute_fit(series)
            print 'Saving figure for {}@{}'.format(gene_name,region_name)
            fig = plot_one_series(series,fit=fit)
            filename = os.path.join(dirname, 'fit-{}-{}.png'.format(gene_name,region_name))
            save_figure(fig, filename, b_close=True)
            
        