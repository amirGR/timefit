# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:49:32 2014

@author: ronnie
"""
from os.path import dirname, join, abspath

def base_dir():
    code_dir = dirname(__file__)
    return abspath(join(code_dir,'..'))

def code_dir():
    return join(base_dir(), 'code')

def resources_dir():
    return join(base_dir(), 'code', 'resources')
    
def data_dir():
    return join(base_dir(), 'data')

def cache_dir():
    return join(base_dir(), 'cache')

def results_dir():
    return join(base_dir(), 'results')
