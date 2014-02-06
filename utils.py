# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 10:38:26 2014

@author: ronnie
"""
from contextlib import contextmanager
from os import makedirs
import os.path
import matplotlib.pyplot as plt

@contextmanager
def interactive(b):
    b_prev = plt.isinteractive()
    plt.interactive(b)
    try:
        yield
    finally:
        plt.interactive(b_prev)

def ensure_dir(d):
    if not os.path.exists(d):
        makedirs(d)
