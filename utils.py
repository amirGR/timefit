# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 10:38:26 2014

@author: ronnie
"""
from contextlib import contextmanager
import matplotlib.pyplot as plt

@contextmanager
def interactive(b):
    b_prev = plt.isinteractive()
    plt.interactive(b)
    yield
    plt.interactive(b_prev)
