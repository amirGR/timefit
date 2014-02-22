# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 11:04:35 2014

@author: ronnie
"""
from os.path import dirname, join, abspath
import sys

script_dir = dirname(__file__)
code_dir = abspath(join(script_dir,'..'))
sys.path.append(code_dir)

import utils
utils.disable_all_warnings()
