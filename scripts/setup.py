from os.path import dirname, join, abspath
import sys

script_dir = dirname(__file__)
code_dir = abspath(join(script_dir,'..'))
sys.path.append(code_dir)

import utils
utils.disable_all_warnings()
