from os.path import dirname, join, abspath
import sys

script_dir = dirname(__file__)
code_dir = abspath(join(script_dir,'..'))
sys.path.append(code_dir)

from utils.misc import disable_all_warnings
disable_all_warnings()
