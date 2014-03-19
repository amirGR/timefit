import setup
import re
import sys
from os.path import join
import config as cfg
import utils
from all_fits import get_all_fits
from command_line import get_common_parser, process_common_inputs
from plots import save_fits_and_create_html

def do_fits(data, fitter, k_of_n, html_dir):
    print """
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
""".format(fitter)
    get_all_fits(data, fitter, k_of_n)    
    if html_dir is not None:
        basedir = join(html_dir, fitter.shape.cache_name()) 
        print """
==============================================================================================
==============================================================================================
==== Writing HTML to {}
==============================================================================================
==============================================================================================
""".format(basedir)
        save_fits_and_create_html(data, fitter, basedir)

def parse_k_of_n(s):
    """Parse a string that looks like "3/5" and return tuple (3,5)"""
    if s is None:
        return None
    m = re.match('(\d+)/(\d+)',s)
    if m is None:
        print '{} is not a valid part description. Format is k/n.'.format(s)
        sys.exit(-1)
    return tuple(int(x) for x in m.groups())

if __name__ == '__main__':
    utils.disable_all_warnings()
    parser = get_common_parser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--part', help='Compute only part of the genes. format: <k>/<n> e.g. 1/4. (k=1..n)')
    group.add_argument('--html', metavar='HTMLDIR', help='Create html for the fits under HTMLDIR')
    args = parser.parse_args()
    k_of_n = parse_k_of_n(args.part)
    data, fitter = process_common_inputs(args)
    do_fits(data, fitter, k_of_n, args.html)
