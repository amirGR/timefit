import setup
import re
import sys
from os.path import join
import config as cfg
import utils
from all_fits import get_all_fits, save_as_mat_file
from command_line import get_common_parser, process_common_inputs
from plots import save_fits_and_create_html
from project_dirs import cache_dir, fit_results_relative_path

def do_fits(data, fitter, k_of_n):
    print """
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
""".format(fitter)
    fits = get_all_fits(data, fitter, k_of_n)    
    return fits
    
def create_html(fits, html_dir, k_of_n):
    print """
==============================================================================================
==============================================================================================
==== Writing HTML
==============================================================================================
==============================================================================================
"""
    save_fits_and_create_html(data, fitter, html_dir, k_of_n=k_of_n)

def save_mat_file(fits,filename):
    if filename is None:
        filename = join(cache_dir(), fit_results_relative_path(data,fitter) + '.mat')
    print """
==============================================================================================
==============================================================================================
==== Saving matlab file
==============================================================================================
==============================================================================================
"""
    save_as_mat_file(fits, filename)

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
    NOT_USED = (None,)
    parser = get_common_parser()
    parser.add_argument('--part', help='Compute only part of the genes. format: <k>/<n> e.g. 1/4. (k=1..n)')
    parser.add_argument('--html', nargs='?', metavar='DIR', default=NOT_USED, help='Create html for the fits. Optionally override output directory.')
    parser.add_argument('--mat', nargs='?', metavar='FILENAME', default=NOT_USED, help='Save the fits also as matlab .mat file. Optionally override output filename.')
    args = parser.parse_args()
    if args.part is not None and args.mat != NOT_USED:
        print '--mat cannot be used with --part'
        sys.exit(-1)
    k_of_n = parse_k_of_n(args.part)
    data, fitter = process_common_inputs(args)
    fits = do_fits(data, fitter, k_of_n)
    if args.html != NOT_USED:
        create_html(fits, args.html, k_of_n)
    if args.mat != NOT_USED:
        save_mat_file(fits,args.mat)
