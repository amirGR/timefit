import setup
from os.path import join
import config as cfg
import utils
from all_fits import get_all_fits
from command_line import get_common_parser, process_common_inputs
from plots import save_fits_and_create_html

def do_fits(data, fitter, html_dir):
    print """
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
""".format(fitter)
    get_all_fits(data,fitter)    
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

if __name__ == '__main__':
    utils.disable_all_warnings()
    parser = get_common_parser()
    parser.add_argument('--html', metavar='HTMLDIR', help='Create html for the fits under HTMLDIR')
    args = parser.parse_args()
    data, fitter = process_common_inputs(args)
    do_fits(data, fitter, args.html)
